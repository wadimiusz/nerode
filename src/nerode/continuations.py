"""Nerode geometry, version 2: compare prefixes through common continuations.

Version 1 (geometry.py) compared the residual vector at the last position of
prefix u with that of prefix v. A transformer does not have to keep the whole
Myhill-Nerode state in that one vector: later positions can read earlier ones
through attention, so the state is the whole cache. This version compares what
is *readable* from the cache, which is also how the Nerode relation is defined:
u ~ v iff uw and vw behave the same for every continuation w.

For every pair of same-length prefixes (u, v) and every continuation w of
length W (all |Sigma|^W of them, no sampling):

    D_int(u, v) = mean_w cosine_distance(h(uw), h(vw))
                  h = residual stream at the last position of BOS+u+w, per layer
    D_beh(u, v) = mean_w TV(P_model(z | uw), P_model(z | vw)),  z in Sigma^H

Behaviour is the model's own (its own Nerode classes), not the automaton's.
At the final layer the next-token distribution is a linear readout of h, so a
correlation between D_int and the next-symbol part of D_beh is partly forced.
We therefore also report the partial rank correlation with the "far" part of the
future (symbols 2..H after w) controlling for the next-symbol distance.

Automata (see automata.py):
- plain:  random PDFA, every state has its own next-symbol distribution;
- parity: states are (group g, parity bit b). Symbol 0 flips b, symbol 2 reads
          b (the next group depends on it) and resets it. b changes nothing about
          the next symbol and is not revealed by the last token, so telling
          (g, 0) from (g, 1) needs memory of the whole prefix.

Metrics per layer (pairs of same-length prefixes):
- rho_beh:    Spearman(D_int, D_beh)
- rho_far|nx: partial Spearman(D_int, D_far | D_next)
- rho_beh_w0: Spearman(D_int computed WITHOUT continuations, D_beh): version 1
- hid:        parity automata only. Within prefixes ending in the same symbol
              (the last symbol leaks parity), compare mean D_int of pairs that
              share the group but differ in parity with same-class pairs and
              with different-group pairs: (hidden - same) / (different - same).
              0: the hidden parity is invisible in the geometry; 1: it separates
              prefixes as much as a different group does. Reported with
              continuations and without them (hid_w0). With continuations the
              "embed" row is degenerate: it only sees the last symbol of w.

Command line: scripts/nerode_continuations.py.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from nanogpt.model import GPT
from tqdm.auto import tqdm

from nerode.automata import (
    entropy_rate,
    group_separation,
    hand_parity_pdfa,
    minimise,
    parity_pdfa,
    random_pdfa,
    sample,
)
from nerode.readout import future_logprob, last_residuals, parity_bit_accuracy
from nerode.stats import cosine_dist_matrix, partial_spearman, spearman, tv_matrix
from nerode.training import eval_loss, fit, make_model


def hidden_score(
    d: np.ndarray, strata: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> float:
    """(hidden - same class) / (different group - same class), averaged over
    last-symbol strata weighted by the number of hidden-parity pairs.
    0: parity invisible in the geometry; 1: as far apart as different groups."""
    scores, weights = [], []
    for same, hidden, diff in strata:
        m_same, m_hid, m_diff = d[same].mean(), d[hidden].mean(), d[diff].mean()
        if m_diff - m_same > 1e-6:
            scores.append((m_hid - m_same) / (m_diff - m_same))
            weights.append(hidden.sum())
    return float(np.average(scores, weights=weights)) if scores else float("nan")


def run_one(
    kind: str,
    seed: int,
    n_prefix: int,
    prefix_len: int,
    w_len: int,
    horizon: int,
    steps: int,
    seq_len: int = 24,
    w_symbols: list[int] | None = None,
    fresh: bool = False,
    batch: int = 128,
    min_sep: float = 0.0,
) -> dict:
    rng = np.random.default_rng(seed)
    if kind == "parity":
        p = parity_pdfa(3, rng, min_sep=min_sep)
    elif kind == "parity_hand":
        p = hand_parity_pdfa()
    else:
        p = random_pdfa(6, 3, rng)
    n_sym = p.n_symbols
    bos = n_sym
    cls_of_state = minimise(p)

    syms, states = sample(p, 6000, seq_len, rng)
    x = torch.tensor(np.concatenate([np.full((len(syms), 1), bos), syms], 1))
    h_opt = entropy_rate(p, states)

    # prefixes: distinct, same length, sampled from the automaton
    psyms, pstates = sample(p, 4 * n_prefix, prefix_len, rng)
    _, keep = np.unique(psyms, axis=0, return_index=True)
    keep = np.sort(keep)[:n_prefix]
    psyms, pstate = psyms[keep], pstates[keep, -1]
    pcls = cls_of_state[pstate]
    n = len(psyms)
    prefixes = np.concatenate([np.full((n, 1), bos), psyms], 1)

    w_alphabet = w_symbols if w_symbols is not None else list(range(n_sym))
    ws = np.array(list(itertools.product(w_alphabet, repeat=w_len)))
    zs = np.array(list(itertools.product(range(n_sym), repeat=horizon)))
    iu = np.triu_indices(n, 1)

    # pair categories for the parity metric, stratified by the last symbol of
    # the prefix (the last symbol leaks parity: after a 2 it is always 0)
    strata: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if kind.startswith("parity"):
        g = pstate // 2
        same_cls = pcls[iu[0]] == pcls[iu[1]]
        same_g = g[iu[0]] == g[iu[1]]
        last0, last1 = psyms[iu[0], -1], psyms[iu[1], -1]
        for sym in range(n_sym):
            in_s = (last0 == sym) & (last1 == sym)
            masks = (in_s & same_cls, in_s & same_g & ~same_cls, in_s & ~same_g)
            if all(mk.sum() >= 5 for mk in masks):
                strata.append(masks)

    rows = []
    for model_kind in ("random", "trained"):
        model = make_model(n_sym, seq_len, 64, 2, seed)
        if model_kind == "trained":
            monitor = None
            if kind.startswith("parity"):

                def monitor(m: GPT) -> dict[str, str]:
                    acc = parity_bit_accuracy(m, p)
                    return {"bit@3": f"{acc[3]:.3f}", "bit@7": f"{acc[7]:.3f}"}

            fit(
                model,
                p,
                x[:5000],
                steps=steps,
                seed=seed,
                seq_len=seq_len,
                batch=batch,
                fresh=fresh,
                desc=f"{kind} seed {seed}",
                h_opt=h_opt,
                monitor=monitor,
            )
        loss = eval_loss(model, x[5000:])
        if model_kind == "trained" and kind.startswith("parity"):
            bit_acc = parity_bit_accuracy(model, p)

        # internal states without continuation (version 1) and with each w
        h0 = last_residuals(model, prefixes)  # (L, n, d)
        n_layers = h0.shape[0]
        d_int = np.zeros((n_layers, n, n))
        d_beh = np.zeros((n, n))
        d_next = np.zeros((n, n))
        d_far = np.zeros((n, n))
        for w in tqdm(
            ws, desc=f"  analysis ({model_kind})", leave=False, dynamic_ncols=True
        ):
            ctx = np.concatenate([prefixes, np.tile(w, (n, 1))], 1)
            hw = last_residuals(model, ctx)
            for layer in range(n_layers):
                d_int[layer] += cosine_dist_matrix(hw[layer])
            pz = np.exp(future_logprob(model, ctx, zs, n_sym))  # (n, |zs|)
            pz /= pz.sum(1, keepdims=True)
            d_beh += tv_matrix(pz)
            pz3 = pz.reshape(n, n_sym, -1)
            d_next += tv_matrix(pz3.sum(2))  # first symbol after w
            d_far += tv_matrix(pz3.sum(1))  # symbols 2..H after w
        d_int /= len(ws)
        for mat in (d_beh, d_next, d_far):
            mat /= len(ws)

        names = ["embed"] + [f"block{i}" for i in range(n_layers - 2)] + ["norm_f"]
        for layer, name in enumerate(names):
            di, di0 = d_int[layer][iu], cosine_dist_matrix(h0[layer])[iu]
            row = dict(
                model=model_kind,
                layer=name,
                loss=loss,
                rho_beh=spearman(di, d_beh[iu]),
                rho_far=partial_spearman(di, d_far[iu], d_next[iu]),
                rho_beh_w0=spearman(di0, d_beh[iu]),
                hid=float("nan"),
                hid_w0=float("nan"),
            )
            if strata:
                row["hid"] = hidden_score(di, strata)
                row["hid_w0"] = hidden_score(di0, strata)
            rows.append(row)
    return dict(
        kind=kind,
        seed=seed,
        n_cls=int(cls_of_state.max()) + 1,
        n_prefix=n,
        h_opt=h_opt,
        rows=rows,
        sep=group_separation(p.emit[0::2]) if kind.startswith("parity") else None,
        bit_acc=bit_acc if kind.startswith("parity") else None,
    )
