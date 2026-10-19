"""Nerode geometry, version 2: compare prefixes through common continuations.

Version 1 (nerode_geometry.py) compared the residual vector at the last position
of prefix u with that of prefix v. A transformer does not have to keep the whole
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

Automata:
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

Run from the repo root:  uv run python experiments/nerode_continuations.py
Faster smoke run:        uv run python experiments/nerode_continuations.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "nerode_geometry", Path(__file__).with_name("nerode_geometry.py")
)
assert _spec is not None and _spec.loader is not None
ng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ng)

PDFA = ng.PDFA
DEVICE = ng.DEVICE


# ---------------------------------------------------------------- automata


def parity_pdfa(n_groups: int, rng: np.random.Generator) -> Any:
    """States 2g + b. Alphabet {0, 1, 2}: 0 flips b, 1 keeps it, 2 reads and
    resets it. Emissions depend on the group only."""
    n_sym = 3
    group_next = rng.integers(0, n_groups, size=(n_groups, n_sym))
    # where symbol 2 leads when b = 1: always a different group than for b = 0
    alt = (group_next[:, 2] + rng.integers(1, n_groups, size=n_groups)) % n_groups
    emit_g = rng.dirichlet(np.full(n_sym, 0.7), size=n_groups)
    emit_g = 0.85 * emit_g + 0.15 / n_sym
    n_states = 2 * n_groups
    delta = np.zeros((n_states, n_sym), dtype=np.int64)
    emit = np.zeros((n_states, n_sym))
    for g in range(n_groups):
        for b in range(2):
            q = 2 * g + b
            emit[q] = emit_g[g]
            delta[q, 0] = 2 * group_next[g, 0] + (1 - b)
            delta[q, 1] = 2 * group_next[g, 1] + b
            delta[q, 2] = 2 * (group_next[g, 2] if b == 0 else alt[g])
    return PDFA(delta, emit / emit.sum(1, keepdims=True))


# ---------------------------------------------------------------- model passes


@torch.no_grad()
def last_residuals(model, seqs: np.ndarray, batch: int = 4096) -> np.ndarray:
    """(n, T) token ids -> (n_layers + 2, n, d): last-position residual after the
    embedding, each block and the final norm."""
    out = []
    for i in range(0, len(seqs), batch):
        x = torch.tensor(seqs[i : i + batch])
        res = ng.residuals(model, x)
        out.append(np.stack([h[:, -1] for _, h in res]))
    return np.concatenate(out, axis=1)


@torch.no_grad()
def future_logprob(
    model, ctx: np.ndarray, futures: np.ndarray, n_sym: int, batch: int = 4096
) -> np.ndarray:
    """log P_model(z | ctx) for every context and every future z.

    ctx: (n, T) token ids; futures: (m, H). Returns (n, m)."""
    model.eval()
    n, m, h = len(ctx), len(futures), futures.shape[1]
    seqs = np.concatenate([np.repeat(ctx, m, axis=0), np.tile(futures, (n, 1))], axis=1)
    t0 = ctx.shape[1]
    out = np.zeros(len(seqs))
    for i in range(0, len(seqs), batch):
        x = torch.tensor(seqs[i : i + batch]).to(DEVICE)
        logits = model(x[:, :-1]).float()[..., :n_sym]
        logp = torch.log_softmax(logits, -1)  # renormalised over real symbols
        # token at position t0 + k is predicted at position t0 + k - 1
        idx = x[:, t0:]
        lp = logp[:, t0 - 1 : t0 - 1 + h].gather(-1, idx[..., None])[..., 0]
        out[i : i + batch] = lp.sum(-1).cpu().numpy()
    return out.reshape(n, m)


# ---------------------------------------------------------------- statistics


def ranks(x: np.ndarray) -> np.ndarray:
    return x.argsort().argsort().astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    c = np.corrcoef(np.stack([rx, ry, rz]))
    rxy, rxz, ryz = c[0, 1], c[0, 2], c[1, 2]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2)))


def cosine_dist_matrix(h: np.ndarray) -> np.ndarray:
    """h: (n, d) -> (n, n)."""
    hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-9)
    return 1.0 - hn @ hn.T


def tv_matrix(p: np.ndarray) -> np.ndarray:
    """p: (n, k) distributions -> (n, n) total variation."""
    return 0.5 * np.abs(p[:, None, :] - p[None, :, :]).sum(-1)


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


# ---------------------------------------------------------------- experiment


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
) -> dict:
    rng = np.random.default_rng(seed)
    if kind == "parity":
        p = parity_pdfa(3, rng)
    else:
        p = ng.random_pdfa(6, 3, rng)
    n_sym = p.n_symbols
    bos = n_sym
    cls_of_state = ng.minimise(p)

    syms, states = ng.sample(p, 6000, seq_len, rng)
    x = torch.tensor(np.concatenate([np.full((len(syms), 1), bos), syms], 1))
    h_opt = ng.entropy_rate(p, states)

    # prefixes: distinct, same length, sampled from the automaton
    psyms, pstates = ng.sample(p, 4 * n_prefix, prefix_len, rng)
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
    if kind == "parity":
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
        model = ng.make_model(n_sym, seq_len, 64, 2, seed)
        if model_kind == "trained":
            ng.train(model, x[:5000], steps=steps, lr=3e-3)
        loss = ng.eval_loss(model, x[5000:])

        # internal states without continuation (version 1) and with each w
        h0 = last_residuals(model, prefixes)  # (L, n, d)
        n_layers = h0.shape[0]
        d_int = np.zeros((n_layers, n, n))
        d_beh = np.zeros((n, n))
        d_next = np.zeros((n, n))
        d_far = np.zeros((n, n))
        for w in ws:
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
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=["plain", "parity"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-prefix", type=int, default=300)
    ap.add_argument("--prefix-len", type=int, default=12)
    ap.add_argument("--w-len", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--out", default="experiments/nerode_out/continuations.txt")
    ap.add_argument(
        "--w-symbols",
        type=int,
        nargs="+",
        default=None,
        help="restrict continuations to these symbols; e.g. 0 1 keeps the parity "
        "of parity automata hidden and irrelevant to the next symbol",
    )
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.steps, a.n_prefix = [0], 200, 60

    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(
        f"device {DEVICE}; prefixes of length {a.prefix_len}, continuations of "
        f"length {a.w_len}, behaviour horizon {a.horizon}, {a.steps} steps"
    )
    for kind in a.kinds:
        for seed in a.seeds:
            t0 = time.time()
            r = run_one(
                kind,
                seed,
                a.n_prefix,
                a.prefix_len,
                a.w_len,
                a.horizon,
                a.steps,
                w_symbols=a.w_symbols,
            )
            emit(
                f"\n{kind} PDFA, seed {seed}: {r['n_cls']} Nerode classes, "
                f"{r['n_prefix']} prefixes, optimal loss {r['h_opt']:.3f} "
                f"({time.time() - t0:.0f}s)"
            )
            emit(
                f"  {'model':8s} {'layer':7s} {'loss':>6s} {'rho_beh':>8s} "
                f"{'rho_far|nx':>10s} {'rho_w0':>7s} {'hid':>6s} {'hid_w0':>7s}"
            )
            for row in r["rows"]:
                emit(
                    f"  {row['model']:8s} {row['layer']:7s} {row['loss']:6.3f} "
                    f"{row['rho_beh']:8.3f} {row['rho_far']:10.3f} "
                    f"{row['rho_beh_w0']:7.3f} {row['hid']:6.3f} "
                    f"{row['hid_w0']:7.3f}"
                )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
