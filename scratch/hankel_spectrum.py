"""Spectrum of the Hankel matrix of a model's language (24.11.2026, exploratory).

The minimal linear generator of a language (a GHMM / weighted automaton) has
dimension equal to the rank of its Hankel matrix: rows are prefixes u, columns
are continuations v, the entry is the probability of the string uv (Fliess).
Here the matrix is built by full enumeration, so no texts are chosen:

    rows    all prefixes of length <= MAX_U   (strings after BOS)
    columns all continuations of length <= MAX_V
    (a) entry P(uv)       -- weighted by the model's own measure
    (b) entry P(v | u)    -- every prefix counts the same

The probabilities come from one forward pass over every string of length
MAX_U + MAX_V - 1 (the output at each position is the next-symbol distribution
of the prefix up to it), so every node of the prefix tree is covered exactly.
For the true automata the same tree is filled from the PDFA.

This computes the target from the model's outputs, not from the weights: it is
the yardstick a weights-only bridge has to reproduce, not the bridge itself.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from nerode.automata import PDFA, make_automaton
from nerode.probes import SEQ_LEN
from nerode.training import DEVICE, ckpt_path, make_model

MAX_U = 9
MAX_V = 4
N_SYM = 3


def all_strings(length: int) -> np.ndarray:
    """All strings over {0,1,2} of this length, in base-3 order (first symbol
    most significant), shape (3**length, length)."""
    if length == 0:
        return np.zeros((1, 0), dtype=np.int64)
    return np.array(list(itertools.product(range(N_SYM), repeat=length)), dtype=np.int64)


def tree_logp_model(model, depth: int, batch: int = 8192) -> list[np.ndarray]:
    """lp[d][i] = log P(string with base-3 index i of length d), d = 0..depth."""
    model.eval()
    n = depth - 1  # strings of this length give distributions at nodes 0..n
    s = all_strings(n)
    dist = [np.zeros((N_SYM**d, N_SYM)) for d in range(n + 1)]
    with torch.no_grad():
        for i in range(0, len(s), batch):
            chunk = s[i : i + batch]
            x = torch.tensor(np.concatenate([np.full((len(chunk), 1), N_SYM), chunk], 1))
            logits = model(x.to(DEVICE))[..., :N_SYM].double()
            lpm = torch.log_softmax(logits, -1).cpu().numpy()  # (b, n+1, 3)
            for d in range(n + 1):
                # node of depth d = first d symbols; its index in base 3
                idx = np.zeros(len(chunk), dtype=np.int64)
                for t in range(d):
                    idx = idx * N_SYM + chunk[:, t]
                dist[d][idx] = lpm[:, d]
    return _accumulate(dist, depth)


def tree_logp_pdfa(p: PDFA, depth: int) -> list[np.ndarray]:
    dist = []
    states = np.zeros(1, dtype=np.int64)
    for d in range(depth):
        dist.append(np.log(p.emit[states]))
        states = p.delta[np.repeat(states, N_SYM), np.tile(np.arange(N_SYM), len(states))]
    return _accumulate(dist, depth)


def _accumulate(dist: list[np.ndarray], depth: int) -> list[np.ndarray]:
    lp = [np.zeros(1)]
    for d in range(depth):
        lp.append((lp[d][:, None] + dist[d]).reshape(-1))
    return lp


def hankel(lp: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(a) P(uv) and (b) P(v|u); rows ordered by (|u|, index), columns by (|v|, index)."""
    rows_a, rows_b = [], []
    for lu in range(MAX_U + 1):
        nu = N_SYM**lu
        cols = []
        for lv in range(MAX_V + 1):
            nv = N_SYM**lv
            cols.append(lp[lu + lv].reshape(nu, nv))  # index(uv) = index(u)*3^|v| + index(v)
        luv = np.concatenate(cols, 1)
        rows_a.append(np.exp(luv))
        rows_b.append(np.exp(luv - lp[lu][:, None]))
    return np.concatenate(rows_a), np.concatenate(rows_b)


def spectrum(h: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(h, compute_uv=False)
    return s / s[0]


def load(kind: str, seed: int, layers: int, steps: int):
    p = make_automaton(kind, seed, 0.1)
    m = make_model(p.n_symbols, SEQ_LEN, 64, layers, seed)
    path = ckpt_path(
        m, p, torch.zeros((1, SEQ_LEN + 1), dtype=torch.long),
        steps=steps, seed=seed, seq_len=SEQ_LEN, batch=128, fresh=True,
    )
    if not path.exists():
        return p, None
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return p, m


def check_tree(model, lp, rng, n: int = 200) -> float:
    """Max |log P| difference between the tree and a direct forward pass on
    random strings of length MAX_U + MAX_V (index bookkeeping check)."""
    L = MAX_U + MAX_V
    s = rng.integers(0, N_SYM, size=(n, L))
    x = torch.tensor(np.concatenate([np.full((n, 1), N_SYM), s], 1))
    with torch.no_grad():
        logits = model(x[:, :-1].to(DEVICE))[..., :N_SYM].double()
        lpm = torch.log_softmax(logits, -1).cpu().numpy()
    direct = np.take_along_axis(lpm, s[..., None], -1)[..., 0].sum(1)
    idx = np.zeros(n, dtype=np.int64)
    for t in range(L):
        idx = idx * N_SYM + s[:, t]
    return float(np.abs(direct - lp[L][idx]).max())


def report(name: str, h_a: np.ndarray, h_b: np.ndarray, k: int, show: int = 14) -> None:
    for tag, h in (("a", h_a), ("b", h_b)):
        s = spectrum(h)
        tail = float((s[k:] ** 2).sum() / (s**2).sum())
        head = " ".join(f"{v:.1e}" for v in s[:show])
        print(f"{name:34s} ({tag}) k={k} s_k+1/s_1={s[k]:.2e} tail={tail:.2e} | {head}", flush=True)


def main() -> None:
    global MAX_U, MAX_V
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--max-u", type=int, default=MAX_U)
    ap.add_argument("--max-v", type=int, default=MAX_V)
    ap.add_argument("--only", nargs="*", default=None, help="kinds to run")
    a = ap.parse_args()
    MAX_U, MAX_V = a.max_u, a.max_v
    rng = np.random.default_rng(0)
    depth = MAX_U + MAX_V
    kinds = ["flipflop", "parity_hand_reset", "parity_hand"]
    true_rank = {}
    for kind in (a.only or kinds):
        p = make_automaton(kind, 0, 0.1)
        h_a, h_b = hankel(tree_logp_pdfa(p, depth))
        s = spectrum(h_a)
        k = int((s > 1e-10).sum())
        true_rank[kind] = k
        report(f"TRUE {kind}", h_a, h_b, k)
    if a.ladder:
        runs = [(kind, 0, 1, st) for kind in ("flipflop", "parity_hand_reset")
                for st in (3000, 10000, 30000, 100000)]
    else:
        runs = [(kind, seed, 1, 100000) for kind in kinds for seed in (0, 1, 2)]
        runs += [(kind, seed, 2, 100000) for kind in ("parity_hand_reset", "parity_hand")
                 for seed in (0, 1, 2)]
    if a.only:
        runs = [r for r in runs if r[0] in a.only]
    for kind, seed, layers, steps in runs:
        p, m = load(kind, seed, layers, steps)
        name = f"{kind} s{seed} {layers}L {steps // 1000}k"
        if m is None:
            print(f"{name:34s} -- no checkpoint", flush=True)
            continue
        lp = tree_logp_model(m, depth)
        err = check_tree(m, lp, rng)
        h_a, h_b = hankel(lp)
        report(name + f" [chk {err:.0e}]", h_a, h_b, true_rank[kind])


if __name__ == "__main__":
    main()
