"""Does the canonical (SVD-ordered) GHMM of a model's language make its
forgetting readable in a few directions? (25.11.2026, exploratory)

From the Hankel matrix of the model's language (full enumeration, as in
hankel_spectrum.py) a weighted automaton of rank r is built by the standard
spectral recipe:

    H ~ U S V^T (rank r),  F = H V,  A_x = F^+ H_x V,  alpha = H[eps, :] V,
    beta = F^+ H[:, eps],  f(w) ~ alpha A_{w1} ... A_{wn} beta,

where H_x[u, v] = f(u x v). Option --weight b rescales every row by 1 / f(u)
(each prefix counts the same) before the SVD; the rescaling is applied to H and
H_x alike, so only which directions are kept changes.

The forgetting measure w(g): for prefixes after which the true automaton is in
state q, the next-symbol distribution d is projected on the segment from the
true emission e(q) to the emission of the "forgotten" state q' (parity: the
same group, the other bit, before the final 2; flipflop: the other reset):
w = <d - e(q), e(q') - e(q)> / |e(q') - e(q)|^2, averaged over prefixes.
The network itself is run only for the reference curve.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import hankel_spectrum as hs  # noqa: E402

from nerode.automata import PDFA  # noqa: E402
from nerode.training import DEVICE  # noqa: E402

N = 3


def strings_upto(n: int) -> list[tuple[int, ...]]:
    return [s for k in range(n + 1) for s in itertools.product(range(N), repeat=k)]


def index(s: tuple[int, ...]) -> int:
    i = 0
    for a in s:
        i = i * N + a
    return i


def f_of(lp, s) -> float:
    return float(np.exp(lp[len(s)][index(s)]))


def hankel_blocks(lp, max_u: int, max_v: int):
    """Rows: prefixes of length <= max_u - 1, columns: continuations of length
    <= max_v, both by (length, base-3 index), as in strings_upto."""
    def block(mid: int | None):
        rows = []
        for lu in range(max_u):
            cols = []
            for lv in range(max_v + 1):
                if mid is None:
                    cols.append(lp[lu + lv].reshape(N**lu, N**lv))
                else:
                    cols.append(lp[lu + 1 + lv].reshape(N**lu, N, N**lv)[:, mid, :])
            rows.append(np.concatenate(cols, 1))
        return np.exp(np.concatenate(rows))
    return block(None), [block(x) for x in range(N)]


def spectral(h, hx, r: int, weight: str):
    d = np.ones(len(h)) if weight == "a" else 1.0 / h[:, 0]  # column 0 is v = eps
    hw = h * d[:, None]
    _, _, vt = np.linalg.svd(hw, full_matrices=False)
    v = vt[:r].T
    f = hw @ v
    fp = np.linalg.pinv(f)
    a = [fp @ ((hx_x * d[:, None]) @ v) for hx_x in hx]
    alpha = h[0] @ v  # row 0 is u = eps (f(eps) = 1, weight 1)
    beta = fp @ hw[:, 0]
    return alpha, a, beta


def ghmm_next(alpha, a, beta, prefix) -> np.ndarray:
    s = alpha.copy()
    for x in prefix:
        s = s @ a[x]
    p = np.array([s @ a[x] @ beta for x in range(N)])
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def net_next(model, prefixes: list[tuple[int, ...]]) -> np.ndarray:
    out = []
    with torch.no_grad():
        for pre in prefixes:
            x = torch.tensor([[N, *pre]]).to(DEVICE)
            out.append(torch.softmax(model(x)[0, -1, :N].double(), -1).cpu().numpy())
    return np.array(out)


def run_pdfa(p: PDFA, s) -> int:
    q = 0
    for a in s:
        q = int(p.delta[q, a])
    return q


def probes(kind: str, p: PDFA, gmax: int):
    """Per gap g: list of (prefix, true state, forgotten state)."""
    out = {}
    for g in range(gmax + 1):
        items = []
        if kind.startswith("parity"):
            for gap in itertools.product((0, 1), repeat=g):
                pre = (1, 2, *gap)
                q = run_pdfa(p, pre)
                items.append(((*pre, 2), int(p.delta[q, 2]), int(p.delta[q ^ 1, 2])))
        else:
            for r in (1, 2):
                pre = (r, *([0] * g))
                q = run_pdfa(p, pre)
                items.append((pre, q, 1 - q))
        out[g] = items
    return out


def w_of(d: np.ndarray, e_true: np.ndarray, e_flip: np.ndarray) -> float:
    u = e_flip - e_true
    return float((d - e_true) @ u / (u @ u))


def curve(p, items_by_g, predict) -> list[float]:
    ws = []
    for g, items in items_by_g.items():
        ds = predict([it[0] for it in items])
        ws.append(np.mean([w_of(d, p.emit[t], p.emit[f]) for d, (_, t, f) in zip(ds, items)]))
    return ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="parity_hand_reset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--max-u", type=int, default=9)
    ap.add_argument("--max-v", type=int, default=4)
    ap.add_argument("--gmax", type=int, default=10)
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 3, 4, 6, 8, 10, 14, 20, 30, 40, 60])
    a = ap.parse_args()

    p, m = hs.load(a.kind, a.seed, a.layers, a.steps)
    lp_net = hs.tree_logp_model(m, a.max_u + a.max_v)
    lp_true = hs.tree_logp_pdfa(p, a.max_u + a.max_v)
    items = probes(a.kind, p, a.gmax)
    fmt = lambda ws: " ".join(f"{w:5.2f}" for w in ws)  # noqa: E731
    print(f"{a.kind} s{a.seed} {a.layers}L {a.steps // 1000}k, U{a.max_u} V{a.max_v}; w(g) for g = 0..{a.gmax}")
    print(f"{'net':>14s}  {fmt(curve(p, items, lambda ps: net_next(m, ps)))}", flush=True)
    for name, lp in (("true", lp_true), ("model", lp_net)):
        h, hx = hankel_blocks(lp, a.max_u, a.max_v)
        for weight in ("a", "b"):
            for r in a.ranks:
                if r > min(h.shape):
                    continue
                if name == "true" and r not in (2, 6):
                    continue
                al, aa, be = spectral(h, hx, r, weight)
                ws = curve(p, items, lambda ps: np.array([ghmm_next(al, aa, be, q) for q in ps]))
                print(f"{name:>5s} ({weight}) r={r:<3d} {fmt(ws)}", flush=True)


if __name__ == "__main__":
    main()
