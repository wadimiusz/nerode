"""Test 3.4 on one-layer flipflop: lengthen the memory by editing weights (24.11.2026).

From test 3.3 the horizon is where the reset's score s_r(n) in the memory head drops below the
log-mass of the rest. Edit: raise every reset's key score by about c at all distances, in the memory
head only, through the slowest RoPE pair (angle ~1.8e-4 per position, so cos ~ 1 over the context).
W_K gets a rank-one change that moves the keys of tokens 1 and 2 and leaves 0 and BOS unchanged
(their normalised embeddings are linearly independent). c is the smallest value for which the
formula predicts horizon >= TARGET; the formula uses only the edited score table. The network is
run afterwards to check the prediction, and on automaton strings to check nothing else broke.
"""
from __future__ import annotations

import copy
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from forget_from_F import BOS, TRUE, lse  # noqa: E402
from object_form import load  # noqa: E402

from nerode.automata import entropy_rate, sample  # noqa: E402
from nerode.reads import scores, token_tables  # noqa: E402

MEM = {0: 1, 2: 0}  # memory head per seed (test 3.3)
TARGET = 14


def margins(sc, h, r, max_n=22):
    """s_r(n) - F_rest(n) in head h for the prefix BOS, r, n zeros."""
    out = []
    for n in range(max_n + 1):
        a = 0 if n else r
        rest = [sc[h, a, 0, j] for j in range(n)] + [sc[h, a, BOS, n + 1]]
        out.append(sc[h, a, r, n] - lse(rest))
    return np.array(out)


def horizon(sc, h, max_n=22):
    for n in range(max_n + 1):
        if min(margins(sc, h, r, max_n)[n] for r in (1, 2)) < 0:
            return n
    return None


@torch.no_grad()
def edited(m, h, c):
    m2 = copy.deepcopy(m)
    attn, blk = m2.blocks[0].attn, m2.blocks[0]
    dh = attn.d_head
    z = blk.norm1(m2.tok_emb.weight).double().numpy()  # (4, 64)
    w = np.linalg.pinv(z) @ np.array([0.0, 1.0, 1.0, 0.0])  # w.z_x = 1 for resets, 0 otherwise
    q, _, _ = token_tables(m2)
    # two slowest RoPE pairs (cos ~ 1 over the context); solve Re(q_a . conj(delta)) = c sqrt(dh)
    # for every query token a in {0, 1, 2}: 3 equations, 4 real unknowns, least norm
    cols = [dh - 4, dh - 3, dh - 2, dh - 1]  # (re, im) of pairs dh/2 - 2 and dh/2 - 1
    A = q[h, :3][:, cols]  # Re(q conj d) = q_re d_re + q_im d_im per pair
    dsol = np.linalg.lstsq(A, np.full(3, c * np.sqrt(dh)), rcond=None)[0]
    dk = np.zeros(attn.n_heads * dh)
    dk[h * dh + np.array(cols)] = dsol
    attn.W_K.weight += torch.tensor(np.outer(dk, w), dtype=attn.W_K.weight.dtype)
    return m2


@torch.no_grad()
def gap(p, m, seed):
    syms, st = sample(p, 4000, 23, np.random.default_rng(10**6 + seed))
    x = torch.tensor(np.concatenate([np.full((len(syms), 1), BOS), syms], 1))
    logp = torch.log_softmax(m(x).double(), -1).numpy()[:, :-1, :3]
    e = p.emit[st[:, :-1]]
    return float((e * (np.log(e) - logp)).sum(-1).mean())  # mean KL(true || model)


@torch.no_grad()
def main() -> None:
    for seed, h in MEM.items():
        p, m, _ = load("flipflop", seed)
        base = horizon(scores(m, 24), h)
        c = next(c for c in range(0, 3001, 5)
                 if min(margins(scores(edited(m, h, c), 24), h, r)[:TARGET].min() for r in (1, 2)) >= 5)
        m2 = edited(m, h, c)
        pred = horizon(scores(m2, 24), h)
        obs = []
        for r in (1, 2):
            net = torch.softmax(m2(torch.tensor([[BOS, r] + [0] * 22])).double(), -1)[0].numpy()
            tv = [0.5 * np.abs(net[1 + n, :3] - TRUE[r]).sum() for n in range(23)]
            obs.append(next((n for n in range(23) if tv[n] > 0.01), None))
            w = 1 / (1 + np.exp(-np.clip(margins(scores(m2, 24), h, r), -50, 50)))
            print(f"   r={r}: reset weight from weights by n: " + " ".join(f"{x:.2f}" for x in w)
                  + "\n        network TV by n:             " + " ".join(f"{x:.2f}" for x in tv))
        print(f"seed {seed}, head {h}: horizon {base} -> predicted {pred} with c = {c}; "
              f"edited network forgets at n = {obs} (r = 1, 2); "
              f"mean KL to truth on automaton strings {gap(p, m, seed):.4f} -> {gap(p, m2, seed):.4f}", flush=True)


if __name__ == "__main__":
    main()
