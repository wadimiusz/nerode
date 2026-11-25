"""How a one-layer parity_hand_reset model gets the bit (24.11.2026, exploratory).

In one layer a value depends on its token only, so a head's read is a mixture of the four value
vectors with weights pi = (pi_BOS, pi_0, pi_1, pi_2), and the network's output at a position is a
function of the current token and the two heads' pi. Strings "BOS 1 2 [gap] 2", gap = all 0/1
strings of length g: at the last 2 the model must add 1 + bit to the group, bit = parity of zeros
in the gap. pi comes from the score table (weights); the network is run on the same strings only
to check what it actually predicts there.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from object_form import load  # noqa: E402

from nerode.reads import scores  # noqa: E402

BOS = 3


def auc(a, b):
    """P(random a > random b) + 0.5 P(tie)."""
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None]).mean()
    eq = (a[:, None] == b[None]).mean()
    return gt + 0.5 * eq


@torch.no_grad()
def main(kind: str = "parity_hand_reset") -> None:
    for seed in (0, 1, 2):
        p, m, _ = load(kind, seed)
        sc = scores(m, 24)
        H = sc.shape[0]
        print(f"== {kind} seed {seed}")
        for g in range(1, 11):
            gaps = np.array(list(itertools.product((0, 1), repeat=g)))
            strings = np.array([[BOS, 1, 2, *gp, 2] for gp in gaps])
            t = strings.shape[1] - 1
            pi = np.zeros((len(strings), H, 4))
            for i, s in enumerate(strings):
                d = t - np.arange(t + 1)
                for h in range(H):
                    lg = sc[h, 2, s, d]
                    w = np.exp(lg - lg.max()); w /= w.sum()
                    np.add.at(pi[i, h], s, w)
            bit = (gaps == 0).sum(1) % 2
            # true state before the last 2 and the network's error vs the flipped-bit alternative
            q = np.zeros(len(strings), dtype=int)
            for j in range(1, t):
                q = p.delta[q, strings[:, j]]
            right = p.emit[p.delta[q, 2]]
            wrong = p.emit[p.delta[q ^ 1, 2]]  # same group, other bit
            out = torch.softmax(m(torch.tensor(strings)).double(), -1)[:, t, :3].numpy()
            d_r = np.abs(out - right).sum(1); d_w = np.abs(out - wrong).sum(1)
            acc = (d_r < d_w).mean()
            # which pi component separates the bit
            seps = {f"h{h}:pi{'012B'[x]}": abs(auc(pi[bit == 1, h, x], pi[bit == 0, h, x]) - 0.5) * 2
                    for h in range(H) for x in range(4)}
            best = sorted(seps.items(), key=lambda kv: -kv[1])[:3]
            # nearest neighbour across the bit in 6-dim pi space (BOS dropped: pi sums to 1)
            X = pi[:, :, :3].reshape(len(strings), -1)
            D = np.linalg.norm(X[:, None] - X[None], axis=-1); np.fill_diagonal(D, np.inf)
            nn_other = (bit[D.argmin(1)] != bit).mean()
            # arrangement vs count: spread of pi_0 of the best head within fixed counts vs overall
            h_best = int(best[0][0][1])
            n0 = (gaps == 0).sum(1)
            within = np.mean([pi[n0 == k, h_best, 1].std() for k in np.unique(n0) if (n0 == k).sum() > 1] or [0])
            print(f"  g={g:2d}: net picks right bit {acc:.2f}; nn across bit {nn_other:.2f}; "
                  f"best separators (|2AUC-1|): " + ", ".join(f"{k} {v:.2f}" for k, v in best)
                  + f"; pi0 of h{h_best}: std within count {within:.3f}, overall {pi[:, h_best, 1].std():.3f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
