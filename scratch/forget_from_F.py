"""Test 3.3 on one-layer flipflop: the forgetting horizon from the log-mass object F (24.11.2026).

Prefix "BOS, reset r, n zeros". In each head, at the last position, the reset enters the read with
score s_r(n) = score[h, a, r, n] and everything else with log-mass
F_rest(n) = logsumexp(score[h, a, 0, j] for j < n, score[h, a, BOS, n + 1]),
where a is the token at the reading position (0, or r when n = 0). The reset's weight in the read is
sigmoid(s_r - F_rest); the predicted horizon n* is the first n with s_r(n) < F_rest(n) in the memory
head. All of this is from the score table (weights). The network is run on the same strings only to
check: TV between its next-token distribution and the true state's emission, per n.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from object_form import load  # noqa: E402

from nerode.reads import scores  # noqa: E402

TRUE = {1: np.array([0.60, 0.28, 0.12]), 2: np.array([0.60, 0.12, 0.28])}  # state after reset 1 / 2
BOS = 3


def lse(x):
    m = max(x)
    return m + np.log(sum(np.exp(v - m) for v in x))


@torch.no_grad()
def main(kind: str = "flipflop") -> None:
    for seed in (0, 1, 2):
        _, m, _ = load(kind, seed)
        sc = scores(m, 24)  # (h, a, x, d)
        ns = range(0, 23)
        print(f"== {kind} seed {seed}")
        for r in (1, 2):
            inp = torch.tensor([[BOS, r] + [0] * n for n in [22]])
            net = torch.softmax(m(inp).double(), -1)[0].numpy()  # positions 1.. = after r + n zeros
            tv = [0.5 * np.abs(net[1 + n, :3] - TRUE[r]).sum() for n in ns]
            observed = next((n for n in ns if tv[n] > 0.01), None)
            line = [f" r={r}: net TV>0.01 from n={observed}; TV by n: " + " ".join(f"{t:.3f}" for t in tv[:14])]
            for h in range(sc.shape[0]):
                margin, srs, frs = [], [], []
                for n in ns:
                    a = 0 if n else r
                    s_r = sc[h, a, r, n]
                    rest = [sc[h, a, 0, j] for j in range(n)] + [sc[h, a, BOS, n + 1]]
                    f = lse(rest)
                    margin.append(s_r - f); srs.append(s_r); frs.append(f)
                nstar = next((n for n in ns if margin[n] < 0), None)
                line.append(f"   head {h}: n*={nstar}; s_r - F_rest by n: " + " ".join(f"{x:.0f}" for x in margin[:14])
                            + f" | s_r: " + " ".join(f"{x:.0f}" for x in srs[:12])
                            + f" | F_rest: " + " ".join(f"{x:.0f}" for x in frs[:12]))
            print("\n".join(line), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
