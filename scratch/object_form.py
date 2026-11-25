"""Form of the exact one-layer object S(p) = H u(p) (24.11.2026, exploratory).

H from the weights (reads_matrix, project + normalize). Prefixes sampled from
the automaton are used only to place points and colour them by the true state.
Window: prefix = BOS + 16 symbols (17 positions), future offsets tau <= 8, so
every distance stays inside the trained context (max position 24).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from nerode.automata import make_automaton, sample
from nerode.probes import SEQ_LEN
from nerode.reads import one_hot_prefix, reads_matrix
from nerode.training import DEVICE, ckpt_path, make_model

N_SYM = 16
N_LAGS = N_SYM + 1
N_AHEAD = SEQ_LEN - N_SYM  # 24 - 16 = 8


def load(kind: str, seed: int):
    p = make_automaton(kind, seed, 0.1)
    m = make_model(p.n_symbols, SEQ_LEN, 64, 1, seed)
    path = ckpt_path(m, p, torch.zeros((1, SEQ_LEN + 1), dtype=torch.long),
                     steps=100_000, seed=seed, seq_len=SEQ_LEN, batch=128, fresh=True)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    twin = make_model(p.n_symbols, SEQ_LEN, 64, 1, seed)
    return p, m.cpu().eval(), twin.cpu().eval()


def state_share(S: np.ndarray, lab: np.ndarray) -> float:
    mu = S.mean(0)
    tot = ((S - mu) ** 2).sum()
    if not np.isfinite(tot) or tot <= 1e-10 * len(S) * (float((S ** 2).mean()) + 1e-300):
        return float("nan")  # (almost) constant: the share is undefined
    btw = sum((lab == c).sum() * ((S[lab == c].mean(0) - mu) ** 2).sum() for c in np.unique(lab))
    return float(btw / tot)


def main() -> None:
    kinds = sys.argv[1:] or ["parity_hand_reset", "parity_hand", "flipflop"]
    for kind in kinds:
        for seed in (0, 1, 2):
            p, m, twin = load(kind, seed)
            syms, states = sample(p, 3000, N_SYM, np.random.default_rng(7 + seed))
            bos = p.n_symbols
            U = np.stack([one_hot_prefix([bos, *s], N_LAGS, p.n_symbols + 1) for s in syms])
            lab = states[:, N_SYM]
            for name, net in (("trained", m), ("twin", twin)):
                H = reads_matrix(net, N_LAGS, N_AHEAD, range(p.n_symbols), project=True, normalize=True)
                sv = np.linalg.svd(H, compute_uv=False)
                rel = sv / sv[0]
                S = U @ H.T
                print(f"{kind} s{seed} {name:7s} H {H.shape} sv/sv0: "
                      + " ".join(f"{r:.1e}" for r in rel[:12])
                      + f"  | ratios r_k/r_k+1 max at k={1 + int(np.argmax(rel[:20] / rel[1:21]))}"
                      + f"  | state share {state_share(S, lab):.3f}", flush=True)


if __name__ == "__main__":
    main()
