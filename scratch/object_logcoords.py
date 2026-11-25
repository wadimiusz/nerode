"""The exact one-layer object in log coordinates (24.11.2026, exploratory).

For every future read (head h, query token a, offset tau) a prefix p enters
only through its numerator N_p and denominator D_p. Here they are written as
r_p = N_p / D_p (the prefix's own average read, W_O-projected, 64 numbers) and
log D_p (its log mass). Both come from the weight tables (scores, values);
prefixes sampled from the automaton only place the points and give the true
state for colouring. Window as in object_form.py: BOS + 16 symbols, tau <= 8.
"""
from __future__ import annotations

import sys

import numpy as np

from nerode.automata import sample
from nerode.reads import _value_rows, scores

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from object_form import N_AHEAD, N_SYM, load, state_share  # noqa: E402


def log_coords(model, seqs: np.ndarray, n_sym: int):
    """r[n, h, a, tau, comp], logD[n, h, a, tau] for prefixes BOS + seqs."""
    n, t = seqs.shape
    toks = np.concatenate([np.full((n, 1), n_sym), seqs], 1)  # (n, T)
    T = t + 1
    sc = scores(model, T - 1 + N_AHEAD)  # (h, a, x, d)
    vx = _value_rows(model, project=True)[..., :-1]  # (h, x, comp)
    lag = T - 1 - np.arange(T)  # lag of each position
    taus = np.arange(1, N_AHEAD + 1)
    dist = taus[:, None] + lag[None, :]  # (tau, pos)
    H = sc.shape[0]
    r = np.zeros((n, H, n_sym, N_AHEAD, vx.shape[-1]))
    logD = np.zeros((n, H, n_sym, N_AHEAD))
    for h in range(H):
        for a in range(n_sym):
            s = sc[h, a][toks[:, None, :], dist[None]]  # (n, tau, pos)
            m = s.max(-1, keepdims=True)
            w = np.exp(s - m)
            Z = w.sum(-1, keepdims=True)
            logD[:, h, a] = (m + np.log(Z))[..., 0]
            r[:, h, a] = np.einsum("ntp,npc->ntc", w / Z, vx[h][toks])
    return r, logD


def main() -> None:
    rows = [("flipflop", 0), ("flipflop", 2), ("parity_hand_reset", 1), ("parity_hand_reset", 0), ("parity_hand", 0)]
    out = {}
    for kind, seed in rows:
        p, m, _ = load(kind, seed)
        syms, states = sample(p, 3000, N_SYM, np.random.default_rng(7 + seed))
        lab = states[:, N_SYM]
        r, logD = log_coords(m, syms, p.n_symbols)
        best = []
        for h in range(r.shape[1]):
            for a in range(r.shape[2]):
                for ti in range(N_AHEAD):
                    sr = state_share(r[:, h, a, ti], lab)
                    sd = state_share(logD[:, h, a, ti, None], lab)
                    best.append((sr, sd, h, a, ti + 1))
        best = [b for b in best if np.isfinite(b[0])]
        best.sort(reverse=True)
        print(f"{kind} s{seed}: top cells by state share of r (share_r, share_logD, head, a, tau):")
        for b in best[:4]:
            print("   ", tuple(round(v, 3) if isinstance(v, float) else v for v in b))
        sds = sorted([b for b in best if np.isfinite(b[1])], key=lambda b: -b[1])[:3]
        print("    top by logD:", [tuple(round(v, 3) if isinstance(v, float) else v for v in b) for b in sds])
        _, _, h, a, tau = best[0]
        out[f"{kind} s{seed}"] = (r[:, h, a, tau - 1], logD[:, h, a, tau - 1], lab, syms, (h, a, tau))
    np.save("/tmp/logcoords.npy", out, allow_pickle=True)


if __name__ == "__main__":
    main()
