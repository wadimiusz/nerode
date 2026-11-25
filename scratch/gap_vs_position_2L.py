"""Two-layer parity_hand_reset: does losing the bit depend on the gap or on the absolute position?
(24.11.2026). Prefixes = head of length L0 sampled from the automaton, then 2, then g symbols with 2
removed; belief on the flipped bit as in scripts/belief_by_gap.py (k = 3 continuations)."""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from nerode.extraction import Prefixes, belief_weights, signatures_from_model, state_signatures  # noqa: E402
from nerode.automata import sample  # noqa: E402
from nerode.probes import load_models  # noqa: E402

K, N = 3, 300


def prefixes(p, L0, g, rng):
    heads, hs = sample(p, N, max(L0, 1), rng)
    emit = p.emit.copy(); emit[:, 2] = 0; emit /= emit.sum(1, keepdims=True); cum = emit.cumsum(1)
    seqs, states = [], []
    for i in range(N):
        u = heads[i, :L0].tolist() + [2]; q = int(p.delta[hs[i, L0], 2])
        for _ in range(g):
            a = int((rng.random() > cum[q]).sum()); u.append(a); q = int(p.delta[q, a])
        seqs.append(tuple(u)); states.append(q)
    uniq = {}
    for s, q in zip(seqs, states):
        uniq.setdefault(s, q)
    keys = sorted(uniq, key=len)
    return Prefixes(seqs=keys, weight=np.full(len(keys), 1 / len(keys)),
                    state=np.array([uniq[k] for k in keys]), index={k: i for i, k in enumerate(keys)})


def main():
    max_len = 24 - K
    for seed in range(5):
        p, m, _ = load_models("parity_hand_reset", seed, 100_000, 128, 0.1)
        per_state = state_signatures(p, K)
        rng = np.random.default_rng(seed)
        print(f"== seed {seed}: flip weight, rows gap g, columns head length L0")
        Ls = [0, 2, 4, 8, 12]
        print("   g  " + "  ".join(f"L0={L:2d}" for L in Ls))
        for g in (6, 10, 12, 14, 16, 18):
            row = []
            for L0 in Ls:
                if L0 + 1 + g > max_len:
                    row.append("   -  "); continue
                pre = prefixes(p, L0, g, rng)
                b, _ = belief_weights(signatures_from_model(m, pre, p.n_symbols, K), per_state)
                n = np.arange(len(b))
                row.append(f"{b[n, pre.state ^ 1].mean():.3f} ")
            print(f"  {g:2d}  " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
