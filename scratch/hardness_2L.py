"""How hard is attention in the two-layer models, layer by layer? (diagnostic; runs the network)

Per layer and head, on strings from the automaton: the largest attention weight on a single
position, and, in layer 1 only, the largest total weight on one key TOKEN (weights on copies of the
same token add up: they carry the same value there, so for the read only the token matters).
'hard' = share of positions where that maximum is >= 0.99.
"""
import sys, numpy as np, torch
sys.path.insert(0, "scripts")
from nerode.probes import load_models, SEQ_LEN
from nerode.automata import make_automaton, sample
from nerode.heads import attention_weights

for kind in ("parity_hand", "parity_hand_reset", "parity"):
    for seed in range(5):
        try:
            _, m, _ = load_models(kind, seed, 100_000, 128, 0.1)
        except Exception:
            continue
        p = make_automaton(kind, seed, 0.1)
        syms, _ = sample(p, 1000, SEQ_LEN, np.random.default_rng(3))
        x = torch.tensor(np.concatenate([np.full((1000, 1), p.n_symbols), syms], 1)[:, :SEQ_LEN])
        att = attention_weights(m.cpu(), x)
        row = []
        for layer, A in enumerate(att):
            A = A.numpy()[:, :, 1:]                              # skip position 0 (sees only BOS)
            for h in range(A.shape[1]):
                pos_max = A[:, h].max(-1)
                s = f"L{layer + 1}h{h}: pos {np.median(pos_max):.2f}/{(pos_max >= 0.99).mean():.2f}"
                if layer == 0:
                    toks = x.numpy()
                    mass = np.stack([(A[:, h] * (toks[:, None, :] == t)).sum(-1) for t in range(4)], -1).max(-1)
                    s += f", token {np.median(mass):.2f}/{(mass >= 0.99).mean():.2f}"
                row.append(s)
        print(f"{kind} s{seed}: " + " | ".join(row), flush=True)
