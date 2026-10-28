"""Where does a trained model lose to the true PDFA?

KL(true || model) per position, split into positions whose next-symbol
distribution needs memory beyond (group of the previous state, last token)
-- "hard" -- and the rest -- "easy". Uses shared-emission automata from
nerode_geometry.py, where states 2k and 2k+1 emit identically.

Run from the repo root:  uv run python experiments/where_errors.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "nerode_geometry", Path(__file__).with_name("nerode_geometry.py")
)
assert _spec is not None and _spec.loader is not None
ng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ng)


def main() -> None:
    for n_states, n_sym, seed in [(6, 3, 3), (8, 3, 4)]:
        rng = np.random.default_rng(seed)
        p = ng.random_pdfa(n_states, n_sym, rng, shared=True)
        partner = np.arange(n_states) ^ 1  # states 2k <-> 2k+1 share emissions
        group = np.arange(n_states) // 2
        syms, states = ng.sample(p, 6000, 24, rng)
        bos = n_sym
        x = torch.tensor(np.concatenate([np.full((len(syms), 1), bos), syms], 1))
        model = ng.make_model(n_sym, 24, 64, 2, seed)
        ng.train(model, x[:5000], steps=1500, lr=3e-3)
        model.eval()
        with torch.no_grad():
            logits = model(x[5000:, :-1].to(ng.DEVICE))
            logp = torch.log_softmax(logits, -1)[..., :n_sym].float().cpu().numpy()
        st = states[5000:, :24]  # state generating the next symbol at each position
        true = p.emit[st]
        kl = (true * (np.log(true) - logp)).sum(-1)  # (n_seq, 24)
        prev, a = st[:, :-1], syms[5000:, :23]
        hard = group[p.delta[prev, a]] != group[p.delta[partner[prev], a]]
        k = kl[:, 1:]
        print(f"\nshared PDFA {n_states} states, seed {seed}: mean KL {kl.mean():.4f}")
        print(f"  easy positions ({(~hard).mean():.0%}): KL {k[~hard].mean():.4f}")
        print(f"  hard positions ({hard.mean():.0%}): KL {k[hard].mean():.4f}")
        by_t = [
            k[:, t][hard[:, t]].mean() if hard[:, t].any() else np.nan
            for t in range(23)
        ]
        print("  hard KL by position:", " ".join(f"{v:.3f}" for v in by_t[::3]))


if __name__ == "__main__":
    main()
