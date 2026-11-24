"""Train parity (and other) models into the checkpoint cache, with nothing else.

The probe and extraction scripts load models from the cache and never train
them. The original parity models were trained by nerode_continuations.py,
which also runs its whole analysis; this script trains the same models (same
size, fresh data every batch, same cache key, so scripts/bit_probe.py,
extract_automaton.py, extraction_tail.py and belief_by_gap.py find them) and
reports the final loss against the optimal one. Models already in the cache
are skipped.

Run from the repository root, e.g. for the reset variant of the demo automaton:
    uv run python scripts/train_parity.py --kinds parity_hand_reset

--layers 1 trains one-layer models (for the reads matrix, scripts/reads_spectrum.py);
the number of layers is part of the cache key, so they never mix with the
two-layer models.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nerode.automata import AUTOMATON_KINDS, entropy_rate, make_automaton, sample
from nerode.probes import SEQ_LEN
from nerode.training import eval_loss, fit, make_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", choices=AUTOMATON_KINDS, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--layers", type=int, default=2)
    a = ap.parse_args()

    for kind in a.kinds:
        for seed in a.seeds:
            p = make_automaton(kind, seed, a.min_sep)
            syms, states = sample(p, 4000, SEQ_LEN, np.random.default_rng(10**6 + seed))
            h_opt = entropy_rate(p, states)
            x = torch.tensor(
                np.concatenate([np.full((len(syms), 1), p.n_symbols), syms], 1)
            )
            model = make_model(p.n_symbols, SEQ_LEN, 64, a.layers, seed)
            fit(
                model,
                p,
                torch.zeros((1, SEQ_LEN + 1), dtype=torch.long),  # not in the key
                steps=a.steps,
                seed=seed,
                seq_len=SEQ_LEN,
                batch=a.batch,
                fresh=True,
                desc=f"{kind} seed {seed} ({a.layers}L)",
                h_opt=h_opt,
            )
            loss = eval_loss(model, x)
            print(
                f"{kind} seed {seed} ({a.layers}L): held-out loss {loss:.4f}, optimal {h_opt:.4f}, "
                f"gap {loss - h_opt:+.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
