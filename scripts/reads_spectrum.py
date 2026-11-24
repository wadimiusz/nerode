"""The singular values of the reads matrix of one-layer models.

For each trained one-layer model (from the checkpoint cache, trained by
`scripts/train_parity.py --layers 1`) and its random-weights twin, build the
reads matrix H from the weights (src/nerode/reads.py) and print its singular
values relative to the largest one, and the numerical rank at a few
tolerances: the dimension of the linear state through which a prefix can act
on later reads. Next to it, the number of states of the minimal automaton.

Two versions of H: "heads" counts everything the heads read; "residual"
multiplies each head's numerator by its block of W_O first, so it counts only
what reaches the residual stream. Each (head, query) block is divided by its
largest weight first (`normalize`), which changes no read and keeps exp from
overflowing; the spectrum still weighs entries by absolute size, so under
nearly hard attention (scores in the hundreds) it shows little more than the
few largest weights.

No sequences are run through the model.

Run from the repository root, e.g.:
    uv run python scripts/reads_spectrum.py --kinds flipflop parity_hand_reset
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from centigpt.model import GPT

from nerode.automata import AUTOMATON_KINDS, PDFA, make_automaton, minimise
from nerode.probes import SEQ_LEN
from nerode.reads import reads_matrix
from nerode.training import DEVICE, ckpt_path, make_model

TOLS = (1e-2, 1e-3, 1e-4, 1e-6)


def load_one_layer(
    kind: str, seed: int, steps: int, batch: int, min_sep: float
) -> tuple[PDFA, GPT, GPT]:
    p = make_automaton(kind, seed, min_sep)
    trained = make_model(p.n_symbols, SEQ_LEN, 64, 1, seed)
    dummy = torch.zeros((1, SEQ_LEN + 1), dtype=torch.long)
    path = ckpt_path(
        trained,
        p,
        dummy,
        steps=steps,
        seed=seed,
        seq_len=SEQ_LEN,
        batch=batch,
        fresh=True,
    )
    if not path.exists():
        raise RuntimeError(
            f"{kind} seed {seed}: one-layer model not in the checkpoint cache "
            f"({path}); train it with scripts/train_parity.py --layers 1 first"
        )
    trained.load_state_dict(torch.load(path, map_location=DEVICE))
    random = make_model(p.n_symbols, SEQ_LEN, 64, 1, seed)
    return p, trained.cpu().eval(), random.cpu().eval()


def report(name: str, h: np.ndarray, n_show: int) -> None:
    sv = np.linalg.svd(h, compute_uv=False)
    rel = sv / sv[0]
    ranks = "  ".join(f"rank@{tol:g}={int((rel > tol).sum())}" for tol in TOLS)
    head = " ".join(f"{r:.1e}" for r in rel[:n_show])
    print(f"  {name:8s} {ranks}\n           {head}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", choices=AUTOMATON_KINDS, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--lags", type=int, default=12, help="prefix positions")
    ap.add_argument("--ahead", type=int, default=12, help="future offsets")
    ap.add_argument("--show", type=int, default=16, help="singular values shown")
    a = ap.parse_args()
    if a.lags + a.ahead > SEQ_LEN + 1:
        ap.error(f"--lags + --ahead must be at most {SEQ_LEN + 1} (the context)")

    for kind in a.kinds:
        for seed in a.seeds:
            p, trained, random = load_one_layer(kind, seed, a.steps, a.batch, a.min_sep)
            n_classes = len(set(minimise(p).tolist()))
            queries = list(range(p.n_symbols))
            print(
                f"{kind} seed {seed}: minimal automaton {n_classes} states",
                flush=True,
            )
            for project, label in ((False, "heads"), (True, "residual")):
                h_tr = reads_matrix(trained, a.lags, a.ahead, queries, project, True)
                h_rd = reads_matrix(random, a.lags, a.ahead, queries, project, True)
                print(f" {label}: H is {h_tr.shape[0]} x {h_tr.shape[1]}")
                report("trained", h_tr, a.show)
                report("random", h_rd, a.show)


if __name__ == "__main__":
    main()
