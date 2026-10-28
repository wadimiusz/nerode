"""Version 1: is the Nerode class readable from the residual stream at the
last position of the prefix? See nerode.geometry for the metrics.

Run from the repository root:  uv run python scripts/nerode_geometry.py
With PCA plots:                uv run --extra plots python scripts/nerode_geometry.py
Outputs go to out/ (or $NERODE_OUT): geometry.txt and the plots.
"""

from __future__ import annotations

import argparse

from tqdm.auto import tqdm

from nerode.geometry import run_one
from nerode.training import OUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--fresh", action="store_true", help="train on fresh samples every batch"
    )
    a = ap.parse_args()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # (n_states, n_symbols, seed, shared emissions)
    settings = [
        (4, 3, 0, False),
        (6, 3, 1, False),
        (8, 4, 2, False),
        (6, 3, 3, True),
        (8, 3, 4, True),
    ]
    lines = []

    def emit(line: str):
        tqdm.write(line)
        lines.append(line)

    for n_states, n_symbols, seed, shared in tqdm(
        settings, desc="automata", unit="automaton", dynamic_ncols=True
    ):
        r = run_one(
            n_states,
            n_symbols,
            seed,
            shared=shared,
            plot_dir=out_dir,
            steps=a.steps,
            fresh=a.fresh,
            batch=a.batch,
        )
        emit(
            f"\n{'SHARED-emission' if shared else 'plain'} PDFA: {r['n_states']} states "
            f"-> {r['n_cls']} Nerode classes ({r['n_hidden_cls']} share their next-symbol "
            f"distribution), {r['n_symbols']} symbols, seed {r['seed']} ({r['secs']:.0f}s)"
        )
        emit(f"  training: {a.steps} steps, batch {a.batch}, fresh data {a.fresh}")
        emit(
            f"  optimal loss {r['h_opt']:.3f} nats | class from (position, last token): "
            f"acc {r['trivial']:.3f}, within same-next-symbol groups "
            f"{r['trivial_hidden']:.3f} (chance 0.5)"
        )
        emit(
            f"  {'model':8s} {'layer':7s} {'loss':>6s} {'probe':>6s} {'p_hid':>6s} "
            f"{'centr':>6s} {'c_hid':>6s} {'ratio':>6s} {'rho':>6s}"
        )
        for row in r["rows"]:
            emit(
                f"  {row['model']:8s} {row['layer']:7s} {row['loss']:6.3f} "
                f"{row['probe_acc']:6.3f} {row['probe_hidden']:6.3f} "
                f"{row['centroid_acc']:6.3f} {row['centroid_hidden']:6.3f} "
                f"{row['dist_ratio']:6.3f} {row['behav_rho']:6.3f}"
            )
    (out_dir / "geometry.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
