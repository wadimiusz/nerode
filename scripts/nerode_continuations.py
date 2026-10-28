"""Version 2: compare prefixes through their continuations, against the
model's own behaviour. See nerode.continuations for the metrics.

Run from the repository root:  uv run python scripts/nerode_continuations.py
Faster smoke run:              uv run python scripts/nerode_continuations.py --quick
The defaults reproduce the pilot; `--fresh --steps 100000 --min-sep 0.1` the
well-trained models (see the README).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm.auto import tqdm

from nerode.continuations import run_one
from nerode.training import DEVICE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=["plain", "parity"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-prefix", type=int, default=300)
    ap.add_argument("--prefix-len", type=int, default=12)
    ap.add_argument("--w-len", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--out", default="out/continuations.txt")
    ap.add_argument(
        "--w-symbols",
        type=int,
        nargs="+",
        default=None,
        help="restrict continuations to these symbols; e.g. 0 1 keeps the parity "
        "of parity automata hidden and irrelevant to the next symbol",
    )
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--fresh", action="store_true", help="train on fresh samples every batch"
    )
    ap.add_argument(
        "--min-sep",
        type=float,
        default=0.0,
        help="parity automata: smallest allowed KL between group emissions "
        "(0 reproduces the original automata; 0.1 makes the hidden bit matter)",
    )
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.steps, a.n_prefix = [0], 200, 60

    lines: list[str] = []

    def emit(s: str) -> None:
        tqdm.write(s)
        lines.append(s)

    emit(
        f"device {DEVICE}; prefixes of length {a.prefix_len}, continuations of "
        f"length {a.w_len}, behaviour horizon {a.horizon}, {a.steps} steps, "
        f"batch {a.batch}, fresh data {a.fresh}, min group separation {a.min_sep}"
    )
    runs = [(kind, seed) for kind in a.kinds for seed in a.seeds]
    for kind, seed in tqdm(runs, desc="all runs", unit="run", dynamic_ncols=True):
        t0 = time.time()
        r = run_one(
            kind,
            seed,
            a.n_prefix,
            a.prefix_len,
            a.w_len,
            a.horizon,
            a.steps,
            w_symbols=a.w_symbols,
            fresh=a.fresh,
            batch=a.batch,
            min_sep=a.min_sep,
        )
        emit(
            f"\n{kind} PDFA, seed {seed}: {r['n_cls']} Nerode classes, "
            f"{r['n_prefix']} prefixes, optimal loss {r['h_opt']:.3f} "
            f"({time.time() - t0:.0f}s)"
        )
        if r["bit_acc"] is not None:
            acc = " ".join(f"{k}:{v:.3f}" for k, v in r["bit_acc"].items())
            emit(
                f"  group separation {r['sep']:.3f}; trained model tracks the "
                f"hidden bit, weighted accuracy by prefix length: {acc}"
            )
        emit(
            f"  {'model':8s} {'layer':7s} {'loss':>6s} {'rho_beh':>8s} "
            f"{'rho_far|nx':>10s} {'rho_w0':>7s} {'hid':>6s} {'hid_w0':>7s}"
        )
        for row in r["rows"]:
            emit(
                f"  {row['model']:8s} {row['layer']:7s} {row['loss']:6.3f} "
                f"{row['rho_beh']:8.3f} {row['rho_far']:10.3f} "
                f"{row['rho_beh_w0']:7.3f} {row['hid']:6.3f} "
                f"{row['hid_w0']:7.3f}"
            )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
