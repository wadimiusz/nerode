"""Where is the hidden parity bit kept? Probes on the trained parity models.

For every cached parity model (and a random-weights twin):
1. sample sequences from its automaton and take the residual stream at every
   position (positions right after a "2" are dropped: there the bit is always 0);
2. train a linear probe and an MLP probe (two hidden layers) to predict the
   bit, on disjoint train and test sequences;
3. compare with guessing the bit from the group and the last k symbols
   (k = 1..6), and break the MLP's accuracy down by the number of symbols since
   the last "2". Capacity check: the same MLP predicting the group;
4. check that the model applies the bit when it is read: right after a "2",
   does its next-symbol distribution point to the right group, by the number of
   symbols since the previous "2"?

The models come from the checkpoint cache, so train them first, e.g.
    uv run python scripts/nerode_continuations.py --kinds parity_hand parity \\
        --seeds 0 1 2 3 4 --fresh --steps 100000 --min-sep 0.1
then, from the repository root:
    uv run python scripts/bit_probe.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nerode.probes import (
    WINDOW,
    applied_by_gap,
    features,
    fit_probe,
    load_models,
    window_baseline,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["parity_hand", "parity"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--n-seq", type=int, default=2500)
    ap.add_argument("--out", default="out/bit_probe.json")
    ap.add_argument(
        "--layers", nargs="+", default=["block0", "block1", "norm_f", "all"]
    )
    a = ap.parse_args()

    rows: list[dict] = []
    gaps: dict[int, list[int]] = {}
    t0 = time.time()
    for kind in a.kinds:
        for seed in a.seeds:
            p, trained, random = load_models(kind, seed, a.steps, a.batch, a.min_sep)
            for gap, (ok, n) in applied_by_gap(
                trained, p, 4000, np.random.default_rng(777 + seed)
            ).items():
                g = gaps.setdefault(gap, [0, 0])
                g[0] += ok
                g[1] += n
            for model_kind, model in (("trained", trained), ("random", random)):
                f = features(model, p, a.n_seq, np.random.default_rng(1234 + seed))
                tr = f["seq"] < int(0.8 * a.n_seq)
                te = ~tr
                wins = {
                    k: round(
                        window_baseline(f["group"], f["win"], f["bit"], tr, te, k), 3
                    )
                    for k in range(1, WINDOW + 1)
                }
                dte = np.minimum(f["dist"][te], 7)  # 7 means 7 or more
                for layer, h in f["layers"].items():
                    if layer not in a.layers:
                        continue
                    lin, _ = fit_probe(
                        h[tr], f["bit"][tr], h[te], f["bit"][te], 2, None
                    )
                    mlp, pred = fit_probe(
                        h[tr], f["bit"][tr], h[te], f["bit"][te], 2, 256
                    )
                    grp, _ = fit_probe(
                        h[tr], f["group"][tr], h[te], f["group"][te], 3, 256
                    )
                    ok = pred == f["bit"][te]
                    by_dist = {
                        int(d): round(float(ok[dte == d].mean()), 3)
                        for d in np.unique(dte)
                    }
                    rows.append(
                        dict(
                            kind=kind,
                            seed=seed,
                            model=model_kind,
                            layer=layer,
                            window_baseline=wins,
                            linear=round(lin, 3),
                            mlp=round(mlp, 3),
                            group_mlp=round(grp, 3),
                            mlp_by_symbols_since_2=by_dist,
                        )
                    )
            print(f"{kind} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    print("\nMLP probe for the bit, mean over models, by symbols since the last 2:")
    for model_kind in ("trained", "random"):
        for layer in a.layers:
            sel = [r for r in rows if r["model"] == model_kind and r["layer"] == layer]
            by = {
                d: np.mean([r["mlp_by_symbols_since_2"][d] for r in sel])
                for d in range(1, 8)
            }
            line = " ".join(f"{d}:{v:.2f}" for d, v in by.items())
            print(
                f"  {model_kind:7s} {layer:6s} overall {np.mean([r['mlp'] for r in sel]):.3f} "
                f"linear {np.mean([r['linear'] for r in sel]):.3f} "
                f"group {np.mean([r['group_mlp'] for r in sel]):.3f} | {line} (7 = 7+)"
            )
    wb = {
        k: np.mean([r["window_baseline"][k] for r in rows if r["model"] == "trained"])
        for k in range(1, WINDOW + 1)
    }
    print(
        "guess from group + last k symbols:",
        " ".join(f"k={k}:{v:.3f}" for k, v in wb.items()),
    )
    print(
        "model picks the right group right after a 2, by symbols since the previous 2:"
    )
    print(
        "  "
        + " ".join(f"{g}:{ok / n:.3f}" for g, (ok, n) in sorted(gaps.items()))
        + " (10 = 10+)"
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(probes=rows, applied_by_gap=gaps), indent=1))


if __name__ == "__main__":
    main()
