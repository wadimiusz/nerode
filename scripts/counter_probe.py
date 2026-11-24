"""In which layer does the state appear, and at which position?

For every cached parity model, prefixes that end g symbols after the last 2
(extraction.sample_gap_prefixes) get a 2 appended. An MLP probe reads the
automaton state after the prefix (6 classes: group and bit) from the residual
stream, after the token embedding, after each block and after the final norm,
at two positions:

- "last":  the last symbol of the prefix: the next prediction needs the group
           here (it sets the next-symbol distribution) but not the bit;
- "at 2":  the appended 2, where the next group depends on the whole state.

One probe per (model, position, layer), trained on all gaps pooled, tested on
held-out sequences, accuracy reported per gap; the same for the random-weights
twin, as a floor. Finding 3 of the write-up says the bit is not carried
forward but computed when needed; if so, at long gaps the state should be
readable "at 2" but not at "last", and the layer where it appears "at 2" is
where it is computed.

Run from the repository root:
    uv run python scripts/counter_probe.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nerode.extraction import sample_gap_prefixes
from nerode.probes import SEQ_LEN, fit_probe, load_models
from nerode.readout import residuals
from nerode.training import DEVICE, OUT_DIR

POSITIONS = ("last", "at 2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["parity_hand", "parity_hand_reset"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=600, help="prefixes per gap")
    ap.add_argument("--gaps", type=int, nargs="+", default=list(range(0, 19, 3)))
    ap.add_argument("--out", default=str(OUT_DIR / "counter_probe.txt"))
    a = ap.parse_args()

    lines: list[str] = []
    records: list[dict] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(f"device {DEVICE}; {a.n} prefixes per gap; MLP probe for the 6 states")
    for kind in a.kinds:
        acc: dict[tuple, list[float]] = {}
        layers: list[str] = []
        for seed in a.seeds:
            p, trained, random = load_models(kind, seed, a.steps, a.batch, a.min_sep)
            rng = np.random.default_rng(seed)
            seqs, state, gap = [], [], []
            for g in a.gaps:
                pre = sample_gap_prefixes(p, a.n, g, SEQ_LEN - 2, rng)
                seqs += pre.seqs
                state += pre.state.tolist()
                gap += [g] * len(pre.seqs)
            y, gaps = np.array(state), np.array(gap)
            lens = np.array([len(u) for u in seqs])
            bos = p.n_symbols
            x = torch.tensor(
                [[bos, *u, 2] + [bos] * (SEQ_LEN - 1 - len(u)) for u in seqs]
            )
            test = np.random.default_rng(seed + 7).random(len(seqs)) < 0.25
            for model_kind, model in (("trained", trained), ("random", random)):
                res = residuals(model, x)
                layers = [name for name, _ in res]
                for name, h in res:
                    for pos, t in (("last", lens), ("at 2", lens + 1)):
                        feats = h[np.arange(len(seqs)), t]
                        _, pred = fit_probe(
                            feats[~test], y[~test], feats[test], y[test], 6, hidden=128
                        )
                        ok = pred == y[test]
                        for g in a.gaps:
                            m = gaps[test] == g
                            key = (model_kind, pos, name, g)
                            acc.setdefault(key, []).append(float(ok[m].mean()))
        emit(
            f"\n{kind}: probe accuracy for the state (mean over {len(a.seeds)} seeds;"
            " chance about 1/6, the group alone gives up to 1/2)"
        )
        for model_kind in ("trained", "random"):
            for pos in POSITIONS:
                emit(f"  {model_kind}, {pos}:")
                emit(
                    "    "
                    + f"{'layer':>8s} "
                    + " ".join(f"{'g=' + str(g):>6s}" for g in a.gaps)
                )
                for name in layers:
                    vals = [np.mean(acc[(model_kind, pos, name, g)]) for g in a.gaps]
                    emit(f"    {name:>8s} " + " ".join(f"{v:6.2f}" for v in vals))
                    records.append(
                        {
                            "kind": kind,
                            "model": model_kind,
                            "position": pos,
                            "layer": name,
                            "acc": dict(zip(map(str, a.gaps), vals)),
                        }
                    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
