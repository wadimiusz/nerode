"""Probes trained only on long-gap positions (gap >= 8, >= 11), so that short
gaps do not dominate training; trained model vs. random-weights twin
(26.10.2026). Needs the cached models (see scripts/bit_probe.py).

Run from the repository root:  uv run python scripts/bit_probe_long_gaps.py
"""

from __future__ import annotations

import numpy as np
import torch

from nerode.automata import sample
from nerode.probes import SEQ_LEN as L
from nerode.probes import fit_probe, load_models, symbols_since_2
from nerode.readout import residuals


def main() -> None:
    out: dict[tuple, list[float]] = {}
    for kind in ["parity_hand", "parity"]:
        for seed in range(5):
            p, trained, rnd = load_models(kind, seed, 100_000, 128, 0.1)
            rng = np.random.default_rng(999 + seed)
            n = 12000
            syms, states = sample(p, n, L, rng)
            x = torch.tensor(np.concatenate([np.full((n, 1), p.n_symbols), syms], 1))
            st = states[:, :L]
            dist = symbols_since_2(syms)
            seq = np.repeat(np.arange(n)[:, None], L, 1)
            for lo in (8, 11):
                keep = dist >= lo
                bit = (st % 2)[keep]
                s = seq[keep]
                tr = s < int(0.8 * n)
                te = ~tr
                maj = (bit[te] == int(bit[tr].mean() > 0.5)).mean()
                out.setdefault((lo, "majority"), []).append(maj)
                out.setdefault((lo, "n_test"), []).append(int(te.sum()))
                for mname, model in (("trained", trained), ("random", rnd)):
                    with torch.no_grad():
                        res = dict(residuals(model, x[:, :L]))
                    for layer in ("block1", "all"):
                        h = (
                            res[layer][keep]
                            if layer != "all"
                            else np.concatenate(
                                [
                                    res["embed"][keep],
                                    res["block0"][keep],
                                    res["block1"][keep],
                                ],
                                -1,
                            )
                        )
                        for hid in (None, 256):
                            a, _ = fit_probe(
                                h[tr], bit[tr], h[te], bit[te], 2, hid, epochs=40
                            )
                            out.setdefault(
                                (lo, mname, layer, "mlp" if hid else "lin"), []
                            ).append(a)
            print(kind, seed, "done", flush=True)
    for lo in (8, 11):
        print(
            f"\ngap >= {lo}: majority {np.mean(out[(lo, 'majority')]):.3f}, "
            f"test positions per model ~{int(np.mean(out[(lo, 'n_test')]))}"
        )
        for mname in ("trained", "random"):
            for layer in ("block1", "all"):
                per_model = " ".join(f"{v:.2f}" for v in out[(lo, mname, layer, "mlp")])
                print(
                    f"  {mname:7s} {layer:6s} "
                    f"lin {np.mean(out[(lo, mname, layer, 'lin')]):.3f}  "
                    f"mlp {np.mean(out[(lo, mname, layer, 'mlp')]):.3f}  "
                    f"(per-model mlp: {per_model})"
                )


if __name__ == "__main__":
    main()
