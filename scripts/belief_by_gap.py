"""Where does a parity model's uncertainty go when the last 2 was long ago?

For every gap g (number of symbols since the last 2) and every cached model,
sample prefixes that end exactly g symbols after a 2 (extraction.
sample_gap_prefixes: natural samples hardly contain long gaps), take the
model's signature of each prefix (its distribution over all continuations of
length k) and write it as a mixture of the automaton's states
(extraction.belief_weights). Reported, averaged over the prefixes at each gap:

- error:      total variation between the model and the automaton;
- unexpl.:    what no mixture of states explains (TV to the best mixture);
- true:       belief on the true state;
- flip:       belief on the state with the same group and the other bit;
- groups:     belief on states of the other two groups.

For parity_hand, whose group and bit are one counter mod 6 (automata.
hand_counter), the belief is also split by how far each state's counter is
from the true one: shift 0 is the true state, 3 the flipped bit, 1 and 5 one
step off, 2 and 4 two steps off. Everything also goes to a JSON file next to
the text output, for scripts/plot_belief_by_gap.py.

The question it was written for (Nerode research log, 29.10.2026): in the demo
automaton symbols 0 and 1 rotate the groups, so over a long stretch the model
must count mod 6; in parity_hand_reset symbol 1 resets the group, so only the
bit must be carried. Prediction, written down before training the reset
models: at long gaps the uncertainty of parity_hand goes mostly to the other
groups, that of parity_hand_reset mostly to the flipped bit.

Needs the cached models (scripts/train_parity.py). Run from the repository
root:
    uv run python scripts/belief_by_gap.py --kinds parity_hand parity_hand_reset
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nerode.automata import hand_counter
from nerode.extraction import (
    belief_weights,
    model_error,
    sample_gap_prefixes,
    signatures_from_model,
    state_signatures,
)
from nerode.probes import SEQ_LEN, load_models
from nerode.training import DEVICE, OUT_DIR

COLS = ("error", "unexpl.", "true", "flip", "groups")
SHIFTS = ("0", "+-1", "+-2", "3")  # counter shifts, +1 and -1 (= 5) together


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["parity_hand", "parity_hand_reset"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=400, help="prefixes per gap")
    ap.add_argument("--gaps", type=int, nargs="+", default=list(range(0, 19, 2)))
    ap.add_argument("--out", default=str(OUT_DIR / "belief_by_gap.txt"))
    a = ap.parse_args()

    max_len = SEQ_LEN - a.k
    lines: list[str] = []
    records: list[dict] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(
        f"device {DEVICE}; {a.n} prefixes per gap, continuations of length {a.k}, "
        f"prefixes up to {max_len} symbols"
    )
    header = f"  {'gap':>4s} " + " ".join(f"{c:>8s}" for c in COLS)
    for kind in a.kinds:
        per_kind: dict[int, list[np.ndarray]] = {}
        for seed in a.seeds:
            t0 = time.time()
            p, model, _ = load_models(kind, seed, a.steps, a.batch, a.min_sep)
            per_state = state_signatures(p, a.k)
            rng = np.random.default_rng(seed)
            emit(f"\n{kind}, seed {seed}")
            emit(header)
            for gap in a.gaps:
                pre = sample_gap_prefixes(p, a.n, gap, max_len, rng)
                sig = signatures_from_model(model, pre, p.n_symbols, a.k)
                b, unexpl = belief_weights(sig, per_state)
                n = np.arange(len(b))
                true = b[n, pre.state]
                flip = b[n, pre.state ^ 1]  # states are 2 * group + bit
                row = np.array(
                    [
                        np.average(v, weights=pre.weight)
                        for v in (
                            model_error(sig, per_state[pre.state]),
                            unexpl,
                            true,
                            flip,
                            1.0 - true - flip,
                        )
                    ]
                )
                per_kind.setdefault(gap, []).append(row)
                line = f"  {gap:4d} " + " ".join(f"{v:8.4f}" for v in row)
                rec = {"kind": kind, "seed": seed, "gap": gap}
                rec.update(dict(zip(COLS, map(float, row))))
                if kind == "parity_hand":
                    by_shift = counter_shifts(b, pre.state, pre.weight)
                    rec["shift"] = dict(zip(SHIFTS, map(float, by_shift)))
                    line += "   shift " + " ".join(
                        f"{k} {v:.3f}" for k, v in zip(SHIFTS, by_shift)
                    )
                records.append(rec)
                emit(line)
            emit(f"  ({time.time() - t0:.0f}s)")
        emit(f"\n{kind}, mean over seeds (min-max of 'flip' and 'groups')")
        emit(header)
        for gap, rows in per_kind.items():
            r = np.array(rows)
            emit(
                f"  {gap:4d} "
                + " ".join(f"{v:8.4f}" for v in r.mean(0))
                + f"   flip {r[:, 3].min():.3f}-{r[:, 3].max():.3f},"
                + f" groups {r[:, 4].min():.3f}-{r[:, 4].max():.3f}"
            )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(records, indent=1))


def counter_shifts(b: np.ndarray, state: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Belief by counter shift, for parity_hand: mean belief on states whose
    counter is 0, +-1, +-2 and 3 away from the true state's."""
    n = np.array([hand_counter(q) for q in range(b.shape[1])])
    shift = (n[None, :] - n[state][:, None]) % 6  # (prefix, state)
    dist = np.minimum(shift, 6 - shift)  # 0, 1, 2 or 3
    per = np.stack([(b * (dist == d)).sum(1) for d in range(4)], 1)
    return np.average(per, axis=0, weights=weight)


if __name__ == "__main__":
    main()
