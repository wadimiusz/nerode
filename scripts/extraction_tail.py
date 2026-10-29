"""What is the long tail of small states in the extracted automata?

extract_automaton.py finds, for every parity model, six large states (each at
least 1% of the prefix mass) that match the true automaton, plus thousands of
tiny ones. This script looks at the tiny ones at one radius eps:

- how much prefix mass they hold;
- the model's error on their prefixes: total variation between the model's and
  the automaton's distributions over continuations, against the same error on
  prefixes in the large states (if the tail is where the model is wrong, its
  error should be larger than eps, and the large states' smaller);
- what they are near: for every tail prefix, the nearest large state (by the
  mean signature of its prefixes). Near the large state of the prefix's own
  true state means noise around the right answer; near the state with the
  same group and the other bit means the model got the hidden bit wrong;
- where they are: share of prefixes in the tail by the number of symbols since
  the last 2, and by prefix length;
- the model's uncertainty about the hidden bit: its signature written as a
  mixture of the true state's and the wrong-bit state's signatures; the weight
  on the wrong bit, and how much error the mixture leaves unexplained;
- transitions out of large states that do not land in a large state, e.g. one
  state's extensions by "2" split across tiny states.

Needs the cached models (see scripts/bit_probe.py). Run from the repository
root:  uv run python scripts/extraction_tail.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from nerode.extraction import (
    extract,
    majority_state,
    model_error,
    sample_prefixes,
    signatures_from_model,
    state_signatures,
    symbols_since_last,
    wrong_bit_weight,
)
from nerode.probes import SEQ_LEN, load_models
from nerode.training import DEVICE, OUT_DIR

GAP_BINS = [(0, 2), (3, 5), (6, 9), (10, 99)]
LEN_BINS = [(0, 5), (6, 10), (11, 15), (16, 99)]
ERR_BINS = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 9.0)]


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["parity_hand", "parity"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-seq", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--out", default=str(OUT_DIR / "extraction_tail.txt"))
    a = ap.parse_args()

    max_len = SEQ_LEN - a.k
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(
        f"device {DEVICE}; eps {a.eps}, continuations of length {a.k}, prefixes "
        f"up to {max_len} symbols from {a.n_seq} sampled sequences"
    )
    for kind in a.kinds:
        for seed in a.seeds:
            t0 = time.time()
            p, model, _ = load_models(kind, seed, a.steps, a.batch, a.min_sep)
            pre = sample_prefixes(p, a.n_seq, max_len, np.random.default_rng(seed))
            sig = signatures_from_model(model, pre, p.n_symbols, a.k)
            per_state = state_signatures(p, a.k)
            right = per_state[pre.state]
            err = model_error(sig, right)
            # states are 2 * group + bit, so state ^ 1 flips the bit
            lam, resid = wrong_bit_weight(sig, right, per_state[pre.state ^ 1])
            labels, ext = extract(sig, pre, a.eps, p.n_symbols)
            report(emit, kind, seed, pre, sig, err, lam, resid, labels, ext)
            emit(f"  ({time.time() - t0:.0f}s)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def report(emit, kind, seed, pre, sig, err, lam, resid, labels, ext) -> None:
    w = pre.weight
    big = ext.mass >= 0.01
    in_tail = ~big[labels]
    big_ids = np.flatnonzero(big)
    main = majority_state(pre, labels, len(ext.mass))
    emit(
        f"\n{kind}, seed {seed}: {int(big.sum())} large states, "
        f"{int((~big).sum())} tiny ones holding {w[in_tail].sum():.3f} of the mass"
    )

    # the model's error, in large states and in the tail
    emit(
        f"  model error TV(model, automaton), mean: large states "
        f"{wmean(err[~in_tail], w[~in_tail]):.4f}, tail "
        f"{wmean(err[in_tail], w[in_tail]):.4f}"
    )
    cells = []
    for lo, hi in ERR_BINS:
        m = (err >= lo) & (err < hi)
        if w[m].sum() > 0:
            cells.append(
                f"{lo:g}-{hi if hi < 9 else 'up':}: {w[m].sum():.3f} of mass, "
                f"{w[m & in_tail].sum() / w[m].sum():.2f} in tail"
            )
    emit("  by model error: " + "; ".join(cells))

    # what the tail is near
    cent = np.stack(
        [np.average(sig[labels == s], axis=0, weights=w[labels == s]) for s in big_ids]
    )
    tail_idx = np.flatnonzero(in_tail)
    if len(tail_idx):
        d = 0.5 * np.abs(sig[tail_idx, None, :] - cent[None, :, :]).sum(-1)
        near = big_ids[d.argmin(1)]
        q_near = main[near]
        q_true = pre.state[tail_idx]
        wt = w[tail_idx]
        same = q_near == q_true
        flip = (q_near // 2 == q_true // 2) & ~same
        emit(
            f"  tail prefixes nearest to the large state of: their own true state "
            f"{wt[same].sum() / wt.sum():.3f}, same group but the other bit "
            f"{wt[flip].sum() / wt.sum():.3f}, another group "
            f"{wt[~same & ~flip].sum() / wt.sum():.3f}"
        )

    # where the tail is
    gap = np.array([symbols_since_last(u, 2) for u in pre.seqs])
    length = np.array([len(u) for u in pre.seqs])
    for name, x, bins in (
        ("since last 2", gap, GAP_BINS),
        ("length", length, LEN_BINS),
    ):
        cells = []
        for lo, hi in bins:
            m = (x >= lo) & (x <= hi)
            if w[m].sum() > 0:
                rng = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
                cells.append(
                    f"{rng}: tail {w[m & in_tail].sum() / w[m].sum():.3f}, "
                    f"error {wmean(err[m], w[m]):.4f}, wrong bit "
                    f"{wmean(lam[m], w[m]):.3f}, unexplained {wmean(resid[m], w[m]):.4f}"
                )
        emit(f"  by {name}: " + "; ".join(cells))

    # transitions out of large states that do not land in a large state
    for s in big_ids:
        members = np.flatnonzero(labels == s)
        for a in range(ext.delta.shape[1]):
            ext_w: dict[int, float] = {}
            for i in members:
                j = pre.index.get((*pre.seqs[i], a))
                if j is not None:
                    t = int(labels[j])
                    ext_w[t] = ext_w.get(t, 0.0) + float(w[j])
            total = sum(ext_w.values())
            if total == 0:
                continue
            to_big = sum(v for t, v in ext_w.items() if big[t])
            best_big = max(
                (t for t in ext_w if big[t]), key=lambda t: ext_w[t], default=-1
            )
            if to_big / total < 0.9 or not big[ext.delta[s, a]]:
                emit(
                    f"  state {s} (g{main[s] // 2}b{main[s] % 2}) on {a}: "
                    f"{to_big / total:.2f} of extensions land in large states "
                    f"(most in {best_big}, g{main[best_big] // 2}b"
                    f"{main[best_big] % 2}); the vote picked {ext.delta[s, a]}"
                    f"{'' if big[ext.delta[s, a]] else ' (tiny)'}, "
                    f"{ext_w.get(int(ext.delta[s, a]), 0.0) / total:.2f} of them"
                )


if __name__ == "__main__":
    main()
