"""Extract each parity model's automaton from its behaviour and compare it with
the true one (see src/nerode/extraction.py for the method).

For every cached parity model, and for every radius eps:
- states: how many states the extracted automaton has, and how many of them
  hold at least 1% of the prefix mass;
- purity: share of prefix mass in states that do not mix true classes;
- determinism: how often the extensions u·a of one state's prefixes agree on
  the next state;
- coverage, KL: on held-out sequences, the share of positions the extracted
  automaton can follow, and KL(model || extracted) next to
  KL(model || true automaton) on the same positions, in nats per position.
  The extracted automaton describes the model better than the truth does
  when its KL is lower.

For one radius (--detail-eps) it also lists the states with at least 1% of the
mass: which true state (group g, bit b) most of their prefixes are in, and how
many symbols back the last "2" was. A true state split into several
extracted states shows what the model distinguishes that the automaton does not.

As a check of the pipeline, every automaton is first run through it with the
true probabilities instead of a model's ("oracle"): that must give the true
number of states with purity 1.

The models come from the checkpoint cache (see scripts/bit_probe.py). Run from
the repository root:
    uv run python scripts/extract_automaton.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nerode.automata import minimise
from nerode.extraction import (
    extract,
    heldout_kl,
    majority_state,
    n_true_classes,
    purity,
    sample_prefixes,
    signatures_from_model,
    signatures_from_pdfa,
    symbols_since_last,
)
from nerode.probes import SEQ_LEN, load_models
from nerode.training import DEVICE, OUT_DIR

EPS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["parity_hand", "parity"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=3, help="continuation length")
    ap.add_argument("--n-seq", type=int, default=2000, help="sampled sequences")
    ap.add_argument("--eps", type=float, nargs="+", default=EPS)
    ap.add_argument("--detail-eps", type=float, default=0.02)
    ap.add_argument("--out", default=str(OUT_DIR / "extract_automaton.txt"))
    a = ap.parse_args()

    max_len = SEQ_LEN - a.k  # BOS + prefix + continuation must fit the context
    lines: list[str] = []
    rows: list[dict] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(
        f"device {DEVICE}; continuations of length {a.k}, prefixes up to "
        f"{max_len} symbols from {a.n_seq} sampled sequences; models: "
        f"{a.steps} steps, batch {a.batch}, min group separation {a.min_sep}"
    )
    for kind in a.kinds:
        for seed in a.seeds:
            t0 = time.time()
            p, model, _ = load_models(kind, seed, a.steps, a.batch, a.min_sep)
            n_sym = p.n_symbols
            true_cls = minimise(p)
            pre = sample_prefixes(p, a.n_seq, max_len, np.random.default_rng(seed))
            emit(
                f"\n{kind}, seed {seed}: {n_true_classes(p)} true states, "
                f"{len(pre.seqs)} distinct prefixes"
            )
            sig_true = signatures_from_pdfa(p, pre, a.k)
            labels, ext = extract(sig_true, pre, a.detail_eps, n_sym)
            emit(
                f"  oracle (true probabilities, eps {a.detail_eps}): "
                f"{len(ext.mass)} states, purity "
                f"{purity(pre, labels, true_cls):.3f}, determinism "
                f"{ext.determinism:.3f}"
            )
            sig = signatures_from_model(model, pre, n_sym, a.k)
            emit(
                f"  {'eps':>6s} {'states':>6s} {'>=1%':>5s} {'purity':>7s} "
                f"{'determ':>7s} {'cover':>6s} {'KL ext':>8s} {'KL true':>8s}"
            )
            for eps in a.eps:
                labels, ext = extract(sig, pre, eps, n_sym)
                ev = heldout_kl(
                    model, p, ext, 2000, SEQ_LEN, np.random.default_rng(99 + seed)
                )
                row = {
                    "kind": kind,
                    "seed": seed,
                    "eps": eps,
                    "states": len(ext.mass),
                    "states_1pct": int((ext.mass >= 0.01).sum()),
                    "purity": purity(pre, labels, true_cls),
                    "determinism": ext.determinism,
                    **ev,
                }
                rows.append(row)
                emit(
                    f"  {eps:6.3f} {row['states']:6d} {row['states_1pct']:5d} "
                    f"{row['purity']:7.3f} {row['determinism']:7.3f} "
                    f"{ev['coverage']:6.3f} {ev['kl_ext']:8.5f} "
                    f"{ev['kl_true']:8.5f}"
                )
            detail(emit, pre, sig, a.detail_eps, n_sym)
            emit(f"  ({time.time() - t0:.0f}s)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))


def detail(emit, pre, sig, eps: float, n_sym: int) -> None:
    """States with at least 1% of the mass at radius eps: their main true state
    (g = group, b = bit) and how far back the last 2 was in their prefixes."""
    labels, ext = extract(sig, pre, eps, n_sym)
    main = majority_state(pre, labels, len(ext.mass))
    gap = np.array([symbols_since_last(u, 2) for u in pre.seqs])
    emit(f"  states at eps {eps} with >= 1% of the mass:")
    emit(
        f"    {'state':>5s} {'mass':>6s} {'true':>5s} {'share':>6s} "
        f"{'gap since 2: mean':>17s} {'range':>7s}  next state on 0 1 2"
    )
    for s in np.argsort(-ext.mass):
        if ext.mass[s] < 0.01:
            break
        m = labels == s
        w = pre.weight[m]
        share = w[pre.state[m] == main[s]].sum() / w.sum()
        g = gap[m]
        nxt = " ".join(f"{int(d):>2d}" for d in ext.delta[s])
        emit(
            f"    {int(s):5d} {ext.mass[s]:6.3f} g{main[s] // 2}b{main[s] % 2} "
            f"{share:6.3f} {np.average(g, weights=w):17.1f} "
            f"{int(g.min()):>3d}-{int(g.max()):<3d}  {nxt}"
        )


if __name__ == "__main__":
    main()
