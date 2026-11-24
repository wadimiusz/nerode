"""Which attention head carries the count, and where does it look?

Part 1, ablation. For every cached parity model, each of its four heads in
turn is replaced by its mean output over natural samples (mean ablation; the
zero-ablation loss is shown alongside as a check). Reported per head:
- loss: the extra loss on natural samples, in nats over the optimal loss;
- true@g: the model's belief on the true state (as in belief_by_gap.py) on
  prefixes that end g symbols after the last 2.
A head whose removal hurts long gaps much more than short ones carries
information from far back.

Part 2, attention. For prefixes that end g symbols after the last 2, a 2 is
appended: this is where the model has to know the group to predict what comes
next. At that position, the share of each head's attention on: BOS, symbols
before the last 2, the last 2 itself, the g symbols since it ("gap"), and the
position itself. Within the gap: how much more a 1 is looked at than a 0
(ratio of mean attention per token), and how unevenly the gap is covered
(coefficient of variation of attention across gap positions; 0 = uniform).

Run from the repository root:
    uv run python scripts/head_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nerode.automata import entropy_rate, sample
from nerode.extraction import (
    belief_weights,
    sample_gap_prefixes,
    signatures_from_model,
    state_signatures,
)
from nerode.heads import ablate, attention_weights, head_means
from nerode.probes import SEQ_LEN, load_models
from nerode.training import DEVICE, OUT_DIR, eval_loss

HEADS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["parity_hand", "parity_hand_reset"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=300, help="prefixes per gap")
    ap.add_argument("--gaps", type=int, nargs="+", default=[0, 4, 8, 12, 16])
    ap.add_argument("--out", default=str(OUT_DIR / "head_ablation.txt"))
    a = ap.parse_args()

    lines: list[str] = []
    records: list[dict] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(f"device {DEVICE}; {a.n} prefixes per gap, gaps {a.gaps}")
    for kind in a.kinds:
        for seed in a.seeds:
            p, model, _ = load_models(kind, seed, a.steps, a.batch, a.min_sep)
            rng = np.random.default_rng(seed)
            syms, states = sample(p, 2000, SEQ_LEN, rng)
            x = torch.tensor(np.concatenate([np.full((2000, 1), p.n_symbols), syms], 1))
            h_opt = entropy_rate(p, states)
            means = head_means(model, x[:, :-1])
            per_state = state_signatures(p, a.k)
            pres = {
                g: sample_gap_prefixes(p, a.n, g, SEQ_LEN - a.k, rng) for g in a.gaps
            }

            emit(
                f"\n{kind}, seed {seed}: ablation (mean; zero-ablation loss in brackets)"
            )
            emit(
                f"  {'head':>8s} {'loss':>16s} "
                + " ".join(f"{'true@' + str(g):>8s}" for g in a.gaps)
            )
            conditions: list[list[tuple[int, int]]] = [[]] + [[h] for h in HEADS]
            for heads in conditions:
                with ablate(model, heads, means):
                    loss = eval_loss(model, x) - h_opt
                    trues = [
                        belief_true(model, pres[g], per_state, p.n_symbols, a.k)
                        for g in a.gaps
                    ]
                with ablate(model, heads):
                    loss0 = eval_loss(model, x) - h_opt
                name = "none" if not heads else f"L{heads[0][0]}H{heads[0][1]}"
                emit(
                    f"  {name:>8s} {loss:+8.4f} ({loss0:+.3f}) "
                    + " ".join(f"{v:8.3f}" for v in trues)
                )
                records.append(
                    {
                        "kind": kind,
                        "seed": seed,
                        "part": "ablation",
                        "head": name,
                        "loss": loss,
                        "loss_zero": loss0,
                        "true": dict(zip(map(str, a.gaps), trues)),
                    }
                )
            attention_report(
                emit, records, kind, seed, model, p.n_symbols, pres, a.gaps
            )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(records, indent=1))


def belief_true(model, pre, per_state: np.ndarray, n_sym: int, k: int) -> float:
    """Mean belief on the true state over the prefixes (see belief_by_gap.py)."""
    sig = signatures_from_model(model, pre, n_sym, k)
    b, _ = belief_weights(sig, per_state)
    return float(b[np.arange(len(b)), pre.state].mean())


def attention_report(emit, records, kind, seed, model, n_sym, pres, gaps) -> None:
    emit(
        "  attention at an appended 2, by head: shares on BOS / before / last 2 / "
        "gap / self; in the gap: 1-vs-0 ratio, unevenness (CV)"
    )
    for g in [g for g in gaps if g >= 4]:
        seqs = pres[g].seqs
        x = torch.tensor([[n_sym, *u, 2] + [n_sym] * (21 - len(u)) for u in seqs])
        # padding after the appended 2 is never attended to (causal mask)
        weights = attention_weights(model, x)
        for layer, head in HEADS:
            w = weights[layer][:, head].cpu().numpy()
            parts = np.zeros(5)
            ratio, cv = [], []
            for i, u in enumerate(seqs):
                t = len(u) + 1  # position of the appended 2
                i2 = len(u) - g  # position of the last 2 in u (BOS is 0)
                row = w[i, t]
                parts += [
                    row[0],
                    row[1:i2].sum(),
                    row[i2],
                    row[i2 + 1 : t].sum(),
                    row[t],
                ]
                gap_w = row[i2 + 1 : t]
                gap_s = np.array(u[len(u) - g :])
                if (gap_s == 0).any() and (gap_s == 1).any():
                    ratio.append(gap_w[gap_s == 1].mean() / gap_w[gap_s == 0].mean())
                cv.append(gap_w.std() / gap_w.mean())
            parts /= len(seqs)
            emit(
                f"    g={g:2d} L{layer}H{head}: "
                + " / ".join(f"{v:.2f}" for v in parts)
                + f"; 1:0 {np.median(ratio):.2f}, CV {np.mean(cv):.2f}"
            )
            records.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "part": "attention",
                    "gap": g,
                    "head": f"L{layer}H{head}",
                    "shares": parts.tolist(),
                    "ratio_1_0": float(np.median(ratio)),
                    "cv": float(np.mean(cv)),
                }
            )


if __name__ == "__main__":
    main()
