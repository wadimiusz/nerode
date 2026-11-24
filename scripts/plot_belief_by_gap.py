"""Figures from scripts/belief_by_gap.py (reads its JSON output).

1. belief_by_gap.png: for each automaton, the model's belief on the true state,
   on the flipped bit and on the other groups, against the number of symbols
   since the last 2. Lines are means over seeds, bands the range over seeds.
2. belief_by_shift.png: for parity_hand only, the belief split by how far the
   believed state's counter mod 6 is from the true one (0, +-1, +-2, 3).

Run from the repository root, after belief_by_gap.py:
    uv run --extra plots python scripts/plot_belief_by_gap.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nerode.training import OUT_DIR

# Reference categorical palette, first slots (light mode), and text inks.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

TITLES = {
    "parity_hand": "0 and 1 rotate the groups\n(group and bit: one counter mod 6)",
    "parity_hand_reset": "1 resets the group\n(only the bit must be carried)",
}
PARTS = (
    ("true", "true state", BLUE),
    ("flip", "flipped bit", ORANGE),
    ("groups", "other groups", AQUA),
)
SHIFTS = (
    ("0", "0 (true)", BLUE),
    ("+-1", "±1", ORANGE),
    ("+-2", "±2", AQUA),
    ("3", "3 (flipped bit)", YELLOW),
)


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("symbols since the last 2", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.0)


def series(rows: list[dict], key, gaps: list[int]):
    vals = np.array([[key(r) for r in rows if r["gap"] == g] for g in gaps])
    return vals.mean(1), vals.min(1), vals.max(1)


def plot_lines(ax, rows, gaps, parts, min_gap: float = 0.065) -> None:
    """Mean lines with seed-range bands, and direct labels at the right end,
    pushed apart vertically so that they never overlap."""
    ends = []
    for key, label, color in parts:
        mean, lo, hi = series(rows, key, gaps)
        ax.fill_between(gaps, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            gaps, mean, color=color, linewidth=2, marker="o", markersize=4, label=label
        )
        ends.append([float(mean[-1]), label])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):  # push labels up until they are min_gap apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
    for y, label in ends:
        ax.annotate(
            label,
            (gaps[-1], y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK2,
            annotation_clip=False,
        )
    ax.set_xticks(gaps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT_DIR / "belief_by_gap.json"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    rec = json.loads(Path(a.json).read_text())
    out = Path(a.out_dir)
    kinds = [k for k in TITLES if any(r["kind"] == k for r in rec)]
    gaps = sorted({r["gap"] for r in rec})

    fig, axes = plt.subplots(
        1, len(kinds), figsize=(5.2 * len(kinds), 4), sharey=True, facecolor=SURFACE
    )
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        rows = [r for r in rec if r["kind"] == kind]
        style(ax, TITLES[kind])
        plot_lines(
            ax, rows, gaps, [(lambda r, k=k: r[k], lab, c) for k, lab, c in PARTS]
        )
        ax.set_xlim(gaps[0], gaps[-1] + 4.5)
    np.atleast_1d(axes)[0].set_ylabel(
        "share of the model's belief", color=INK2, fontsize=9
    )
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5, loc="center left")
    n_seeds = len({r["seed"] for r in rec})
    fig.suptitle(
        "Where a trained transformer's uncertainty goes when the last 2 was long ago"
        f"\n(model's prediction as a mixture of automaton states; mean of "
        f"{n_seeds} seeds, band = range)",
        fontsize=10.5,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "belief_by_gap.png", dpi=160)

    rows = [r for r in rec if r["kind"] == "parity_hand" and "shift" in r]
    if rows:
        fig, ax = plt.subplots(figsize=(5.6, 4), facecolor=SURFACE)
        style(ax, "Demo automaton: belief by distance on the counter mod 6")
        plot_lines(
            ax,
            rows,
            gaps,
            [(lambda r, k=k: r["shift"][k], lab, c) for k, lab, c in SHIFTS],
        )
        ax.set_xlim(gaps[0], gaps[-1] + 5)
        ax.set_ylabel("share of the model's belief", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="center left")
        fig.tight_layout()
        fig.savefig(out / "belief_by_shift.png", dpi=160)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
