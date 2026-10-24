"""Where is the hidden parity bit kept? Probes on the trained parity models.

`nerode_continuations.py` compares whole residual vectors (cosine distances) and
finds the hidden bit invisible at the current position. A bit kept in a small or
nonlinear part of the vector would slip past that measure, so this script trains
probes directly:

1. For every cached parity model (and a random-weights twin), sample sequences
   from its automaton and take the residual stream at every position. Positions
   right after a "2" are dropped: there the bit is always 0.
2. Train a linear probe (logistic regression) and an MLP probe (two hidden
   layers) to predict the bit, on disjoint train and test sequences.
3. Compare with guessing the bit from the group and the last k symbols
   (k = 1..6), and break the MLP's accuracy down by the number of symbols since
   the last "2". Capacity check: the same MLP predicting the group.
4. Check that the model applies the bit when it is read: right after a "2",
   does its next-symbol distribution point to the right group, by the number of
   symbols since the previous "2"?

The models come from the checkpoint cache, so train them first, e.g.
    uv run python experiments/nerode_continuations.py --kinds parity_hand parity \\
        --seeds 0 1 2 3 4 --fresh --steps 100000 --min-sep 0.1
then, from the repo root:
    uv run python experiments/bit_probe.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "nerode_continuations", Path(__file__).with_name("nerode_continuations.py")
)
assert _spec is not None and _spec.loader is not None
nc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nc)
ng = nc.ng

SEQ_LEN = 24
WINDOW = 6  # longest window of last symbols for the baseline


def _not_cached(*args: Any, **kwargs: Any) -> float:
    raise RuntimeError(
        "model not in the checkpoint cache; train it with "
        "nerode_continuations.py first (see the module docstring)"
    )


def load_models(kind: str, seed: int, steps: int, batch: int, min_sep: float):
    if kind == "parity_hand":
        p = nc.hand_parity_pdfa()
    else:
        p = nc.parity_pdfa(3, np.random.default_rng(seed), min_sep=min_sep)
    trained = ng.make_model(3, SEQ_LEN, 64, 2, seed)
    dummy = torch.zeros((1, SEQ_LEN + 1), dtype=torch.long)
    train_fresh, ng.train_fresh = ng.train_fresh, _not_cached
    try:
        ng.fit(
            trained,
            p,
            dummy,
            steps=steps,
            seed=seed,
            seq_len=SEQ_LEN,
            batch=batch,
            fresh=True,
            desc=f"{kind} seed {seed}",
        )
    finally:
        ng.train_fresh = train_fresh
    random = ng.make_model(3, SEQ_LEN, 64, 2, seed)
    return p, trained, random


def features(model, p, n_seq: int, rng: np.random.Generator) -> dict:
    syms, states = ng.sample(p, n_seq, SEQ_LEN, rng)
    bos = p.n_symbols
    x = torch.tensor(np.concatenate([np.full((n_seq, 1), bos), syms], 1))
    res = ng.residuals(model, x[:, :SEQ_LEN])  # position t has read t symbols
    st = states[:, :SEQ_LEN]  # state after t symbols
    last = np.concatenate([np.full((n_seq, 1), -1), syms[:, : SEQ_LEN - 1]], 1)
    keep = np.zeros_like(st, dtype=bool)
    keep[:, 1:] = last[:, 1:] != 2  # drop BOS and positions right after a 2
    padded = np.concatenate([np.full((n_seq, WINDOW), -1), syms[:, :SEQ_LEN]], 1)
    win = np.stack(
        [padded[:, WINDOW - k : WINDOW - k + SEQ_LEN] for k in range(1, WINDOW + 1)],
        -1,
    )
    dist = np.zeros((n_seq, SEQ_LEN), dtype=np.int64)  # symbols since the last 2
    for t in range(1, SEQ_LEN):
        dist[:, t] = np.where(syms[:, t - 1] == 2, 0, dist[:, t - 1] + 1)
    seq_id = np.repeat(np.arange(n_seq)[:, None], SEQ_LEN, 1)
    return {
        "bit": (st % 2)[keep],
        "group": (st // 2)[keep],
        "win": win[keep],
        "dist": dist[keep],
        "seq": seq_id[keep],
        "layers": {name: h[keep] for name, h in res if name != "embed"},
    }


def window_baseline(group, win, bit, tr, te, k: int) -> float:
    """Best guess of the bit from (group, last k symbols), fit on train."""
    keys = np.concatenate([group[:, None], win[:, :k]], 1)
    table: dict[tuple, list[int]] = {}
    for key, b in zip(map(tuple, keys[tr]), bit[tr]):
        table.setdefault(key, [0, 0])[b] += 1
    pred = np.array(
        [int(np.argmax(table.get(key, [1, 0]))) for key in map(tuple, keys[te])]
    )
    return float((pred == bit[te]).mean())


def fit_probe(xtr, ytr, xte, yte, n_cls: int, hidden: int | None, epochs: int = 12):
    """Linear probe if hidden is None, else an MLP with two hidden layers."""
    torch.manual_seed(0)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr_t = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
    xte_t = torch.tensor((xte - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    d = xtr.shape[1]
    net: nn.Module
    if hidden is None:
        net = nn.Linear(d, n_cls)
    else:
        net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_cls),
        )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    n = len(xtr_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i : i + 1024]
            loss = nn.functional.cross_entropy(net(xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(xte_t).argmax(1).numpy()
    return float((pred == yte).mean()), pred


@torch.no_grad()
def applied_by_gap(model, p, n_seq: int, rng: np.random.Generator) -> dict:
    """Right after each 2: does the model's next-symbol distribution point to
    the right group (nearest group emission in KL)? Keyed by the number of
    symbols since the previous 2 (or since the start), capped at 10."""
    model.eval()
    syms, states = ng.sample(p, n_seq, SEQ_LEN, rng)
    x = torch.tensor(np.concatenate([np.full((n_seq, 1), p.n_symbols), syms], 1))
    logits = model(x[:, :SEQ_LEN].to(ng.DEVICE))[..., : p.n_symbols].float()
    pm = torch.softmax(logits, -1).cpu().numpy()
    groups = p.emit[0::2]
    out: dict[int, list[int]] = {}
    for i in range(n_seq):
        prev = -1
        for t in range(1, SEQ_LEN):
            if syms[i, t - 1] != 2:
                continue
            gap = min(t - 2 - prev, 10)
            prev = t - 1
            kl = (groups * np.log(groups / pm[i, t][None])).sum(-1)
            ok = int(kl.argmin() == states[i, t] // 2)
            counts = out.setdefault(gap, [0, 0])
            counts[0] += ok
            counts[1] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["parity_hand", "parity"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-sep", type=float, default=0.1)
    ap.add_argument("--n-seq", type=int, default=2500)
    ap.add_argument("--out", default="experiments/nerode_out/bit_probe.json")
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
        for layer in ("block0", "block1", "norm_f"):
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
