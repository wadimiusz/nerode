"""Bit probe vs. window baselines, by exact gap since the last 2 (26.10.2026).

Run from the repo root: PYTHONPATH=src python experiments/bit_probe_by_gap.py
The MLP probe is trained on all positions; window baselines guess the bit from
the group, the last k symbols and the gap length."""

import importlib.util

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("bp", "experiments/bit_probe.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)
ng = bp.ng
L = bp.SEQ_LEN
BUCKETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, (11, 13), (14, 23)]


def bucket(d):
    for i, b in enumerate(BUCKETS):
        if (isinstance(b, int) and d == b) or (
            isinstance(b, tuple) and b[0] <= d <= b[1]
        ):
            return i
    return -1


acc = {}
for kind in ["parity_hand", "parity"]:
    for seed in range(5):
        p, trained, rnd = bp.load_models(kind, seed, 100_000, 128, 0.1)
        rng = np.random.default_rng(4321 + seed)
        n = 4000
        syms, states = ng.sample(p, n, L, rng)
        x = torch.tensor(np.concatenate([np.full((n, 1), p.n_symbols), syms], 1))
        st = states[:, :L]
        last = np.concatenate([np.full((n, 1), -1), syms[:, : L - 1]], 1)
        keep = np.zeros_like(st, bool)
        keep[:, 1:] = last[:, 1:] != 2
        dist = np.zeros((n, L), int)
        for t in range(1, L):
            dist[:, t] = np.where(syms[:, t - 1] == 2, 0, dist[:, t - 1] + 1)
        W = 14
        pad = np.concatenate([np.full((n, W), -1), syms[:, :L]], 1)
        win = np.stack([pad[:, W - k : W - k + L] for k in range(1, W + 1)], -1)
        bit = (st % 2)[keep]
        grp = (st // 2)[keep]
        d = dist[keep]
        w = win[keep]
        seq = np.repeat(np.arange(n)[:, None], L, 1)[keep]
        tr = seq < int(0.8 * n)
        te = ~tr
        b = np.array([bucket(v) for v in d])
        for mname, model in (("trained", trained), ("random", rnd)):
            res = dict(ng.residuals(model, x[:, :L]))
            for layer in ("block1", "all"):
                h = (
                    res[layer][keep]
                    if layer != "all"
                    else np.concatenate(
                        [res["embed"][keep], res["block0"][keep], res["block1"][keep]],
                        -1,
                    )
                )
                _, pred = bp.fit_probe(h[tr], bit[tr], h[te], bit[te], 2, 256)
                ok = pred == bit[te]
                for i in range(len(BUCKETS)):
                    m = b[te] == i
                    if m.sum():
                        acc.setdefault((mname, layer, i), []).append(ok[m].mean())
        for k in (6, 10, 14):
            keys = np.concatenate(
                [grp[:, None], w[:, :k], np.minimum(d, 30)[:, None]], 1
            )
            table = {}
            for key, y in zip(map(tuple, keys[tr]), bit[tr]):
                table.setdefault(key, [0, 0])[y] += 1
            pred = np.array(
                [int(np.argmax(table.get(key, [1, 0]))) for key in map(tuple, keys[te])]
            )
            ok = pred == bit[te]
            for i in range(len(BUCKETS)):
                m = b[te] == i
                if m.sum():
                    acc.setdefault(("window", k, i), []).append(ok[m].mean())
        for i in range(len(BUCKETS)):
            m = b[te] == i
            mt = b[tr] == i
            if m.sum() and mt.sum():
                acc.setdefault(("majority", "-", i), []).append(
                    (bit[te][m] == int(bit[tr][mt].mean() > 0.5)).mean()
                )
            acc.setdefault(("count", "-", i), []).append(int((b[te] == i).sum()))
        print(kind, seed, "done", flush=True)
lab = [str(x) if isinstance(x, int) else f"{x[0]}-{x[1]}" for x in BUCKETS]
print("dist      ", " ".join(f"{l:>5s}" for l in lab))
for key in [
    ("trained", "block1"),
    ("trained", "all"),
    ("random", "all"),
    ("window", 6),
    ("window", 10),
    ("window", 14),
    ("majority", "-"),
]:
    vals = [
        np.mean(acc.get((key[0], key[1], i), [np.nan])) for i in range(len(BUCKETS))
    ]
    print(f"{key[0][:8] + ' ' + str(key[1]):10s}", " ".join(f"{v:5.2f}" for v in vals))
print(
    "n test    ",
    " ".join(f"{int(np.sum(acc[('count', '-', i)])):5d}" for i in range(len(BUCKETS))),
)
