"""Where is the hidden parity bit kept? Probes on the trained parity models.

`continuations.py` compares whole residual vectors (cosine distances) and finds
the hidden bit invisible at the current position. A bit kept in a small or
nonlinear part of the vector would slip past that measure, so these helpers
train probes directly on the residual stream:

- `load_models`:      a trained parity model from the checkpoint cache and its
                      random-weights twin;
- `features`:         residual stream at every position of sampled sequences,
                      with the true bit, group, recent-symbol window and the
                      number of symbols since the last 2. Positions right after
                      a 2 are dropped: there the bit is always 0;
- `fit_probe`:        a linear probe or an MLP (two hidden layers);
- `window_baseline`:  guessing the bit from the group and the last k symbols;
- `applied_by_gap`:   does the model apply the bit when it is read? Right after
                      a 2, does its next-symbol distribution point to the right
                      group, by the number of symbols since the previous 2?

Command line: scripts/bit_probe.py, scripts/bit_probe_by_gap.py,
scripts/bit_probe_long_gaps.py.
"""

from __future__ import annotations

import numpy as np
import torch
from nanogpt.model import GPT
from torch import nn

from nerode.automata import PDFA, hand_parity_pdfa, parity_pdfa, sample
from nerode.readout import residuals
from nerode.training import DEVICE, ckpt_path, make_model

SEQ_LEN = 24
WINDOW = 6  # longest window of last symbols for the baseline


def load_models(
    kind: str, seed: int, steps: int, batch: int, min_sep: float
) -> tuple[PDFA, GPT, GPT]:
    """The parity automaton, the trained model from the checkpoint cache (never
    trained here) and a random-weights twin with the same initialisation."""
    if kind == "parity_hand":
        p = hand_parity_pdfa()
    else:
        p = parity_pdfa(3, np.random.default_rng(seed), min_sep=min_sep)
    trained = make_model(3, SEQ_LEN, 64, 2, seed)
    dummy = torch.zeros((1, SEQ_LEN + 1), dtype=torch.long)
    path = ckpt_path(
        trained,
        p,
        dummy,
        steps=steps,
        seed=seed,
        seq_len=SEQ_LEN,
        batch=batch,
        fresh=True,
    )
    if not path.exists():
        raise RuntimeError(
            f"{kind} seed {seed}: model not in the checkpoint cache ({path}); "
            "train it with scripts/nerode_continuations.py first "
            "(see the README)"
        )
    trained.load_state_dict(torch.load(path, map_location=DEVICE))
    random = make_model(3, SEQ_LEN, 64, 2, seed)
    return p, trained, random


def features(model: GPT, p: PDFA, n_seq: int, rng: np.random.Generator) -> dict:
    syms, states = sample(p, n_seq, SEQ_LEN, rng)
    bos = p.n_symbols
    x = torch.tensor(np.concatenate([np.full((n_seq, 1), bos), syms], 1))
    res = residuals(model, x[:, :SEQ_LEN])  # position t has read t symbols
    st = states[:, :SEQ_LEN]  # state after t symbols
    last = np.concatenate([np.full((n_seq, 1), -1), syms[:, : SEQ_LEN - 1]], 1)
    keep = np.zeros_like(st, dtype=bool)
    keep[:, 1:] = last[:, 1:] != 2  # drop BOS and positions right after a 2
    padded = np.concatenate([np.full((n_seq, WINDOW), -1), syms[:, :SEQ_LEN]], 1)
    win = np.stack(
        [padded[:, WINDOW - k : WINDOW - k + SEQ_LEN] for k in range(1, WINDOW + 1)],
        -1,
    )
    dist = symbols_since_2(syms)
    seq_id = np.repeat(np.arange(n_seq)[:, None], SEQ_LEN, 1)
    return {
        "bit": (st % 2)[keep],
        "group": (st // 2)[keep],
        "win": win[keep],
        "dist": dist[keep],
        "seq": seq_id[keep],
        "layers": _layer_features(res, keep),
    }


def symbols_since_2(syms: np.ndarray) -> np.ndarray:
    """(n_seq, >= SEQ_LEN) symbols -> (n_seq, SEQ_LEN): at position t, the number
    of symbols since the last 2 among the t symbols read so far (t at most)."""
    n_seq = syms.shape[0]
    dist = np.zeros((n_seq, SEQ_LEN), dtype=np.int64)
    for t in range(1, SEQ_LEN):
        dist[:, t] = np.where(syms[:, t - 1] == 2, 0, dist[:, t - 1] + 1)
    return dist


def _layer_features(res, keep) -> dict:
    """Each block and the final norm separately, plus "all": the token
    embedding and every block concatenated, in case the bit is spread across
    layers rather than held in any single one (Shai et al. 2024, RRXOR)."""
    d = {name: h[keep] for name, h in res}
    out = {name: v for name, v in d.items() if name != "embed"}
    out["all"] = np.concatenate(
        [d["embed"]] + [d[n] for n in d if n.startswith("block")], -1
    )
    return out


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
def applied_by_gap(
    model: GPT, p: PDFA, n_seq: int, rng: np.random.Generator
) -> dict[int, list[int]]:
    """Right after each 2: does the model's next-symbol distribution point to
    the right group (nearest group emission in KL)? Keyed by the number of
    symbols since the previous 2 (or since the start), capped at 10."""
    model.eval()
    syms, states = sample(p, n_seq, SEQ_LEN, rng)
    x = torch.tensor(np.concatenate([np.full((n_seq, 1), p.n_symbols), syms], 1))
    logits = model(x[:, :SEQ_LEN].to(DEVICE))[..., : p.n_symbols].float()
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
