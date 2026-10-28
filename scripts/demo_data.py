"""Export data for the interactive explainer page (parity automaton).

Trains the model on a parity automaton (see nerode.automata) and, for every
prefix up to MAX_LEN symbols, records:
- the true automaton state (group, parity) and Nerode class,
- the true and the model's next-symbol distribution,
- 2-D coordinates of the residual stream at the last position of the prefix,
  and at the last position of the prefix followed by "2" (the symbol that
  reads the parity), both projected with PCA within each group.

Also exports KL(true || model) by position for "easy" and "hard" positions.

Run from the repository root:  uv run python scripts/demo_data.py
Well-trained model on the hand-chosen automaton (needs a GPU for a sane time):
    uv run python scripts/demo_data.py --kind parity_hand --fresh --steps 100000
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from centigpt.model import GPT

from nerode.automata import (
    entropy_rate,
    group_separation,
    hand_parity_pdfa,
    minimise,
    parity_pdfa,
    sample,
)
from nerode.readout import last_residuals, parity_bit_accuracy
from nerode.training import DEVICE, OUT_DIR, eval_loss, fit, make_model

MAX_LEN = 7
SEED = 0
LAYER = 2  # 0 embed, 1 block0, 2 block1, 3 norm_f


def pca2(h: np.ndarray) -> np.ndarray:
    hc = h - h.mean(0)
    _, _, vt = np.linalg.svd(hc, full_matrices=False)
    z = hc @ vt[:2].T
    return z / (np.abs(z).max() + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["parity", "parity_hand"], default="parity")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fresh", action="store_true", help="train on fresh samples")
    ap.add_argument("--min-sep", type=float, default=0.0)
    ap.add_argument("--out", default=str(OUT_DIR / "demo_data.json"))
    args = ap.parse_args()
    seed = args.seed
    rng = np.random.default_rng(seed)
    if args.kind == "parity_hand":
        p = hand_parity_pdfa()
    else:
        p = parity_pdfa(3, rng, min_sep=args.min_sep)
    cls_of_state = minimise(p)
    n_sym, bos, seq_len = 3, 3, 24

    syms, states = sample(p, 6000, seq_len, rng)
    x = torch.tensor(np.concatenate([np.full((len(syms), 1), bos), syms], 1))
    model = make_model(n_sym, seq_len, 64, 2, seed)

    def monitor(m: GPT) -> dict[str, str]:
        acc = parity_bit_accuracy(m, p)
        return {"bit@3": f"{acc[3]:.3f}", "bit@7": f"{acc[7]:.3f}"}

    fit(
        model,
        p,
        x[:5000],
        steps=args.steps,
        seed=seed,
        seq_len=seq_len,
        batch=args.batch,
        fresh=args.fresh,
        desc=f"{args.kind} seed {seed}",
        h_opt=entropy_rate(p, states),
        monitor=monitor,
    )
    loss = eval_loss(model, x[5000:])
    bit_acc = parity_bit_accuracy(model, p, max_len=MAX_LEN)
    h_opt = entropy_rate(p, states)

    prefixes = [
        list(t)
        for k in range(1, MAX_LEN + 1)
        for t in itertools.product(range(n_sym), repeat=k)
    ]
    records = []
    by_len: dict[int, list[int]] = {}
    for i, pr in enumerate(prefixes):
        by_len.setdefault(len(pr), []).append(i)

    # model next-symbol distributions and residuals, batched by length
    probs = np.zeros((len(prefixes), n_sym))
    h_now = np.zeros((len(prefixes), 64))
    h_after2 = np.zeros((len(prefixes), 64))
    with torch.no_grad():
        for k, idx in by_len.items():
            arr = np.array([[bos] + prefixes[i] for i in idx])
            logits = model(torch.tensor(arr).to(DEVICE))[:, -1, :n_sym]
            probs[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
            h_now[idx] = last_residuals(model, arr)[LAYER]
            arr2 = np.concatenate([arr, np.full((len(idx), 1), 2)], 1)
            h_after2[idx] = last_residuals(model, arr2)[LAYER]

    true_state = []
    for pr in prefixes:
        q = 0
        for sym in pr:
            q = int(p.delta[q, sym])
        true_state.append(q)
    true_state_arr = np.array(true_state)
    group = true_state_arr // 2
    parity = true_state_arr % 2

    # PCA within each group, so the plot shows what separates the two parities
    # of the same group (and nothing else)
    xy_now = np.zeros((len(prefixes), 2))
    xy_after = np.zeros((len(prefixes), 2))
    for g in range(3):
        m = group == g
        xy_now[m] = pca2(h_now[m])
        xy_after[m] = pca2(h_after2[m])
    # one PCA over all prefixes, to colour the same cloud by group or by parity
    gl_now = pca2(h_now)
    gl_after = pca2(h_after2)

    for i, pr in enumerate(prefixes):
        q = true_state[i]
        records.append(
            dict(
                s="".join(map(str, pr)),
                g=int(group[i]),
                b=int(parity[i]),
                c=int(cls_of_state[q]),
                pt=[round(float(v), 4) for v in p.emit[q]],
                pm=[round(float(v), 4) for v in probs[i]],
                xn=[round(float(v), 3) for v in xy_now[i]],
                xa=[round(float(v), 3) for v in xy_after[i]],
                gn=[round(float(v), 3) for v in gl_now[i]],
                ga=[round(float(v), 3) for v in gl_after[i]],
            )
        )

    # KL by position, easy vs hard (as in where_errors.py)
    partner = np.arange(p.n_states) ^ 1
    grp = np.arange(p.n_states) // 2
    with torch.no_grad():
        logits = model(x[5000:, :-1].to(DEVICE))
        logp = torch.log_softmax(logits.float(), -1)[..., :n_sym].cpu().numpy()
    st = states[5000:, :seq_len]
    true = p.emit[st]
    kl = (true * (np.log(true) - logp)).sum(-1)
    prev, a = st[:, :-1], syms[5000:, : seq_len - 1]
    hard = grp[p.delta[prev, a]] != grp[p.delta[partner[prev], a]]
    k = kl[:, 1:]
    kl_hard = [
        float(k[:, t][hard[:, t]].mean()) if hard[:, t].any() else None
        for t in range(seq_len - 1)
    ]
    kl_easy = [
        float(k[:, t][~hard[:, t]].mean()) if (~hard[:, t]).any() else None
        for t in range(seq_len - 1)
    ]

    # how separable the hidden parity is, per group, at the two moments
    def sep(xy: np.ndarray) -> float:
        out = []
        for g in range(3):
            m = group == g
            c0, c1 = xy[m & (parity == 0)].mean(0), xy[m & (parity == 1)].mean(0)
            spread = np.linalg.norm(xy[m] - xy[m].mean(0), axis=1).mean()
            out.append(np.linalg.norm(c0 - c1) / spread)
        return float(np.mean(out))

    automaton = dict(
        delta=p.delta.tolist(),
        emit=[[round(float(v), 4) for v in row] for row in p.emit],
    )
    data = dict(
        automaton=automaton,
        loss=round(loss, 4),
        h_opt=round(h_opt, 4),
        training=dict(
            kind=args.kind,
            seed=seed,
            steps=args.steps,
            batch=args.batch,
            fresh=args.fresh,
        ),
        group_separation=round(group_separation(p.emit[0::2]), 4),
        bit_acc={str(k): round(v, 4) for k, v in bit_acc.items()},
        layer="block1",
        max_len=MAX_LEN,
        sep_now=round(sep(xy_now), 3),
        sep_after=round(sep(xy_after), 3),
        kl_hard=[round(v, 4) if v is not None else None for v in kl_hard],
        kl_easy=[round(v, 4) if v is not None else None for v in kl_easy],
        prefixes=records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"loss {loss:.3f} (optimal {h_opt:.3f}); {len(records)} prefixes")
    acc = " ".join(f"{k}:{v:.3f}" for k, v in bit_acc.items())
    print(f"hidden-bit accuracy by prefix length (weighted): {acc}")
    print(f"parity separation: at u {data['sep_now']}, after u+'2' {data['sep_after']}")
    print(f"written {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
