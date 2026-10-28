"""How well-trained is the parity-automaton model, and what does it take to get
it to near-optimal loss?

For each config: train, report eval loss vs entropy rate (optimum), and the
model's accuracy at recovering the next group (which depends on the hidden bit
after a "2"), by prefix length, both weighted by prefix probability and uniform
over all strings. This script has its own fresh-data loop and accuracy
function; they predate `training.train_fresh` and `readout.parity_bit_accuracy`
and are kept so that results/sweep/ stays reproducible.

Run from the repository root:
  uv run python scripts/train_sweep.py --seeds 0 --steps 3000 10000 --fresh-data
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nerode.automata import entropy_rate, parity_pdfa, sample
from nerode.training import DEVICE, eval_loss, make_model, train


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / np.maximum(q, 1e-9))))


def train_fresh(model, p, steps, lr, batch, seq_len, bos, rng):
    """Like `train`, but every batch is freshly sampled from the automaton."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(steps):
        syms, _ = sample(p, batch, seq_len, rng)
        xb = torch.tensor(np.concatenate([np.full((batch, 1), bos), syms], 1)).to(
            DEVICE
        )
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


@torch.no_grad()
def bit_accuracy(model, p, n_sym, bos, max_len):
    G = np.array([p.emit[2 * g] for g in range(3)])
    out = {}
    for k in range(1, max_len + 1):
        prefixes = list(itertools.product(range(n_sym), repeat=k))
        arr = np.array([[bos] + list(t) for t in prefixes])
        logits = model(torch.tensor(arr).to(DEVICE))[:, -1, :n_sym]
        probs = torch.softmax(logits.float(), -1).cpu().numpy()
        wc = wt = uc = 0.0
        wc2 = wt2 = 0.0
        wkl = 0.0
        for t, pm in zip(prefixes, probs):
            q, w = 0, 1.0
            for a in t:
                w *= p.emit[q, a]
                q = int(p.delta[q, a])
            guess = int(np.argmin([kl(G[g], pm) for g in range(3)]))
            ok = guess == q // 2
            wc += w * ok
            wt += w
            uc += ok
            wkl += w * kl(p.emit[q], pm)
            if t[-1] == 2:
                wc2 += w * ok
                wt2 += w
        out[k] = dict(
            acc_weighted=round(wc / wt, 3),
            acc_after2_weighted=round(wc2 / wt2, 3) if wt2 else None,
            acc_uniform=round(uc / len(prefixes), 3),
            kl_weighted=round(wkl / wt, 4),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, nargs="+", default=[3000])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--fresh-data", action="store_true")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    n_sym, bos, seq_len = 3, 3, 24
    results = []
    for seed in args.seeds:
        for steps in args.steps:
            t0 = time.time()
            rng = np.random.default_rng(seed)
            p = parity_pdfa(3, rng)  # same automaton construction as the demo
            syms, states = sample(p, 6000, seq_len, rng)
            x = torch.tensor(np.concatenate([np.full((len(syms), 1), bos), syms], 1))
            model = make_model(n_sym, seq_len, args.d_model, 2, seed)
            if args.fresh_data:
                train_fresh(
                    model,
                    p,
                    steps,
                    args.lr,
                    args.batch,
                    seq_len,
                    bos,
                    np.random.default_rng(seed + 1000),
                )
            else:
                train(model, x[:5000], steps=steps, lr=args.lr, batch=args.batch)
            loss = eval_loss(model, x[5000:])
            h_opt = entropy_rate(p, states)
            acc = bit_accuracy(model, p, n_sym, bos, args.max_len)
            r = dict(
                seed=seed,
                steps=steps,
                batch=args.batch,
                fresh=args.fresh_data,
                d_model=args.d_model,
                lr=args.lr,
                loss=round(loss, 4),
                h_opt=round(h_opt, 4),
                gap=round(loss - h_opt, 4),
                acc=acc,
                secs=round(time.time() - t0),
            )
            results.append(r)
            print(json.dumps(r), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
