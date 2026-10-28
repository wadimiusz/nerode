"""Reading a model: residual streams, next-symbol distributions, the hidden bit."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from centigpt.model import GPT

from nerode.automata import PDFA
from nerode.training import DEVICE


@torch.no_grad()
def residuals(model: GPT, x: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    """Residual stream at every position: after embedding, each block, final norm."""
    model.eval()
    h = model.tok_emb(x.to(DEVICE))
    out = [("embed", h)]
    for i, block in enumerate(model.blocks):
        h = block(h)
        out.append((f"block{i}", h))
    out.append(("norm_f", model.norm_f(h)))
    return [(name, t.float().cpu().numpy()) for name, t in out]


@torch.no_grad()
def last_residuals(model: GPT, seqs: np.ndarray, batch: int = 4096) -> np.ndarray:
    """(n, T) token ids -> (n_layers + 2, n, d): last-position residual after the
    embedding, each block and the final norm."""
    out = []
    for i in range(0, len(seqs), batch):
        x = torch.tensor(seqs[i : i + batch])
        res = residuals(model, x)
        out.append(np.stack([h[:, -1] for _, h in res]))
    return np.concatenate(out, axis=1)


@torch.no_grad()
def future_logprob(
    model: GPT, ctx: np.ndarray, futures: np.ndarray, n_sym: int, batch: int = 4096
) -> np.ndarray:
    """log P_model(z | ctx) for every context and every future z.

    ctx: (n, T) token ids; futures: (m, H). Returns (n, m)."""
    model.eval()
    n, m, h = len(ctx), len(futures), futures.shape[1]
    seqs = np.concatenate([np.repeat(ctx, m, axis=0), np.tile(futures, (n, 1))], axis=1)
    t0 = ctx.shape[1]
    out = np.zeros(len(seqs))
    for i in range(0, len(seqs), batch):
        x = torch.tensor(seqs[i : i + batch]).to(DEVICE)
        logits = model(x[:, :-1]).float()[..., :n_sym]
        logp = torch.log_softmax(logits, -1)  # renormalised over real symbols
        # token at position t0 + k is predicted at position t0 + k - 1
        idx = x[:, t0:]
        lp = logp[:, t0 - 1 : t0 - 1 + h].gather(-1, idx[..., None])[..., 0]
        out[i : i + batch] = lp.sum(-1).cpu().numpy()
    return out.reshape(n, m)


@torch.no_grad()
def parity_bit_accuracy(model: GPT, p: PDFA, max_len: int = 7) -> dict[int, float]:
    """How well the trained model tracks the hidden bit: for every prefix up
    to max_len, guess the true group from the model's next-symbol distribution
    (nearest group emission in KL) and score it, weighting each prefix by its
    probability under the automaton. Prefixes of length k, for k = 1..max_len."""
    model.eval()
    n_sym, bos = p.n_symbols, p.n_symbols
    groups = p.emit[0::2]
    out = {}
    for k in range(1, max_len + 1):
        pre = np.array(list(itertools.product(range(n_sym), repeat=k)))
        arr = np.concatenate([np.full((len(pre), 1), bos), pre], 1)
        logits = model(torch.tensor(arr).to(DEVICE))[:, -1, :n_sym].float()
        pm = torch.softmax(logits, -1).cpu().numpy()
        q = np.zeros(len(pre), dtype=np.int64)
        w = np.ones(len(pre))
        for t in range(k):
            a = pre[:, t]
            w *= p.emit[q, a]
            q = p.delta[q, a]
        kl = (groups[None, :, :] * np.log(groups[None, :, :] / pm[:, None, :])).sum(-1)
        ok = kl.argmin(1) == q // 2
        out[k] = float((w * ok).sum() / w.sum())
    return out
