"""Attention heads of a centigpt GPT: switch one off, or read where it looks.

- `ablate`:            context manager that removes heads from the forward
                       pass, by zeroing their output or replacing it with its
                       mean over reference data (`head_means`);
- `head_means`:        mean output of every head over reference sequences;
- `attention_weights`: the attention pattern of every head (the model calls
                       scaled_dot_product_attention, which does not return it,
                       so it is recomputed from the same queries and keys,
                       with the same rotary embedding and causal mask).

A head's output is its slice of the input to the block's W_O projection,
d_head numbers per position, so both ablations act there.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
from centigpt.model import GPT, Block

from nerode.training import DEVICE

Head = tuple[int, int]  # (layer, head)


def _blocks(model: GPT) -> list[Block]:
    """The model's blocks, typed (nn.ModuleList items are plain Modules)."""
    return [cast(Block, b) for b in model.blocks]


def _pre_proj(model: GPT, x: torch.Tensor) -> list[torch.Tensor]:
    """Input of every layer's W_O, (batch, seq, d_model), for token ids x."""
    out: list[torch.Tensor] = []
    hooks = [
        block.attn.W_O.register_forward_pre_hook(
            lambda _m, inp: out.append(inp[0].detach())
        )
        for block in _blocks(model)
    ]
    try:
        with torch.no_grad():
            model.eval()
            model(x.to(DEVICE))
    finally:
        for h in hooks:
            h.remove()
    return out


def head_means(model: GPT, x: torch.Tensor) -> list[torch.Tensor]:
    """Per layer, the mean of the W_O input over all sequences and positions:
    (d_model,), whose slices are the heads' mean outputs."""
    return [t.mean(dim=(0, 1)) for t in _pre_proj(model, x)]


@contextmanager
def ablate(
    model: GPT, heads: list[Head], means: list[torch.Tensor] | None = None
) -> Iterator[None]:
    """Inside the block, the given heads output zero (means=None) or their mean
    output over reference data (means from `head_means`)."""
    d_head = model.cfg.d_model // model.cfg.n_heads
    hooks = []
    for layer in sorted({h[0] for h in heads}):
        idx = [h[1] for h in heads if h[0] == layer]

        def hook(_m, inp, layer=layer, idx=idx):
            z = inp[0].clone()
            for h in idx:
                sl = slice(h * d_head, (h + 1) * d_head)
                z[..., sl] = 0.0 if means is None else means[layer][sl]
            return (z,)

        hooks.append(_blocks(model)[layer].attn.W_O.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def attention_weights(model: GPT, x: torch.Tensor) -> list[torch.Tensor]:
    """Per layer, (batch, n_heads, seq, seq): row t is where position t looks."""
    model.eval()
    h = model.tok_emb(x.to(DEVICE))
    out = []
    for block in _blocks(model):
        attn = block.attn
        z = block.norm1(h)
        b, t, _ = z.shape
        q = attn.W_Q(z).view(b, t, attn.n_heads, attn.d_head).transpose(1, 2)
        k = attn.W_K(z).view(b, t, attn.n_heads, attn.d_head).transpose(1, 2)
        q, k = attn.apply_rope(q), attn.apply_rope(k)
        scores = q @ k.transpose(-1, -2) / math.sqrt(attn.d_head)
        mask = torch.ones(t, t, dtype=torch.bool, device=scores.device).tril()
        out.append(torch.softmax(scores.masked_fill(~mask, float("-inf")), -1))
        h = block(h)
    return out
