"""The head tools against the model's own forward pass."""

from __future__ import annotations

import torch

from nerode.heads import _blocks, _pre_proj, ablate, attention_weights, head_means
from nerode.training import DEVICE, make_model


def _model_and_input():
    model = make_model(3, 24, 32, 2, seed=0)
    x = torch.randint(0, 4, (5, 12), generator=torch.Generator().manual_seed(0))
    return model, x.to(DEVICE)


def test_attention_weights_reproduce_the_attention_output() -> None:
    model, x = _model_and_input()
    weights = attention_weights(model, x)
    pre = _pre_proj(model, x)
    with torch.no_grad():
        h = model.tok_emb(x)
        for layer, block in enumerate(_blocks(model)):
            attn = block.attn
            z = block.norm1(h)
            b, t, _ = z.shape
            v = attn.W_V(z).view(b, t, attn.n_heads, attn.d_head).transpose(1, 2)
            mine = (weights[layer] @ v).transpose(1, 2).reshape(b, t, -1)
            assert torch.allclose(mine, pre[layer], atol=1e-5)
            assert torch.allclose(
                weights[layer].sum(-1), torch.ones(b, attn.n_heads, t, device=DEVICE)
            )
            h = block(h)


def test_ablation_changes_only_its_head_and_is_undone() -> None:
    model, x = _model_and_input()
    with torch.no_grad():
        before = model(x)
    d_head = model.cfg.d_model // model.cfg.n_heads
    with ablate(model, [(1, 0)]):
        pre = _pre_proj(model, x)
        with torch.no_grad():
            changed = model(x)
    assert torch.all(pre[1][..., :d_head] == 0)  # head 0 of layer 1 silenced
    assert torch.any(pre[1][..., d_head:] != 0)  # head 1 untouched
    assert not torch.allclose(before, changed)
    with torch.no_grad():
        assert torch.allclose(model(x), before)  # hooks removed
    means = head_means(model, x)
    with ablate(model, [(0, 1)], means):
        pre = _pre_proj(model, x)
    assert torch.allclose(
        pre[0][..., d_head:], means[0][d_head:].expand_as(pre[0][..., d_head:])
    )
