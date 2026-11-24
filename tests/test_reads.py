"""The reads matrix: its formulas against the model's own forward pass.

All on a randomly initialised one-layer model: the formulas must hold for any
weights, trained or not."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nerode.probes import SEQ_LEN
from nerode.reads import (
    head_reads,
    one_hot_prefix,
    prefix_contribution,
    reads_matrix,
    scores,
    token_tables,
)
from nerode.training import make_model

VOCAB = 4  # three symbols and BOS
QUERIES = [0, 1, 2]


@pytest.fixture(scope="module")
def model():
    m = make_model(3, SEQ_LEN, 64, 1, seed=0).cpu().eval()
    return m


def random_tokens(n: int, t_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, 3, size=(n, t_len - 1))
    return np.concatenate([np.full((n, 1), 3), syms], 1)


def test_head_reads_match_the_forward_pass(model):
    tokens = random_tokens(5, SEQ_LEN + 1, seed=1)
    block = model.blocks[0]
    w_o = block.attn.W_O.weight.detach().double().numpy()
    ours = head_reads(model, tokens) @ w_o.T
    with torch.no_grad():
        idx = torch.tensor(tokens)
        theirs = block.attn(block.norm1(model.tok_emb(idx))).double().numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("project", [False, True])
def test_reads_matrix_times_one_hot_is_the_prefix_contribution(model, project):
    n_lags, n_ahead = 12, 12
    h = reads_matrix(model, n_lags, n_ahead, QUERIES, project=project)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        length = int(rng.integers(1, n_lags + 1))
        prefix = [3, *rng.integers(0, 3, size=length - 1).tolist()]
        u = one_hot_prefix(prefix, n_lags, VOCAB)
        direct = prefix_contribution(model, prefix, n_ahead, QUERIES, project)
        np.testing.assert_allclose(h @ u, direct, rtol=1e-10, atol=1e-12)


def test_the_prefix_acts_on_later_reads_only_through_its_contribution(model):
    """Read at a later position = (S(p) + the continuation's own share), split
    into numerator and denominator: the prefix enters through S(p) only."""
    t, n_ahead = 9, 10
    tokens = random_tokens(1, t + n_ahead, seed=7)
    seq = tokens[0].tolist()
    prefix = seq[:t]
    s_p = prefix_contribution(model, prefix, n_ahead, QUERIES)
    _, _, v = token_tables(model)
    n_heads, _, d_head = v.shape
    s_p = s_p.reshape(n_heads, len(QUERIES), n_ahead, d_head + 1)
    sc = scores(model, len(seq))
    expected = head_reads(model, tokens)[0].reshape(len(seq), n_heads, d_head)
    for tau in range(1, n_ahead + 1):
        j = t - 1 + tau
        a = seq[j]
        for hi in range(n_heads):
            acc = s_p[hi, a, tau - 1].copy()
            for s in range(t, j + 1):
                k = np.exp(sc[hi, a, seq[s], j - s])
                acc[:d_head] += k * v[hi, seq[s]]
                acc[d_head] += k
            read = acc[:d_head] / acc[d_head]
            np.testing.assert_allclose(read, expected[j, hi], rtol=1e-10, atol=1e-12)


def test_reads_matrix_is_block_hankel(model):
    """Within a block (head, a, component; x), the entry depends on tau + lag."""
    n_lags, n_ahead = 6, 6
    h = reads_matrix(model, n_lags, n_ahead, QUERIES)
    _, _, v = token_tables(model)
    n_heads, _, d_head = v.shape
    h = h.reshape(n_heads, len(QUERIES), n_ahead, d_head + 1, n_lags, VOCAB)
    for tau in range(n_ahead - 1):
        for lag in range(1, n_lags):
            np.testing.assert_allclose(
                h[:, :, tau + 1, :, lag - 1], h[:, :, tau, :, lag]
            )


def test_projected_reads_add_up_to_the_attention_output(model):
    """With `project`, numerator over denominator, summed over heads, is the
    attention output itself (what W_O writes into the residual stream)."""
    n_ahead = 5
    tokens = random_tokens(1, 1 + n_ahead, seed=3)
    prefix = tokens[0, :1].tolist()  # just BOS: then S(p) holds the whole sum
    s_p = prefix_contribution(model, prefix, n_ahead, QUERIES, project=True)
    n_heads = model.cfg.n_heads
    s_p = s_p.reshape(n_heads, len(QUERIES), n_ahead, -1)
    # the read at tau = 1 by a query token a sees only BOS before it plus
    # itself, so check the BOS-only reading against a two-token sequence
    block = model.blocks[0]
    for a in QUERIES:
        idx = torch.tensor([[3, a]])
        with torch.no_grad():
            full = block.attn(block.norm1(model.tok_emb(idx)))[0, 1]
        vx = s_p[:, a, 0]  # BOS's share at distance 1
        sc = scores(model, 1)
        _, _, v = token_tables(model)
        w_o = block.attn.W_O.weight.detach().double().numpy()
        d_head = v.shape[-1]
        out = np.zeros(w_o.shape[0])
        for hi in range(n_heads):
            k_self = np.exp(sc[hi, a, a, 0])
            proj_self = w_o[:, hi * d_head : (hi + 1) * d_head] @ v[hi, a]
            num = vx[hi, :-1] + k_self * proj_self
            den = vx[hi, -1] + k_self
            out += num / den
        np.testing.assert_allclose(out, full.double().numpy(), atol=1e-5, rtol=1e-4)


def test_normalize_only_rescales_each_query_block(model):
    """Normalised H is H with every (head, a) block divided by one number."""
    n_lags, n_ahead = 6, 6
    h = reads_matrix(model, n_lags, n_ahead, QUERIES)
    hn = reads_matrix(model, n_lags, n_ahead, QUERIES, normalize=True)
    n_heads = model.cfg.n_heads
    h = h.reshape(n_heads, len(QUERIES), -1, h.shape[1])
    hn = hn.reshape(n_heads, len(QUERIES), -1, hn.shape[1])
    for hi in range(n_heads):
        for ai in range(len(QUERIES)):
            ratio = h[hi, ai] / hn[hi, ai]
            np.testing.assert_allclose(ratio, ratio.flat[0], rtol=1e-9)


def test_rejects_two_layer_models():
    m = make_model(3, SEQ_LEN, 64, 2, seed=0).cpu()
    with pytest.raises(ValueError):
        scores(m, 3)
