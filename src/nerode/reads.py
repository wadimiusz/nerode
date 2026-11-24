"""The reads matrix of a one-layer model, computed from the weights alone.

In a one-layer model the attention input at every position is the normalised
embedding of that position's token. A head's query, key and value therefore
depend on the token only, and with RoPE the attention score depends only on
the two tokens and the distance between them. A head's read at position t,
where the current token is a, is

    read = sum_s K[a, x_s, t - s] v[x_s]  /  sum_s K[a, x_s, t - s],
    K[a, x, d] = exp(score[a, x, d]),

and a prefix p affects every later read only through its share of these two
sums. Collected over all future queries (token a, offset tau ahead of the end
of p), that share is S(p) = H u(p):

- u(p) is the one-hot record of p, indexed by (lag, token), where lag 0 is the
  last position of p;
- H, the reads matrix, has rows (head, a, tau, component) and columns
  (lag, x), with the entry exp(score[a, x, tau + lag]) * [v[x]; 1]: the first
  d_head components are the numerator, the last one the denominator.

With `project=True` each head's numerator is first multiplied by that head's
block of W_O, so H counts only what reaches the residual stream: directions a
head reads but W_O drops do not add to its rank.

H depends on the weights only. The model's forward pass enters only the tests
that check these formulas against it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from centigpt.model import GPT, Block


def _block(model: GPT) -> Block:
    if model.cfg.n_layers != 1:
        raise ValueError(
            f"the reads matrix is defined for one-layer models; "
            f"this one has {model.cfg.n_layers} layers"
        )
    block = model.blocks[0]
    assert isinstance(block, Block)
    return block


@torch.no_grad()
def token_tables(model: GPT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query, key and value of every token as the attention of the only layer
    sees them, before RoPE: arrays of shape (n_heads, vocab, d_head), float64."""
    block = _block(model)
    attn = block.attn
    x = block.norm1(model.tok_emb.weight)  # (vocab, d_model)

    def per_head(w: torch.nn.Linear) -> np.ndarray:
        y = w(x).reshape(len(x), attn.n_heads, attn.d_head).transpose(0, 1)
        return y.double().cpu().numpy()

    return per_head(attn.W_Q), per_head(attn.W_K), per_head(attn.W_V)


def rope_angles(model: GPT) -> np.ndarray:
    """The RoPE angle per position of each coordinate pair, read from the
    model's own rotation table (the angle at position 1)."""
    freqs_cis = _block(model).attn.freqs_cis  # (1, 1, max_seq_len, d_head / 2)
    return freqs_cis[0, 0, 1].angle().double().cpu().numpy()


def scores(model: GPT, max_dist: int) -> np.ndarray:
    """score[h, a, x, d]: the attention logit of query token a on key token x
    at distance d = 0..max_dist (query position minus key position), with the
    1/sqrt(d_head) scale. RoPE rotates each coordinate pair (2j, 2j + 1) as a
    complex number, so the dot product of the rotated query and key is
    Re sum_j q_j conj(k_j) exp(i d theta_j): only the distance d remains."""
    q, k, _ = token_tables(model)
    d_head = q.shape[-1]
    qc = q[..., 0::2] + 1j * q[..., 1::2]
    kc = k[..., 0::2] + 1j * k[..., 1::2]
    theta = rope_angles(model)
    phase = np.exp(1j * np.outer(np.arange(max_dist + 1), theta))  # (d, pairs)
    s = np.einsum("haj,hxj,dj->haxd", qc, kc.conj(), phase).real
    return s / np.sqrt(d_head)


def head_reads(model: GPT, tokens: np.ndarray) -> np.ndarray:
    """Every head's read (before W_O, heads concatenated as the model does) at
    every position of every sequence, from `scores` and the value table
    instead of the forward pass. tokens: (n, T) ints; returns (n, T, d_model)."""
    n, t_len = tokens.shape
    sc = scores(model, t_len - 1)
    _, _, v = token_tables(model)
    n_heads, _, d_head = v.shape
    out = np.zeros((n, t_len, n_heads, d_head))
    for i in range(n):
        for t in range(t_len):
            past = tokens[i, : t + 1]
            dist = t - np.arange(t + 1)
            logit = sc[:, tokens[i, t], past, dist]  # (heads, t + 1)
            w = np.exp(logit - logit.max(1, keepdims=True))
            w /= w.sum(1, keepdims=True)
            out[i, t] = np.einsum("hs,hsd->hd", w, v[:, past])
    return out.reshape(n, t_len, n_heads * d_head)


def _value_rows(model: GPT, project: bool) -> np.ndarray:
    """[v[h, x]; 1] per head and token, or [W_O_h v[h, x]; 1] with `project`:
    shape (n_heads, vocab, components)."""
    _, _, v = token_tables(model)
    n_heads, vocab, d_head = v.shape
    if project:
        w_o = _block(model).attn.W_O.weight.detach().double().cpu().numpy()
        blocks = w_o.reshape(-1, n_heads, d_head)  # (d_model, h, d_head)
        v = np.einsum("mhd,hxd->hxm", blocks, v)
    return np.concatenate([v, np.ones((n_heads, vocab, 1))], axis=2)


def one_hot_prefix(prefix: Sequence[int], n_lags: int, vocab: int) -> np.ndarray:
    """u(p): entry (lag, x) is 1 if the token `lag` steps back from the last
    position of the prefix is x. Lags beyond the prefix stay 0; a prefix longer
    than n_lags is an error."""
    if len(prefix) > n_lags:
        raise ValueError(f"prefix of length {len(prefix)} > n_lags = {n_lags}")
    u = np.zeros((n_lags, vocab))
    for lag, x in enumerate(reversed(prefix)):
        u[lag, x] = 1.0
    return u.reshape(-1)


def reads_matrix(
    model: GPT,
    n_lags: int,
    n_ahead: int,
    queries: Sequence[int],
    project: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    """H with rows (head, query token a in `queries`, tau = 1..n_ahead,
    component) and columns (lag = 0..n_lags - 1, key token x over the whole
    vocabulary), in that order (row-major). The entry is
    exp(score[h, a, x, tau + lag]) times v[h, x] (or W_O_h v[h, x] with
    `project`) for all components but the last, and times 1 for the last
    (the denominator).

    With `normalize`, each (head, a) block is divided by its largest weight,
    exp(max score over x and distance): a read is a ratio, so scaling every
    weight of one query by the same factor changes no read. This keeps exp
    from overflowing when trained scores run into the hundreds."""
    vocab = model.cfg.vocab_size
    sc = scores(model, n_lags - 1 + n_ahead)
    if normalize:
        sc = sc - sc.max(axis=(2, 3), keepdims=True)
    vx = _value_rows(model, project)  # (h, x, c)
    n_heads, _, n_comp = vx.shape
    tau = np.arange(1, n_ahead + 1)
    lag = np.arange(n_lags)
    dist = tau[:, None] + lag[None, :]  # (tau, lag)
    w = np.exp(sc[:, list(queries)][:, :, :, dist])  # (h, a, x, tau, lag)
    h = np.einsum("haxtl,hxc->hatclx", w, vx)
    return h.reshape(n_heads * len(queries) * n_ahead * n_comp, n_lags * vocab)


def prefix_contribution(
    model: GPT,
    prefix: Sequence[int],
    n_ahead: int,
    queries: Sequence[int],
    project: bool = False,
) -> np.ndarray:
    """S(p) computed directly, as the prefix's share of the numerator and
    denominator of every future read, in the row order of `reads_matrix`.
    The tests check S(p) = H u(p) with this."""
    t = len(prefix)
    sc = scores(model, t - 1 + n_ahead)
    vx = _value_rows(model, project)
    n_heads, _, n_comp = vx.shape
    out = np.zeros((n_heads, len(queries), n_ahead, n_comp))
    for hi in range(n_heads):
        for ai, a in enumerate(queries):
            for ti in range(n_ahead):
                for s, x in enumerate(prefix):
                    k = np.exp(sc[hi, a, x, ti + 1 + (t - 1 - s)])
                    out[hi, ai, ti] += k * vx[hi, x]
    return out.reshape(-1)
