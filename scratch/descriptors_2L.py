"""Two-layer models through position descriptors (24.11.2026, exploratory).

Layer 1's values depend on the token only, so what layer 1 writes into position s is fixed by the
descriptor d_s = (token x_s, pi_s for each layer-1 head), pi_s = the head's attention mass on each
token kind. For layer 2 a position is its descriptor: layer 2's query, key and value are functions
of it. (1) Check: the final output rebuilt from descriptors equals the network's. (2) Explore
parity_hand_reset: strings head + 2 + gap (no 2s) + 2; what layer-1 heads look at inside the gap,
what the descriptors of gap positions encode, where layer 2 looks at the final 2.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from nerode.automata import sample  # noqa: E402
from nerode.probes import load_models  # noqa: E402

BOS = 3


def theta(attn):
    return attn.freqs_cis[0, 0, 1].angle().double().numpy()


def attn_weights(attn, z):
    """z: (T, d) normalised inputs of one string -> softmax weights (H, T, T) and values (H, T, dh)."""
    T = len(z)
    a = attn
    def ph(w):
        return w(z).reshape(T, a.n_heads, a.d_head).transpose(0, 1).double().numpy()
    q, k, v = ph(a.W_Q), ph(a.W_K), ph(a.W_V)
    th = theta(a); rot = np.exp(1j * np.outer(np.arange(T), th))
    qc = (q[..., 0::2] + 1j * q[..., 1::2]) * rot; kc = (k[..., 0::2] + 1j * k[..., 1::2]) * rot
    s = np.einsum("htj,hsj->hts", qc, kc.conj()).real / np.sqrt(a.d_head)
    s = np.where(np.tril(np.ones((T, T), bool))[None], s, -np.inf)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return w, v


@torch.no_grad()
def forward_by_descriptors(m, toks):
    """Returns final probs (T, V), layer-1 pi (H1, T, 4), layer-2 weights (H2, T, T)."""
    b1, b2 = m.blocks
    toks = np.asarray(toks); T = len(toks)
    emb = m.tok_emb.weight
    w1, v1 = attn_weights(b1.attn, b1.norm1(emb[toks]))
    pi = np.zeros((w1.shape[0], T, 4))
    for x in range(4):
        pi[:, :, x] = w1[:, :, toks == x].sum(-1)
    # layer-1 read = sum_x pi_x v(x): values depend on the token only
    vx = np.stack([v1[:, np.where(toks == x)[0][0]] if (toks == x).any() else np.zeros_like(v1[:, 0]) for x in range(4)], 1)
    reads = np.einsum("htx,hxd->thd", pi, vx).reshape(T, -1)
    x1 = emb[toks] + b1.attn.W_O(torch.tensor(reads, dtype=torch.float32))
    x1 = x1 + b1.mlp(b1.norm2(x1))
    w2, v2 = attn_weights(b2.attn, b2.norm1(x1))
    r2 = np.einsum("hts,hsd->thd", w2, v2).reshape(T, -1)
    x2 = x1 + b2.attn.W_O(torch.tensor(r2, dtype=torch.float32))
    x2 = x2 + b2.mlp(b2.norm2(x2))
    return torch.softmax(m.lm_head(m.norm_f(x2)).double(), -1).numpy(), pi, w2


def gap_strings(p, n, rng):
    emit = p.emit.copy(); emit[:, 2] = 0; emit /= emit.sum(1, keepdims=True); cum = emit.cumsum(1)
    out = []
    heads, hs = sample(p, n, 4, rng)
    for i in range(n):
        L0 = int(rng.integers(0, 5)); g = int(rng.integers(1, 17))
        u = [BOS, *heads[i, :L0].tolist(), 2]; q = int(p.delta[hs[i, L0], 2])
        for _ in range(g):
            a = int((rng.random() > cum[q]).sum()); u.append(a); q = int(p.delta[q, a])
        out.append((u + [2], len(u) - g, g))  # string, index of first gap position, gap length
    return out


def main(seed: int = 0, n: int = 1500):
    p, m, _ = load_models("parity_hand_reset", seed, 100_000, 128, 0.1)
    rng = np.random.default_rng(5)
    S = gap_strings(p, n, rng)
    # (1) exactness
    worst = 0.0
    for s, _, _ in S[:200]:
        mine, _, _ = forward_by_descriptors(m, s)
        with torch.no_grad():
            net = torch.softmax(m(torch.tensor([s])).double(), -1)[0].numpy()
        worst = max(worst, 0.5 * np.abs(mine - net).sum(-1).max())
    print(f"seed {seed}: output rebuilt from descriptors vs network: max TV {worst:.2e}")
    # (2) layer 1 inside the gap: attention by offset back from the position; mass on the last 2, BOS
    H1 = m.blocks[0].attn.n_heads
    prof = np.zeros((H1, 12)); cnt = np.zeros(12); on2 = np.zeros(H1); onB = np.zeros(H1); npos = 0
    feats, labels = [], {"run_bit": [], "zeros_last2": [], "dist2": [], "tok": []}
    for s, g0, g in S:
        _, pi, _ = forward_by_descriptors(m, s)
        b1 = m.blocks[0]
        with torch.no_grad():
            w1, _ = attn_weights(b1.attn, b1.norm1(m.tok_emb.weight[np.asarray(s)]))
        last2 = g0 - 1
        for t in range(g0, g0 + g):
            for o in range(min(12, t - last2)):
                prof[:, o] += w1[:, t, t - o]; cnt[o] += 1
            on2 += w1[:, t, last2]; onB += w1[:, t, 0]; npos += 1
            seg = s[g0:t + 1]
            feats.append(np.concatenate([np.eye(3)[s[t]], pi[:, t].ravel()]))
            labels["run_bit"].append(seg.count(0) % 2)
            labels["zeros_last2"].append(s[t - 1:t + 1].count(0) if t - 1 >= g0 else s[t:t + 1].count(0))
            labels["dist2"].append(t - last2)
            labels["tok"].append(s[t])
    print("  layer-1 heads at gap positions: mean weight by offset back (0 = itself, 1 = left neighbour, ...):")
    for h in range(H1):
        print(f"    head {h}: " + " ".join(f"{v:.2f}" for v in prof[h] / np.maximum(cnt, 1))
              + f" | on the last 2: {on2[h] / npos:.2f}, on BOS: {onB[h] / npos:.2f}")
    X = np.array(feats); lab = {k: np.array(v) for k, v in labels.items()}
    np.save(f"/tmp/desc_s{seed}.npy", dict(X=X, **lab), allow_pickle=True)
    # (3) layer 2 at the final 2: weight on gap positions by offset back, on the last 2, on BOS
    H2 = m.blocks[1].attn.n_heads
    prof2 = np.zeros((H2, 17)); cnt2 = np.zeros(17); on2b = np.zeros(H2); onBb = np.zeros(H2)
    for s, g0, g in S:
        _, _, w2 = forward_by_descriptors(m, s)
        t = len(s) - 1
        for o in range(1, g + 1):
            prof2[:, o] += w2[:, t, t - o]; cnt2[o] += 1
        prof2[:, 0] += w2[:, t, t]; cnt2[0] += 1
        on2b += w2[:, t, g0 - 1]; onBb += w2[:, t, 0]
    print("  layer-2 heads at the final 2: mean weight by offset back (0 = the 2 itself, 1..g = gap):")
    for h in range(H2):
        print(f"    head {h}: " + " ".join(f"{v:.2f}" for v in prof2[h] / np.maximum(cnt2, 1))
              + f" | on the previous 2: {on2b[h] / len(S):.2f}, on BOS: {onBb[h] / len(S):.2f}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
