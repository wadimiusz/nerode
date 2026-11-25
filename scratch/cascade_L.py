"""Hard-attention cascade on L layers: diagnostics before building machines (24.11.2026).

The symbol of layer l at a position is (its symbol at layer l-1, the layer-(l-1) symbols that each
head of layer l picks by argmax). The residual after layer l is a function of that symbol alone,
built from the weights: x_l = x_{l-1} + W_O[v(w_0); v(w_1)], then + MLP. Layer 0's symbol is the token.

Per model this prints, on 1000 strings from the automaton and 1000 uniform strings (strings are used
only for this check, not for the construction):
- how often the hard cascade's final distribution is within TV 0.01 of the network's;
- per layer, the share of (position, head) reads with a small margin: best score minus the best score
  of a key whose symbol differs from the winner's;
- whether the positions where the cascade and the network disagree are ones with a small margin;
- the number of distinct symbols per layer seen on these strings.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from nerode.automata import make_automaton, sample
from nerode.probes import SEQ_LEN
from nerode.training import ckpt_path, make_model

T = 24  # BOS + 23 symbols, as in cascade2.py


def load(kind: str, seed: int, steps: int, layers: int):
    p = make_automaton(kind, seed, 0.1)
    m = make_model(p.n_symbols, SEQ_LEN, 64, layers, seed)
    path = ckpt_path(m, p, torch.zeros((1, SEQ_LEN + 1), dtype=torch.long), steps=steps, seed=seed,
                     seq_len=SEQ_LEN, batch=128, fresh=True)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return p, m.cpu().eval()


class HardCascade:
    def __init__(self, m):
        self.m = m
        self.L = len(m.blocks)
        # per layer: symbol tuple -> index; residual vectors of the symbols of that layer
        self.ids = [dict() for _ in range(self.L + 1)]
        self.x = [[] for _ in range(self.L + 1)]
        emb = m.tok_emb.weight.detach()
        for t in range(emb.shape[0]):
            self.ids[0][(t,)] = t
            self.x[0].append(emb[t])
        self.qkv = [None] * self.L  # cached per-layer (q, k, v) arrays for known symbols
        self._outs = {}

    @torch.no_grad()
    def _qkv(self, layer: int, syms: np.ndarray):
        blk = self.m.blocks[layer]
        a = blk.attn
        z = blk.norm1(torch.stack([self.x[layer][s] for s in syms]))
        def ph(w):
            return w(z).reshape(len(syms), a.n_heads, a.d_head).transpose(0, 1).double().numpy()
        return ph(a.W_Q), ph(a.W_K), ph(a.W_V)

    @torch.no_grad()
    def run(self, toks: np.ndarray):
        """Returns final distributions (T, vocab) and margins (L, T, heads)."""
        sym = np.array(toks)
        n = len(sym)
        margins = np.full((self.L, n, 2), np.inf)
        for layer in range(self.L):
            blk = self.m.blocks[layer]
            a = blk.attn
            theta = a.freqs_cis[0, 0, 1].angle().double().numpy()
            uniq, inv = np.unique(sym, return_inverse=True)
            q, k, v = self._qkv(layer, uniq)
            qc = (q[..., 0::2] + 1j * q[..., 1::2])[:, inv]  # (H, n, dh/2)
            kc = (k[..., 0::2] + 1j * k[..., 1::2])[:, inv]
            ph = np.exp(1j * np.outer(np.arange(n), theta))
            sc = np.einsum("htj,hsj->hts", qc * ph, (kc * ph).conj()).real / np.sqrt(q.shape[-1])
            sc = np.where(np.tril(np.ones((n, n), bool))[None], sc, -np.inf)
            new = np.empty(n, dtype=np.int64)
            for t in range(n):
                win = []
                for h in range(a.n_heads):
                    s = int(np.argmax(sc[h, t]))
                    win.append(s)
                    other = sc[h, t, : t + 1][sym[: t + 1] != sym[s]]
                    if len(other):
                        margins[layer, t, h] = sc[h, t, s] - other.max()
                key = (int(sym[t]), *[int(sym[s]) for s in win])
                if key not in self.ids[layer + 1]:
                    reads = torch.tensor(np.concatenate([v[h, inv[s]] for h, s in enumerate(win)]),
                                         dtype=torch.float32)
                    xx = self.x[layer][sym[t]] + a.W_O(reads)
                    xx = xx + blk.mlp(blk.norm2(xx))
                    self.ids[layer + 1][key] = len(self.x[layer + 1])
                    self.x[layer + 1].append(xx)
                new[t] = self.ids[layer + 1][key]
            sym = new
        out = []
        for s in sym:
            if s not in self._outs:
                self._outs[s] = torch.softmax(self.m.lm_head(self.m.norm_f(self.x[self.L][s])).double(), -1).numpy()
            out.append(self._outs[s])
        return np.array(out), margins, sym


@torch.no_grad()
def diagnose(kind: str, seed: int, steps: int, layers: int, n: int = 1000) -> None:
    p, m = load(kind, seed, steps, layers)
    c = HardCascade(m)
    rng = np.random.default_rng(11)
    syms, st = sample(p, n, T - 1, rng)
    uni = rng.integers(0, 3, (n, T - 1))
    line = [f"{kind} {layers}L s{seed}"]
    for name, S in (("aut", syms), ("uni", uni)):
        inp = np.concatenate([np.full((n, 1), p.n_symbols), S], 1)
        net = torch.softmax(m(torch.tensor(inp)).double(), -1).numpy()
        tvs, mins, per_layer = [], [], []
        for j in range(n):
            out, mg, _ = c.run(inp[j])
            tvs.append(0.5 * np.abs(out - net[j]).sum(-1))
            mins.append(mg.min(axis=(0, 2)))
            per_layer.append(mg.min(axis=2))  # (L, T)
        tvs, mins, per_layer = np.concatenate(tvs), np.concatenate(mins), np.concatenate(per_layer, 1)
        bad = tvs > 0.01
        small = mins < 2
        line.append(f"{name}: match {1 - bad.mean():.3f} (max TV {tvs.max():.3f}); "
                    f"margin<2 by layer " + "/".join(f"{(per_layer[l] < 2).mean():.3f}" for l in range(layers))
                    + (f"; mismatches with a margin<2: {small[bad].mean():.2f}" if bad.any() else ""))
        if name == "aut":
            h_true = -(p.emit[st[:, :-1]] * np.log(net[:, :-1, :3] + 1e-30)).sum(-1).mean()  # cross-entropy to truth
            h_opt = -(p.emit[st[:, :-1]] * np.log(p.emit[st[:, :-1]])).sum(-1).mean()
            line.append(f"gap {h_true - h_opt:+.4f}")
    line.append("symbols per layer " + "/".join(str(len(d)) for d in c.ids))
    print(" | ".join(line), flush=True)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000
    for layers in (2, 3, 4):
        for seed in (0, 1, 2):
            diagnose("flipflop", seed, steps, layers)
