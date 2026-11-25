"""Observable rank of the one-layer machine, from the weights (25.11.2026, exploratory).

State of a prefix p: the masses z[h, a, tau, x] = sum over positions of token x in p
of exp(score[h, a, x, tau + lag]) -- linear in the one-hot prefix u: z = H u.
A future output at shift tau whose position holds token a sees only the block
z[:, a, tau, :] (2 heads x 4 tokens): with the continuation w of length tau
(ending in a) adding its own masses m_w, the mix of each head is
pi_h = (z[h, a, tau] + m_w[h]) / sum, and the output is g(a, pi) (the real W_O,
MLP, norms, unembedding on the mixed reads). So the Jacobian of all future
outputs with respect to z is block diagonal over (a, tau).

Local minimality inside the class the architecture defines: directions of the
state span (columns of H) that no future output can see are the kernel of J H.
Compare rank(J H) with rank(H). tau = 0 is the prefix's own output (query = its
last token, no continuation). Continuations: all strings of length tau - 1
followed by a, tau <= TAU. Nothing is run on data; the prefixes where J is
taken are sampled from the automaton (reachable points only).
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer  # noqa: E402

from nerode.automata import sample  # noqa: E402
from nerode.reads import scores, token_tables  # noqa: E402

BOS = 3


def build_H(sc: np.ndarray, n_lags: int, tau_max: int) -> np.ndarray:
    """H[(h, a, tau, x), (lag, x')] = exp(score[h, a, x, tau + lag]) if x == x'."""
    nh = sc.shape[0]
    H = np.zeros((nh, 3, tau_max + 1, 4, n_lags, 4))
    for h, a, t, x, lag in itertools.product(range(nh), range(3), range(tau_max + 1), range(4), range(n_lags)):
        H[h, a, t, x, lag, x] = np.exp(sc[h, a, x, t + lag])
    return H.reshape(nh * 3 * (tau_max + 1) * 4, n_lags * 4)


def u_of(p: tuple[int, ...], n_lags: int) -> np.ndarray:
    u = np.zeros((n_lags, 4))
    for lag, x in enumerate(reversed(p)):
        u[lag, x] = 1.0
    return u.reshape(-1)


def cont_masses(sc: np.ndarray, a: int, tau: int) -> np.ndarray:
    """(n_w, heads, 4) masses of all continuations of length tau ending in a, seen
    from the query a at the last continuation position (distances 0..tau-1)."""
    out = []
    for w in itertools.product(range(3), repeat=tau - 1):
        w = (*w, a)
        m = np.zeros((sc.shape[0], 4))
        for j, x in enumerate(w):
            m[:, x] += np.exp(sc[:, a, x, tau - 1 - j])
        out.append(m)
    return np.array(out)


class Readout:
    def __init__(self, model):
        self.m = model.double()
        _, _, v = token_tables(model)
        self.v = torch.tensor(v)  # (heads, 4, d_head)

    def g(self, a: int, pi: torch.Tensor) -> torch.Tensor:
        """pi: (k, heads, 4) -> (k, 3) next-symbol distribution (renormalised)."""
        m, blk = self.m, self.m.blocks[0]
        reads = torch.einsum("khx,hxd->khd", pi, self.v).reshape(len(pi), -1)
        x = m.tok_emb.weight[a] + blk.attn.W_O(reads)
        x = x + blk.mlp(blk.norm2(x))
        return torch.softmax(m.lm_head(m.norm_f(x))[:, :3], -1)


def jacobian(ro: Readout, sc, z: np.ndarray, last: int, tau_max: int, conts) -> np.ndarray:
    """Rows: all future outputs (3 probs each); columns: z coordinates (h, a, tau, x)."""
    nh = sc.shape[0]
    zt = z.reshape(nh, 3, tau_max + 1, 4)
    ncol = z.size
    rows = []
    for a in range(3):
        for tau in range(tau_max + 1):
            if tau == 0 and a != last:
                continue
            block = torch.tensor(zt[:, a, tau, :], requires_grad=False)
            mw = torch.zeros((1, nh, 4), dtype=torch.float64) if tau == 0 else torch.tensor(conts[a, tau])

            def f(b):
                tot = b[None] + mw
                return ro.g(a, tot / tot.sum(-1, keepdim=True)).reshape(-1)

            jb = torch.autograd.functional.jacobian(f, block, vectorize=True, strategy="forward-mode").reshape(-1, nh * 4).detach().numpy()
            full = np.zeros((len(jb), ncol))
            idx = [np.ravel_multi_index((h, a, tau, x), (nh, 3, tau_max + 1, 4)) for h in range(nh) for x in range(4)]
            full[:, idx] = jb
            rows.append(full)
    return np.concatenate(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="parity_hand_reset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--n-prefix", type=int, default=6)
    a = ap.parse_args()

    p, model, _ = load_one_layer(a.kind, a.seed, 100_000, 128, 0.1)
    sc = np.asarray(scores(model, 24))
    ro = Readout(model)
    H = build_H(sc, a.lags, a.tau)
    H = H / np.linalg.norm(H, axis=0, keepdims=True)  # column rescaling: rank unchanged, better numerics
    sH = np.linalg.svd(H, compute_uv=False)
    conts = {(x, t): cont_masses(sc, x, t) for x in range(3) for t in range(1, a.tau + 1)}
    rng = np.random.default_rng(0)
    syms, _ = sample(p, a.n_prefix, a.lags - 1, rng)
    print(f"{a.kind} s{a.seed}: H {H.shape}, sing. values / max: "
          + " ".join(f"{v:.0e}" for v in (sH / sH[0])[[0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 47]]))
    # check the formula against the network: prefix + continuation, output at the end
    with torch.no_grad():
        worst = 0.0
        for i in range(a.n_prefix):
            pre = (BOS, *map(int, syms[i]))
            for tau in (1, 3, a.tau):
                w = tuple(int(x) for x in rng.integers(0, 3, tau))
                full = (*pre, *w)
                net = torch.softmax(ro.m(torch.tensor([full]))[0, -1, :3], -1).numpy()
                zf = build_H(sc, a.lags, a.tau) @ u_of(pre, a.lags)
                zf = zf.reshape(sc.shape[0], 3, a.tau + 1, 4)[:, w[-1], tau, :]
                mw = np.zeros_like(zf)
                for j, x in enumerate(w):
                    mw[:, x] += np.exp(sc[:, w[-1], x, tau - 1 - j])
                tot = torch.tensor(zf + mw)[None]
                form = ro.g(w[-1], tot / tot.sum(-1, keepdim=True))[0].numpy()
                worst = max(worst, float(np.abs(net - form).max()))
        print(f"formula vs network, max |diff| = {worst:.1e}")
    for i in range(a.n_prefix):
        pre = (BOS, *map(int, syms[i]))
        z = H @ u_of(pre, a.lags)
        J = jacobian(ro, sc, z, pre[-1], a.tau, conts)
        s = np.linalg.svd(J @ H, compute_uv=False)
        rank_h = int((sH > 1e-8 * sH[0]).sum())
        rank_jh = int((s > 1e-8 * s[0]).sum())
        print(f"prefix {''.join(map(str, pre[1:]))}: rank H {rank_h}, rank JH {rank_jh}; JH sing./max: "
              + " ".join(f"{v:.0e}" for v in (s / s[0])[[0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 47]]), flush=True)


if __name__ == "__main__":
    main()
