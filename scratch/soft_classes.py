"""Soft attention: classes of prefixes up to a tolerance eps, from the weights (exploratory).

Read of head h at a position = sum_x pi_x v_h(x): a head's value depends on the key token only, so
the read is a mix of the 4 token values, pi = softmax over tokens of the log-masses
L_x = logsumexp over occurrences of x of the scores s_h(a, x, d). After attention everything acts
on one position, so the output is g(a, pi_0, pi_1), computed by the real layers (W_O, MLP, norms,
unembedding) on the mixed reads. Hard attention is the limit where pi is a vertex.

Prefix p, continuation w of length tau ending in a: the masses add, L = logaddexp(L_p, L_w), BOS
mass from the prefix only. L_w ranges over the soft R (soft_R_size.py): reachable continuation
log-masses, deduplicated on a grid of width eta; every element is an actual continuation (a witness),
so a distance computed on it is a lower bound on the distance over all continuations.

d(p, q) = max over tau <= 24 - len, a, w in soft R of TV(output(p w), output(q w)), plus the own
outputs. Classes: breadth-first search from BOS; a new prefix joins the first class whose
representative is within eps, else opens a class. Strings are used only to check the result.
"""
import sys, time, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.reads import scores, token_tables
from nerode.automata import make_automaton, sample
from soft_R_size import CLIP

T, BOS = 24, 3


def soft_R(sc, a, eta, tau_max):
    out = {}
    L = np.full((1, 2, 3), -np.inf)
    for h in range(2):
        L[0, h, a] = sc[h, a, a, 0]
    out[1] = L
    for tau in range(2, tau_max + 1):
        d = tau - 1
        new = np.repeat(L[:, None], 3, 1)
        for y in range(3):
            for h in range(2):
                new[:, y, h, y] = np.logaddexp(new[:, y, h, y], sc[h, a, y, d])
        new = new.reshape(-1, 2, 3)
        top = new.max(2, keepdims=True)
        new = np.where(new < top - CLIP, -np.inf, new)
        key = np.where(np.isfinite(new), np.round(new / eta), -10**9).astype(np.int64).reshape(len(new), -1)
        _, first = np.unique(key, axis=0, return_index=True)
        L = new[np.sort(first)]
        out[tau] = L
    return out


def logsumexp_by_token(s, toks):
    """s, toks: (n,) scores and tokens -> (4,) log-mass per token (-inf if absent)."""
    out = np.full(4, -np.inf)
    for x in range(4):
        sel = s[toks == x]
        if len(sel):
            mx = sel.max(); out[x] = mx + np.log(np.exp(sel - mx).sum())
    return out


class Soft:
    def __init__(self, model, eta, beta=1.0):
        """beta scales every attention score (beta -> infinity is the hard limit)."""
        self.m = model
        self.sc = beta * np.asarray(scores(model, T))
        _, _, v = token_tables(model)
        self.v = v                                        # (2 heads, 4 tokens, d_head)
        self.R = {a: soft_R(self.sc, a, eta, T - 1) for a in range(3)}

    @torch.no_grad()
    def g(self, a, pi):
        """a: int, pi: (k, 2, 4) -> (k, 4) next-token distributions."""
        m, blk = self.m, self.m.blocks[0]
        reads = np.einsum("khx,hxd->khd", pi, self.v).reshape(len(pi), -1)
        x = m.tok_emb.weight[a] + blk.attn.W_O(torch.tensor(reads, dtype=torch.float32))
        x = x + blk.mlp(blk.norm2(x))
        return torch.softmax(m.lm_head(m.norm_f(x)).double(), -1).numpy()

    def prefix_mass(self, p, a, tau):
        p = np.array(p); n = len(p); dist = (n - 1 - np.arange(n)) + tau
        return np.stack([logsumexp_by_token(self.sc[h, a, p, dist], p) for h in range(2)])

    def own_output(self, p):
        L = self.prefix_mass(p, p[-1], 0)
        pi = np.exp(L - L.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
        return self.g(p[-1], pi[None])[0]

    def part(self, p, k):
        """Part k of the signature: k = 0 the own output (1, 4); k = tau >= 1 the outputs (n, 4)
        over all a and soft-R elements."""
        if k == 0:
            return self.own_output(p)[None]
        chunk = []
        for a in range(3):
            Lp = self.prefix_mass(p, a, k)
            Lw = self.R[a][k]
            comb = np.empty((len(Lw), 2, 4))
            comb[:, :, :3] = np.logaddexp(Lp[None, :, :3], Lw)
            comb[:, :, 3] = Lp[None, :, 3]
            pi = np.exp(comb - comb.max(2, keepdims=True)); pi /= pi.sum(2, keepdims=True)
            chunk.append(self.g(a, pi))
        return np.concatenate(chunk)


class Sig:
    """Signature of a prefix, computed part by part when asked (parts 0 .. 24 - len)."""
    def __init__(self, soft, p):
        self.soft, self.p, self.parts = soft, p, {}
        self.n_parts = T - len(p) + 1

    def get(self, k):
        if k not in self.parts:
            self.parts[k] = self.soft.part(self.p, k)
        return self.parts[k]


def within(new, rep, eps):
    """Max TV over the parts the new (longer or equal) prefix has is <= eps; parts are computed
    lazily and the check stops at the first part that exceeds eps."""
    for k in range(new.n_parts):
        if 0.5 * np.abs(new.get(k) - rep.get(k)).sum(-1).max() > eps:
            return False
    return True


def build(soft, eps, max_classes=1500):
    reps, sigs, delta = [(BOS,)], [Sig(soft, (BOS,))], {}
    i = 0
    while i < len(reps):
        r = reps[i]
        if len(r) < T:
            for x in range(3):
                p = r + (x,); s = Sig(soft, p)
                c = next((j for j, sr in enumerate(sigs) if within(s, sr, eps)), None)
                if c is None:
                    c = len(reps); reps.append(p); sigs.append(s)
                    if len(reps) > max_classes:
                        return reps, sigs, delta, False
                delta[i, x] = c
        i += 1
    return reps, sigs, delta, True


@torch.no_grad()
def check(soft, model, kind, seed, reps, sigs, delta, n=2000):
    """Machine output (own output of the class representative) vs network and vs true state."""
    rng = np.random.default_rng(11)
    syms, states = sample(make_automaton(kind, seed, 0.1), n, T - 1, rng)
    inp = np.concatenate([np.full((n, 1), BOS), syms], 1)
    net = torch.softmax(model(torch.tensor(inp)).double(), -1).numpy()
    tvs, cls = [], np.zeros((n, T), dtype=np.int64)
    for j in range(n):
        c = 0
        for t in range(T):
            if t > 0:
                c = delta[c, inp[j, t]]
            cls[j, t] = c
            tvs.append(0.5 * np.abs(sigs[c].get(0)[0] - net[j, t]).sum())
    tvs = np.array(tvs)
    st = states[:, :T]
    pure = 0
    for c in np.unique(cls):
        pure += np.bincount(st[cls == c]).max()
    return tvs, pure / cls.size, len(np.unique(cls))


@torch.no_grad()
def exactness(soft, model, n=200):
    """The mixed-reads formula against the network on random strings (should be ~1e-6)."""
    rng = np.random.default_rng(0)
    inp = np.concatenate([np.full((n, 1), BOS), rng.integers(0, 3, (n, T - 1))], 1)
    net = torch.softmax(model(torch.tensor(inp)).double(), -1).numpy()
    err = 0.0
    for j in range(n):
        for t in range(T):
            err = max(err, 0.5 * np.abs(soft.own_output(tuple(inp[j, : t + 1])) - net[j, t]).sum())
    return err


if __name__ == "__main__":
    kind, seed, eta = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    epss = [float(e) for e in sys.argv[4:]] or [0.1, 0.03, 0.01]
    _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
    soft = Soft(m, eta)
    print(f"{kind} seed {seed}, eta {eta}: |soft R| at tau=23 = "
          f"{[len(soft.R[a][23]) for a in range(3)]}; formula vs network, max TV {exactness(soft, m):.1e}",
          flush=True)
    for eps in epss:
        t0 = time.time()
        reps, sigs, delta, done = build(soft, eps)
        if not done:
            print(f"  eps {eps}: more than {len(reps) - 1} classes, stopped"); continue
        tvs, purity, used = check(soft, m, kind, seed, reps, sigs, delta)
        print(f"  eps {eps}: {len(reps)} classes ({used} visited by 2000 sampled strings), purity vs true "
              f"state {purity:.3f}; machine vs network TV: median {np.median(tvs):.4f}, <=0.01 "
              f"{(tvs <= 0.01).mean():.3f}, <=0.03 {(tvs <= 0.03).mean():.3f}, max {tvs.max():.3f}; "
              f"{time.time() - t0:.0f} s", flush=True)
