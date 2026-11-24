"""Two-layer models with hard attention: the automaton as a cascade of two one-layer problems
(exploratory).

Layer 1 is the one-layer case: scores depend on (query token, key token, distance), and with argmax
reads the layer's output at a position is fixed by a symbol sigma = (token, winner of head 0, winner of
head 1). After block 1 (attention + MLP) the residual is a function of sigma alone, so layer 2 sees a
finite alphabet: its scores depend on (query sigma, key sigma, distance), and the final output on
(sigma at the position, winning sigma of head 0, winning sigma of head 1). All three tables come from
the weights.

run_product builds two machines, each with the exact one-layer construction (reachable continuation
summaries R): layer 1 over tokens with sigma as the label, layer 2 over sigma strings with the final
output as the label. Their product, run on tokens, is the hard cascade; its states are then merged
when they behave the same on everything still to be read. Concatenating the two signatures instead
(one machine with both labels) keeps every layer-1 distinction, also the ones layer 2 never uses, and
did not finish. Strings are used only to check the result against the network.
"""
import itertools, sys, time, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from nerode.automata import make_automaton, sample
from nerode.training import ckpt_path, make_model
from nerode.probes import SEQ_LEN

T, BOS = 24, 3


def load(kind, seed, steps, layers=2):
    p = make_automaton(kind, seed, 0.1)
    m = make_model(p.n_symbols, SEQ_LEN, 64, layers, seed)
    path = ckpt_path(m, p, torch.zeros((1, SEQ_LEN + 1), dtype=torch.long), steps=steps, seed=seed,
                     seq_len=SEQ_LEN, batch=128, fresh=True)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return p, m.cpu().eval()


def rope_scores(attn, q, k, max_dist):
    """q: (H, Nq, dh), k: (H, Nk, dh) -> (H, Nq, Nk, max_dist + 1), like nerode.reads.scores."""
    theta = attn.freqs_cis[0, 0, 1].angle().double().numpy()
    qc = q[..., 0::2] + 1j * q[..., 1::2]
    kc = k[..., 0::2] + 1j * k[..., 1::2]
    phase = np.exp(1j * np.outer(np.arange(max_dist + 1), theta))
    return np.einsum("haj,hxj,dj->haxd", qc, kc.conj(), phase).real / np.sqrt(q.shape[-1])


def heads(attn, z):
    """z: (N, d_model) normalised inputs -> q, k, v each (H, N, dh), float64."""
    def ph(w):
        y = w(z).reshape(len(z), attn.n_heads, attn.d_head).transpose(0, 1)
        return y.double().numpy()
    return ph(attn.W_Q), ph(attn.W_K), ph(attn.W_V)


class Cascade:
    @torch.no_grad()
    def __init__(self, m):
        b1, b2 = m.blocks[0], m.blocks[1]
        emb = m.tok_emb.weight
        q1, k1, v1 = heads(b1.attn, b1.norm1(emb))
        self.s1 = rope_scores(b1.attn, q1, k1, T)                     # (2, 4, 4, T+1)
        self.sig = list(itertools.product(range(4), range(4), range(4)))   # sigma = (a, w0, w1)
        x1 = []
        for a, w0, w1 in self.sig:
            reads = torch.tensor(np.concatenate([v1[0, w0], v1[1, w1]]), dtype=torch.float32)
            x = emb[a] + b1.attn.W_O(reads)
            x1.append(x + b1.mlp(b1.norm2(x)))
        self.x1 = torch.stack(x1)                                      # (64, d_model)
        q2, k2, v2 = heads(b2.attn, b2.norm1(self.x1))
        self.s2 = rope_scores(b2.attn, q2, k2, T)                     # (2, 64, 64, T+1)
        self.v2, self.b2, self.m = v2, b2, m
        self._out = {}

    @torch.no_grad()
    def out(self, s, u0, u1):
        """Final next-token distribution for sigma s at the position, layer-2 winners u0, u1."""
        key = (s, u0, u1)
        if key not in self._out:
            reads = torch.tensor(np.concatenate([self.v2[0, u0], self.v2[1, u1]]), dtype=torch.float32)
            x = self.x1[s] + self.b2.attn.W_O(reads)
            x = x + self.b2.mlp(self.b2.norm2(x))
            self._out[key] = torch.softmax(self.m.lm_head(self.m.norm_f(x)).double(), -1).numpy()
        return self._out[key]

    def sigmas(self, toks):
        """Layer-1 symbol (index into self.sig) at every position of a token string."""
        toks = np.array(toks); out = []
        for t in range(len(toks)):
            past, dist, a = toks[: t + 1], t - np.arange(t + 1), toks[t]
            w = [past[int(np.argmax(self.s1[h, a, past, dist]))] for h in range(2)]
            out.append(a * 16 + w[0] * 4 + w[1])
        return np.array(out)

    def outputs(self, toks):
        sg = self.sigmas(toks); res = []
        for t in range(len(sg)):
            past, dist, s = sg[: t + 1], t - np.arange(t + 1), sg[t]
            u = [past[int(np.argmax(self.s2[h, s, past, dist]))] for h in range(2)]
            res.append(self.out(s, u[0], u[1]))
        return np.array(res)


@torch.no_grad()
def check_abstraction(c, m, p, n=1000):
    rng = np.random.default_rng(11)
    syms, _ = sample(p, n, T - 1, rng)
    uni = rng.integers(0, 3, (n, T - 1))
    res = []
    for name, S in (("automaton", syms), ("uniform", uni)):
        inp = np.concatenate([np.full((n, 1), BOS), S], 1)
        net = torch.softmax(m(torch.tensor(inp)).double(), -1).numpy()
        tv = np.array([0.5 * np.abs(c.outputs(inp[j]) - net[j]).sum(-1) for j in range(n)])
        res.append(f"{name} strings: TV <= 0.01 at {(tv <= 0.01).mean():.3f} of positions, max {tv.max():.3f}")
    return "; ".join(res)


# ---- exact classes -------------------------------------------------------------------------------

def reach_sets(sc, alphabet, horizon):
    """Generic R: sc[h, q, x, d] over symbols; continuations from `alphabet`. Returns
    R[tau][q] = (vals (k, 2), toks (k, 2)) of reachable continuation summaries."""
    R = {}
    for q in alphabet:
        cur = {((sc[0, q, q, 0], q), (sc[1, q, q, 0], q))}
        R.setdefault(1, {})[q] = cur
        for tau in range(2, horizon + 1):
            d = tau - 1
            cur = {tuple((sc[h, q, y, d], y) if sc[h, q, y, d] >= key[h][0] else key[h] for h in range(2))
                   for key in cur for y in alphabet}
            R.setdefault(tau, {})[q] = cur
    return {(tau, q): (np.array([[k[0][0], k[1][0]] for k in v]), np.array([[k[0][1], k[1][1]] for k in v]))
            for tau, per in R.items() for q, v in per.items()}


def winners(sc, seq, q, tau, vals, toks):
    seq = np.array(seq); n = len(seq); dist = (n - 1 - np.arange(n)) + tau
    w = []
    for h in range(2):
        s = sc[h, q, seq, dist]; i = int(np.argmax(s))
        w.append(np.where(s[i] >= vals[:, h], seq[i], toks[:, h]))
    return w


def bfs(signature, max_classes=5000):
    reps, delta, index = [], {}, {}
    def add(pt, parts):
        ci = len(reps); reps.append(pt)
        for k in range(len(parts)):
            index.setdefault((k, b"".join(parts[: k + 1])), ci)
        return ci
    add((BOS,), signature((BOS,)))
    i = 0
    while i < len(reps):
        r = reps[i]
        if len(r) < T:
            for x in range(3):
                pt = r + (x,); parts = signature(pt)
                ci = index.get((len(parts) - 1, b"".join(parts)))
                delta[i, x] = add(pt, parts) if ci is None else ci
                if len(reps) > max_classes:
                    return reps, delta, False
        i += 1
    return reps, delta, True


# ---- the cascade as a product of two machines -----------------------------------------------------

def tv_clusters(dists, tol):
    """Greedy: each distribution joins the first centre within TV <= tol, else becomes a centre."""
    centres, ids = [], []
    for d in dists:
        for k, c0 in enumerate(centres):
            if 0.5 * np.abs(d - c0).sum() <= tol:
                ids.append(k); break
        else:
            ids.append(len(centres)); centres.append(d)
    return ids, centres


def run_product(c, m, p, tol=0.01, max_classes=200_000):
    """Layer-1 machine (tokens -> sigma) times layer-2 machine (sigma strings -> output), then Moore.

    Layer 1: exact classes with label sigma (as in run_exact). Layer 2: exact classes of the
    one-layer problem over the sigma alphabet, with the final output as label; outputs within TV
    `tol` of a common centre get one label. The product state (layer-1 class, layer-2 class) moves on
    a token x by the layer-1 transition, which emits sigma, and the layer-2 transition on that sigma.
    """
    t0 = time.time()
    R1 = reach_sets(c.s1, [0, 1, 2], T - 1)

    def sig1(p_toks):
        n = len(p_toks); parts = [np.array([c.sigmas(p_toks)[-1]]).tobytes()]
        for tau in range(1, T - n + 1):
            chunk = []
            for a in range(3):
                vals, toks = R1[tau, a]
                w0, w1 = winners(c.s1, p_toks, a, tau, vals, toks)
                chunk.append(a * 16 + w0 * 4 + w1)
            parts.append(np.concatenate(chunk).astype(np.int64).tobytes())
        return parts

    reps1, delta1, done1 = bfs(sig1, max_classes)
    if not done1:
        print(f"  layer 1: more than {max_classes} exact classes, stopped"); return
    emit1 = [int(c.sigmas(r)[-1]) for r in reps1]
    bos = emit1[0]
    alph2 = sorted(set(emit1) - {bos})
    R2 = reach_sets(c.s2, alph2, T - 1)

    # output labels: one per cluster of final distributions
    syms = alph2 + [bos]
    trip = [(s, u0, u1) for s in syms for u0 in syms for u1 in syms]
    ids, centres = tv_clusters([c.out(*t) for t in trip], tol)
    L = np.full((64, 64, 64), -1, dtype=np.int64)
    for t, k in zip(trip, ids):
        L[t] = k
    print(f"  layer 1: {len(reps1)} exact classes, {len(alph2)} reachable sigmas "
          f"{[(s // 16, (s // 4) % 4, s % 4) for s in alph2]}; {len(trip)} output cells -> "
          f"{len(centres)} labels at TV {tol}; {time.time() - t0:.0f} s", flush=True)

    def sig2(sg):
        sg = np.array(sg); n = len(sg); s = sg[-1]
        u = [sg[int(np.argmax(c.s2[h, s, sg, n - 1 - np.arange(n)]))] for h in range(2)]
        parts = [np.array([L[s, u[0], u[1]]]).tobytes()]
        for tau in range(1, T - n + 1):
            chunk = []
            for q in alph2:
                vals, toks = R2[tau, q]
                u0, u1 = winners(c.s2, sg, q, tau, vals, toks)
                chunk.append(L[q, u0, u1])
            parts.append(np.concatenate(chunk).tobytes())
        return parts

    # layer-2 classes, built lazily from the sigma strings the product actually reaches
    reps2, own2, index2, delta2 = [], [], {}, {}
    # as in bfs(): a longer string joins an earlier (shorter) class when its signature equals the
    # first parts of that class's signature, so classes are not tied to a length
    def cls2(sg):
        parts = sig2(sg); ci = index2.get((len(parts) - 1, b"".join(parts)))
        if ci is None:
            ci = len(reps2); reps2.append(tuple(sg)); own2.append(int(np.frombuffer(parts[0], np.int64)[0]))
            for k in range(len(parts)):
                index2.setdefault((k, b"".join(parts[: k + 1])), ci)
        return ci
    def step2(j, s):
        if (j, s) not in delta2:
            delta2[j, s] = cls2(reps2[j] + (s,))
        return delta2[j, s]

    # product states carry their length n (a move exists while n < T)
    start = (0, cls2((bos,)), 1)
    pairs, pid, delta = [start], {start: 0}, {}
    i = 0
    while i < len(pairs):
        a, j, n = pairs[i]
        if n < T:
            for x in range(3):
                a2 = delta1[a, x]; nxt = (a2, step2(j, emit1[a2]), n + 1)
                if nxt not in pid:
                    pid[nxt] = len(pairs); pairs.append(nxt)
                delta[i, x] = pid[nxt]
        i += 1
    own = [own2[j] for a, j, n in pairs]

    # behaviour of a state up to h more symbols, as an id: H[k][h]; computed from the longest states
    ids, H = {}, [None] * len(pairs)
    for k in sorted(range(len(pairs)), key=lambda k: -pairs[k][2]):
        hmax = T - pairs[k][2]; row = [ids.setdefault((own[k],), len(ids))]
        for h in range(1, hmax + 1):
            row.append(ids.setdefault((own[k],) + tuple(H[delta[k, x]][h - 1] for x in range(3)), len(ids)))
        H[k] = row
    # classes: a state joins the first earlier state (BFS order, so not longer) that behaves the same
    # on everything the state still has to read; the earlier state becomes the representative
    cls, index, rep = [0] * len(pairs), {}, []
    for k in range(len(pairs)):
        hk = T - pairs[k][2]; c0 = index.get((hk, H[k][hk]))
        if c0 is None:
            c0 = len(rep); rep.append(k)
            for h in range(hk + 1):
                index.setdefault((h, H[k][h]), c0)
        cls[k] = c0
    mdelta = {(c0, x): cls[delta[r, x]] for c0, r in enumerate(rep) for x in range(3) if (r, x) in delta}
    mlab = [own[r] for r in rep]
    print(f"  layer 2: {len(reps2)} classes over sigma strings; product: {len(pairs)} (state, length) pairs, "
          f"{len(rep)} classes; {time.time() - t0:.0f} s", flush=True)

    # check against the network on strings from the automaton (strings used only here)
    rng = np.random.default_rng(7)
    S, _ = sample(p, 1000, T - 1, rng)
    inp = np.concatenate([np.full((1000, 1), BOS), S], 1)
    with torch.no_grad():
        net = torch.softmax(m(torch.tensor(inp)).double(), -1).numpy()
    tvs = []
    for j in range(1000):
        k = 0
        for t in range(T):
            if t > 0:
                k = mdelta[k, inp[j, t]]
            tvs.append(0.5 * np.abs(centres[mlab[k]] - net[j, t]).sum())
    tvs = np.array(tvs)
    print(f"  machine vs network: TV <= 0.01 at {(tvs <= 0.01).mean():.3f}, <= 0.02 at {(tvs <= 0.02).mean():.3f} "
          f"of positions, max {tvs.max():.3f}", flush=True)
    return dict(reps1=reps1, delta1=delta1, emit1=emit1, alph2=alph2, reps2=reps2, delta2=delta2,
                pairs=pairs, delta=delta, rep=rep, mdelta=mdelta, mlab=mlab, centres=centres)


if __name__ == "__main__":
    kind, steps = sys.argv[1], int(sys.argv[2])
    for seed in [int(s) for s in sys.argv[3].split(",")]:
        p, m = load(kind, seed, steps)
        c = Cascade(m)
        print(f"{kind} seed {seed} (2 layers, {steps} steps): hard cascade vs network: {check_abstraction(c, m, p)}",
              flush=True)
        run_product(c, m, p)
