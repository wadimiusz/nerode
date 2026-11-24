"""Exact Nerode classes of the hard-attention abstraction, from weights only, without
enumerating continuations (no 3^tau).

Setting. Hard abstraction: at every position each head reads the token with the largest score
s_h(a, x, d) (a = query token at that position, x = key token, d = distance); the output label is
lab[a, winner_0, winner_1] (finite 4x4x4 table from the weights, see extract_bfs.Teacher).

Key observation. Take a prefix p (length n, BOS first) and a continuation w of length tau >= 1
whose last token is a. At the last position, head h compares
  - the best prefix entry:        m_p(h, a, tau) = max_i s_h(a, p_i, (n-1-i) + tau), with its token;
  - the best continuation entry:  c_h(w) = max_{d=0..tau-1} s_h(a, w at distance d, d), its token.
The prefix wins if m_p >= c_h (it sits earlier, and argmax takes the earliest position on ties).
So the label depends on p only through (m_p, token) per head, and on w only through
(c_h, token) per head. The set R_tau(a) of all pairs ((c_0, t_0), (c_1, t_1)) reachable by some w
is computed from the score table alone, one distance at a time, and does not depend on p; it has at
most (3 tau + 1)^2 elements. Every element keeps one witness string.

Signature of p (length n) = its own label (tau = 0) and the labels for every element of
R_tau(a), for all a in {0,1,2} and tau = 1..T-n: only continuations that fit into the training
length, so every distance used is < T. (A first version used tau up to T-1 for every prefix; the
scores at distances beyond the training length then created hundreds of spurious classes.)
The BFS machine (see build) is correct on every string of length <= T by induction; the sampled
strings below are a final check only.
"""
import sys, time, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.reads import scores
from extract_bfs import Teacher, verify, T, BOS

H = T - 1


def continuation_sets(sc, horizon):
    """R[tau][a] = (vals (k, 2), toks (k, 2), witnesses list of tuples w_1..w_tau)."""
    R = {}
    for a in range(3):
        cur = {((sc[0, a, a, 0], a), (sc[1, a, a, 0], a)): (a,)}
        R.setdefault(1, {})[a] = cur
        for tau in range(2, horizon + 1):
            d = tau - 1                     # the new token sits at distance tau - 1 (earliest)
            nxt = {}
            for key, wit in cur.items():
                for y in range(3):
                    new = tuple(
                        (sc[h, a, y, d], y) if sc[h, a, y, d] >= key[h][0] else key[h]
                        for h in range(2))
                    nxt.setdefault(new, (y,) + wit)
            cur = nxt
            R.setdefault(tau, {})[a] = cur
    out = {}
    for tau, per_a in R.items():
        for a, dct in per_a.items():
            keys = list(dct)
            vals = np.array([[k[0][0], k[1][0]] for k in keys])
            toks = np.array([[k[0][1], k[1][1]] for k in keys], dtype=np.int64)
            out[tau, a] = (vals, toks, list(dct.values()))
    return out


class ExactSig:
    def __init__(self, model, horizon=H):
        self.teacher = Teacher(model)
        self.lab = self.teacher.lab
        self.horizon = horizon
        self.sc = np.asarray(scores(model, T))
        self.R = continuation_sets(self.sc, horizon)

    def own_label(self, p):
        return int(self.teacher.labels(np.array([p]), len(p) - 1)[0, 0])

    def signature(self, p, horizon=None):
        """List of label arrays: [own label, tau=1, ..., tau=horizon]; default horizon T - n,
        so every distance stays inside the training length."""
        p = np.array(p); n = len(p)
        horizon = T - n if horizon is None else horizon
        parts = [np.array([self.own_label(p)], dtype=np.int8).tobytes()]
        for tau in range(1, horizon + 1):
            dist = (n - 1 - np.arange(n)) + tau
            chunk = []
            for a in range(3):
                vals, toks, _ = self.R[tau, a]
                win = []
                for h in range(2):
                    s = self.sc[h, a, p, dist]
                    i = int(np.argmax(s))            # earliest position on ties
                    m, tp = s[i], p[i]
                    win.append(np.where(m >= vals[:, h], tp, toks[:, h]))
                chunk.append(self.lab[a, win[0], win[1]])
            parts.append(np.concatenate(chunk).astype(np.int8).tobytes())
        return parts


def build(es):
    """BFS over representatives. A candidate rep(c).x of length n is sent to the first class
    whose representative agrees with it on every continuation of length <= T - n (labels on
    R_tau for tau <= T - n). BFS discovers each class first through a shortest string, so any
    string q that ends in class c is at least as long as rep(c); by induction q and rep(c) agree on
    every continuation that still fits into the training length (journal, 18.11.2026)."""
    reps, delta = [], {}
    index = {}                  # (horizon k, signature bytes up to k) -> first class
    def add(p, parts):
        c = len(reps); reps.append(p)
        for k in range(len(parts)):
            index.setdefault((k, b"".join(parts[: k + 1])), c)
        return c
    add((BOS,), es.signature((BOS,)))
    i = 0
    while i < len(reps):
        r = reps[i]
        if len(r) < T:
            for x in range(3):
                p = r + (x,)
                parts = es.signature(p)
                k = len(parts) - 1
                c = index.get((k, b"".join(parts)))
                delta[i, x] = add(p, parts) if c is None else c
        i += 1
    return reps, delta


def moore(reps, delta, out_lab):
    """Coarsest partition with equal labels and transitions into equal blocks (missing
    transitions, at the length cap, count as their own target)."""
    block = {c: out_lab[c] for c in range(len(reps))}
    while True:
        key = {c: (block[c],) + tuple(block.get(delta.get((c, x)), -1) if (c, x) in delta else -2
                                      for x in range(3)) for c in range(len(reps))}
        ids = {k: i for i, k in enumerate(sorted(set(key.values())))}
        new = {c: ids[key[c]] for c in key}
        if len(set(new.values())) == len(set(block.values())):
            return new
        block = new


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "flipflop"
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
    for seed in seeds:
        t0 = time.time()
        _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
        es = ExactSig(m)
        sizes = [len(es.R[tau, a][0]) for tau in (5, 10, 23) for a in range(3)]
        reps, delta = build(es)
        err, cov = verify(es.teacher, reps, delta, kind)
        out_lab = {c: es.own_label(r) for c, r in enumerate(reps)}
        mb = moore(reps, delta, out_lab)
        capped = sum(1 for c, r in enumerate(reps) if len(r) >= T)
        print(f"{kind} seed {seed}: {len(reps)} classes (Moore over the machine: "
              f"{len(set(mb.values()))}; {capped} at the length cap); |R_tau| at tau=5/10/23: "
              f"{sizes[0:3]}/{sizes[3:6]}/{sizes[6:9]}; check on sampled strings: mismatch "
              f"{err:.4f}, coverage {cov:.3f}; {time.time() - t0:.0f} s", flush=True)
        print("  representatives:", ["".join(map(str, r[1:])) or "ε" for r in reps])
