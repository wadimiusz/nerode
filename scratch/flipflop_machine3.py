"""Hard-attention machine of a one-layer flipflop model, from weights only.

For every head h, query token a and shift tau (the future read at tau steps
after the current position; tau = 0 is the current read), the prefix is summed
up by (token of its best pair, rank of that best score among the scores the
continuation could bring for that read). This is exactly what decides every
future hard-attention read; transitions follow from the score table alone."""
import sys, itertools, numpy as np, torch
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.reads import scores, token_tables
from nerode.automata import flipflop_pdfa
BOS, T = 3, 24; MAXD = T - 1
E = flipflop_pdfa().emit; DELTA = flipflop_pdfa().delta

@torch.no_grad()
def output_table(model):
    blk = model.blocks[0]; _, _, v = token_tables(model)
    tab = np.zeros((4, 4, 4, 3))
    for a, w0, w1 in itertools.product(range(4), repeat=3):
        x = model.tok_emb.weight[a] + blk.attn.W_O(torch.tensor(np.concatenate([v[0, w0], v[1, w1]]), dtype=torch.float32))
        x = x + blk.mlp(blk.norm2(x))
        o = torch.softmax(model.lm_head(model.norm_f(x)).double(), 0).numpy()[:3]
        tab[a, w0, w1] = o / o.sum()
    return tab

def build(seed):
    _, m, _ = load_one_layer("flipflop", seed, 100_000, 128, 0.1)
    sc = scores(m, MAXD + 1)                        # (h, a, x, d)
    tab = output_table(m)
    # thresholds for the read at shift tau: scores of continuation tokens at distances 0..tau-1
    # rank only against continuation tokens different from the leader: a same-token
    # overtake does not change what the head reads
    thr = {(h, a, tau, y): np.sort(np.delete(sc[h, a, :3, :tau], y, axis=0).reshape(-1) if y < 3 else sc[h, a, :3, :tau].reshape(-1))
           for h in range(2) for a in range(3) for tau in range(MAXD + 1) for y in range(4)}
    def summ(best_score, tok, h, a, tau):
        return (tok, int(np.searchsorted(thr[(h, a, tau, tok)], best_score)))
    # exact best score is needed to rank; carry it along but key states on (current token, summaries)
    L = int(sys.argv[2])
    order = sorted((h, a, tau) for h in range(2) for a in range(3) for tau in range(MAXD + 1))
    keep = [i for i, (h, a, tau) in enumerate(order) if tau <= L]
    def key(st): return (st[0], tuple(st[2][i] for i in keep))
    def start():
        best = {(h, a, tau): (sc[h, a, BOS, tau], BOS) for h in range(2) for a in range(3) for tau in range(MAXD + 1)}
        return (BOS, best, tuple(summ(best[k][0], best[k][1], *k) for k in sorted(best)))
    def step(st, x):
        cur, best, _ = st
        nb = {}
        for h in range(2):
            for a in range(3):
                for tau in range(MAXD + 1):
                    old = best[(h, a, tau + 1)] if tau < MAXD else (-np.inf, None)
                    new = (sc[h, a, x, tau], x)
                    nb[(h, a, tau)] = old if old[0] > new[0] else new
        return (x, nb, tuple(summ(nb[k][0], nb[k][1], *k) for k in sorted(nb)))
    def out(st):
        cur, best, _ = st
        if cur == BOS:
            a = BOS
            w = [best[(h, 0, 0)][1] for h in range(2)]   # unused: no prediction is made from... see note
        a = cur
        w = [best[(h, min(a, 2), 0)][1] for h in range(2)]
        return tab[a, w[0], w[1]]
    # BFS by time; representative concrete state per key (keys capture all future behaviour)
    s0 = start(); reps = {key(s0): s0}; levels = [{key(s0)}]; trans = {}
    for t in range(T - 1):
        nxt = set()
        for k in levels[t]:
            for x in range(3):
                s2 = step(reps[k], x); k2 = key(s2)
                reps.setdefault(k2, s2); trans[(k, x)] = k2; nxt.add(k2)
        levels.append(nxt)
        print(f"   t={t+1}: {len(nxt)} states, total {len(reps)}", flush=True)
        if len(reps) > 60000:
            raise SystemExit("   too many states, stopping")
    return reps, levels, trans, out

def moore(reps, trans, out, tol):
    keys = list(reps)
    cls = {k: tuple(np.round(out(reps[k]) / tol).astype(int)) for k in keys}
    while True:
        sig = {k: (cls[k],) + tuple(cls.get(trans.get((k, x)), None) for x in range(3)) for k in keys}
        ids = {s: i for i, s in enumerate(sorted(set(sig.values()), key=repr))}
        new = {k: ids[sig[k]] for k in keys}
        if len(set(new.values())) == len(set(cls.values())): return new
        cls = new

def exact_gap(reps, levels, trans, out):
    dist = {(0, next(iter(levels[0]))): 1.0}; gaps = []
    for t in range(T):
        gaps.append(sum(p * float(np.sum(E[q] * np.log(E[q] / out(reps[k])))) for (q, k), p in dist.items()))
        if t == T - 1: break
        nd = {}
        for (q, k), p in dist.items():
            for x in range(3):
                kk = (DELTA[q, x], trans[(k, x)]); nd[kk] = nd.get(kk, 0.0) + p * E[q, x]
        dist = nd
    return float(np.mean(gaps))

for seed in [int(sys.argv[1])]:
    reps, levels, trans, out = build(seed)
    print(f"seed {seed}: machine states {len(reps)} (per t: {[len(L) for L in levels]})")
    for tol in (0.001, 0.01, 0.05):
        c = moore(reps, trans, out, tol)
        print(f"   Moore classes at output tolerance {tol}: {len(set(c.values()))}")
    print(f"   exact expected gap (no sampling): {exact_gap(reps, levels, trans, out):.5f}")
