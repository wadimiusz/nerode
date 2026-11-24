"""How far is the hard-attention abstraction from the network? (debug / validation: the network
IS run here, on purpose.)

Abstraction: every head reads the value of the single highest-score token; everything after
attention (W_O, MLP, norms, lm_head) is applied exactly, through the 4x4x4 output table.
Network: the real forward pass (soft attention).

(a), (b): per-position TV between the two output distributions, on strings sampled from the
automaton (in distribution) and on uniformly random strings (many rare combinations). Also the
score gap (best key minus the best key holding a different token, the smaller over the two
heads) where they agree / disagree.
(c): for random pairs of exact classes (exact_classes.build), the continuation (witness) on which
the abstraction's outputs differ most; does the network also tell the two strings apart?
"""
import sys, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.automata import make_automaton, sample
from exact_classes import ExactSig, build, T, BOS

N = 4000


@torch.no_grad()
def net_probs(model, strings):
    x = torch.tensor(strings, dtype=torch.long)
    return torch.softmax(model(x).double(), -1).numpy()


def abstraction(es, strings):
    """(probs (N, L, 4), gap (N, L)) of the hard abstraction."""
    n, L = strings.shape
    probs = np.zeros((n, L, 4)); gap = np.full((n, L), np.inf)
    rows = np.arange(n)
    for t in range(L):
        a = strings[:, t]; past = strings[:, : t + 1]; dist = t - np.arange(t + 1)
        w = []
        for h in range(2):
            s = es.sc[h][a[:, None], past, dist[None, :]]
            win = past[rows, s.argmax(1)]
            w.append(win)
            # gap to the best key holding a DIFFERENT token (another copy of the winner token
            # carries the same value, so it cannot change the read)
            other = np.where(past != win[:, None], s, -np.inf).max(1)
            gap[:, t] = np.minimum(gap[:, t], s.max(1) - other)
        probs[:, t] = es.teacher.tab[a, w[0], w[1]]
    return probs, gap


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def compare(es, model, strings, label):
    pn = net_probs(model, strings)
    pa, gap = abstraction(es, strings)
    d = tv(pn, pa)
    ok = d <= 0.01
    g_ok = np.median(gap[ok & np.isfinite(gap)]) if (ok & np.isfinite(gap)).any() else np.nan
    g_bad = np.median(gap[~ok & np.isfinite(gap)]) if (~ok & np.isfinite(gap)).any() else np.nan
    bins = [0, 2, 5, 10, 20, np.inf]
    by_gap = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (gap >= lo) & (gap < hi)
        by_gap.append(f"[{lo},{hi}): {sel.mean():.2f} of pos, agree {ok[sel].mean():.2f}" if sel.any()
                      else f"[{lo},{hi}): -")
    print(f"  {label}: agree (TV<=0.01) {ok.mean():.3f}, median TV {np.median(d):.4f}, "
          f"max TV {d.max():.3f}; median gap agree {g_ok:.1f} / disagree {g_bad:.1f}")
    print("     by gap: " + "; ".join(by_gap))


def winners_on_R(es, p, tau, a):
    p = np.array(p); n = len(p); dist = (n - 1 - np.arange(n)) + tau
    vals, toks, wit = es.R[tau, a]
    win = []
    for h in range(2):
        s = es.sc[h, a, p, dist]; i = int(np.argmax(s))
        win.append(np.where(s[i] >= vals[:, h], p[i], toks[:, h]))
    return es.teacher.tab[a, win[0], win[1]], wit


def witness_pairs(es, model, reps, n_pairs=300, seed=0):
    rng = np.random.default_rng(seed)
    tvA, tvN, agree = [], [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(reps), 2, replace=False)
        p, q = reps[i], reps[j]
        best = (0.0, None)
        for tau in range(1, T - max(len(p), len(q)) + 1):
            for a in range(3):
                op, wit = winners_on_R(es, p, tau, a)
                oq, _ = winners_on_R(es, q, tau, a)
                d = tv(op, oq)
                k = int(np.argmax(d))
                if d[k] > best[0]:
                    best = (d[k], wit[k])
        if best[1] is None or best[0] < 0.01:
            continue
        w = tuple(best[1])
        sp, sq = np.array([p + w]), np.array([q + w])
        np_, nq = net_probs(model, sp)[0, -1], net_probs(model, sq)[0, -1]
        ap, aq = abstraction(es, sp)[0][0, -1], abstraction(es, sq)[0][0, -1]
        tvA.append(best[0]); tvN.append(tv(np_, nq))
        agree.append(max(tv(np_, ap), tv(nq, aq)) <= 0.01)
    tvA, tvN, agree = map(np.array, (tvA, tvN, agree))
    if len(tvA) == 0:
        print("  witness pairs: no pair differs by more than 0.01 in the abstraction")
        return
    print(f"  witness pairs ({len(tvA)} with abstraction TV >= 0.01): network also differs "
          f"(TV >= 0.01) in {(tvN >= 0.01).mean():.2f}; median TV abstraction {np.median(tvA):.3f} "
          f"vs network {np.median(tvN):.3f}; network = abstraction on both strings in "
          f"{agree.mean():.2f}")


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["flipflop", "flipflop_long", "parity_hand_reset", "parity_hand"]
    for kind in kinds:
        for seed in (0, 1, 2):
            _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
            es = ExactSig(m)
            print(f"{kind} seed {seed}:", flush=True)
            rng = np.random.default_rng(11)
            syms, _ = sample(make_automaton(kind, seed, 0.1), N, T - 1, rng)
            compare(es, m, np.concatenate([np.full((N, 1), BOS), syms], 1), "automaton strings")
            uni = rng.integers(0, 3, (N, T - 1))
            compare(es, m, np.concatenate([np.full((N, 1), BOS), uni], 1), "uniform strings  ")
            if kind.startswith("flipflop"):
                reps, _ = build(es)
                witness_pairs(es, m, reps)
            sys.stdout.flush()
