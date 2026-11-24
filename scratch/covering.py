"""Covering numbers of the model's prefix states under the soft distance (exploratory, weights only
except for where the prefixes come from).

Prefixes: n distinct strings of length LEN (BOS + LEN-1 symbols) sampled from the automaton.
Distance: d(p, q) = max TV of outputs over the own output and over continuations of length
tau <= TAU_MAX, a in {0, 1, 2}, w in the soft R (soft_classes.Soft, grid eta). Greedy eps-net in
random order: a prefix farther than eps from every centre becomes a centre; the count N(eps) is a
packing number of the sample (between the covering numbers at eps and eps/2).
Slope between two eps = log(N1 / N2) / log(eps2 / eps1): a finite automaton gives a plateau (slope
0), a set of dimension D gives slope about D.
beta scales all scores; large beta approaches hard attention.
"""
import sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.automata import make_automaton, sample
from soft_classes import Soft, BOS

LEN, TAU_MAX, N = 16, 8, 1000
EPSS = [0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]


def signatures(soft, prefixes):
    rows = []
    for p in prefixes:
        rows.append(np.concatenate([soft.part(p, k).ravel() for k in range(TAU_MAX + 1)]))
    return np.array(rows, dtype=np.float32)


def greedy_net(X, eps, rng):
    order = rng.permutation(len(X))
    C = np.empty((0, X.shape[1]), dtype=np.float32)
    for i in order:
        x = X[i]
        if len(C):
            d = (0.5 * np.abs(C - x).reshape(len(C), -1, 4).sum(-1)).max(-1)
            if d.min() <= eps:
                continue
        C = np.vstack([C, x])
    return len(C)


def run(kind, seed, beta, eta=1.0):
    _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
    soft = Soft(m, eta, beta)
    rng = np.random.default_rng(5)
    syms, _ = sample(make_automaton(kind, seed, 0.1), 4 * N, LEN - 1, rng)
    prefixes = list(dict.fromkeys(tuple([BOS] + list(s)) for s in syms))[:N]
    X = signatures(soft, prefixes)
    counts = [greedy_net(X, e, np.random.default_rng(0)) for e in EPSS]
    slopes = [np.log(counts[i + 1] / counts[i]) / np.log(EPSS[i] / EPSS[i + 1]) for i in range(len(EPSS) - 1)]
    print(f"{kind} seed {seed}, beta {beta}: {len(prefixes)} prefixes; N(eps) for eps {EPSS}: {counts}; "
          f"slopes {[round(s, 1) for s in slopes]}", flush=True)


if __name__ == "__main__":
    for kind, seed, beta in [("flipflop", 0, 1.0), ("parity_hand_reset", 1, 1.0), ("parity_hand_reset", 1, 3.0),
                             ("parity_hand_reset", 1, 10.0), ("parity_hand_reset", 0, 1.0),
                             ("parity_hand_reset", 2, 1.0), ("parity_hand", 0, 1.0)]:
        run(kind, seed, beta)
