"""Diagnostics for exact_classes: (1) how far apart the output-table entries are; (2) classes when
the output labels are first merged at a tolerance on the finite 4x4x4 table (single-linkage on
total variation: the tolerance acts on 64 output vectors, not on states); (3) how much
probability mass of the true automaton's strings each machine's classes carry."""
import sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.automata import make_automaton, sample
from exact_classes import ExactSig, build, T


def merged_labels(tab, eps):
    """Single-linkage clusters of the distinct output vectors at TV <= eps."""
    flat = tab.reshape(64, 4)
    tv = 0.5 * np.abs(flat[:, None] - flat[None]).sum(-1)
    lab = np.arange(64)
    changed = True
    while changed:
        changed = False
        for i in range(64):
            for j in range(64):
                if tv[i, j] <= eps and lab[i] != lab[j]:
                    lo = min(lab[i], lab[j]); lab[lab == max(lab[i], lab[j])] = lo; changed = True
    _, lab = np.unique(lab, return_inverse=True)
    return lab.reshape(4, 4, 4)


def mass(reps, delta, kind, n=4000, seed=11):
    syms, _ = sample(make_automaton(kind, 0), n, T - 1, np.random.default_rng(seed))
    visits = np.zeros(len(reps))
    for j in range(n):
        c = 0; visits[c] += 1
        for x in syms[j]:
            c = delta[c, x]; visits[c] += 1
    visits /= visits.sum()
    top = np.sort(visits)[::-1]
    return (visits > 0).sum(), [int((np.cumsum(top) < q).sum() + 1) for q in (0.9, 0.99, 0.999)]


kind = sys.argv[1] if len(sys.argv) > 1 else "flipflop_long"
seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
for seed in seeds:
    _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
    es = ExactSig(m)
    tab = es.teacher.tab
    exact_lab = es.lab.copy()
    print(f"{kind} seed {seed}: {exact_lab.max() + 1} distinct outputs in the 4x4x4 table")
    for eps in (0.0, 0.001, 0.003, 0.01, 0.03):
        es.lab = exact_lab if eps == 0 else merged_labels(tab, eps)
        reps, delta = build(es)
        seen, need = mass(reps, delta, kind)
        print(f"  eps {eps:<6} labels {es.lab.max() + 1:>2}: {len(reps):>5} classes; visited by 4000 "
              f"sampled strings {seen}; classes covering 90/99/99.9% of visits {need}", flush=True)
