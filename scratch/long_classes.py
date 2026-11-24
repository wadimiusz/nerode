"""Exact classes of the hard abstraction when strings may be longer than the training context
(exploratory): maximal length L > 24 uses RoPE scores at distances the model never trained on.
Same construction as exact_classes.py with T replaced by L; the own label is computed from the
extended score table. Prints the class count per L and the longest representative.
"""
import sys, time, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.reads import scores
import exact_classes as E

MAX_CLASSES = 20000


class LongSig(E.ExactSig):
    def __init__(self, model, L):
        E.T = L
        self.teacher = E.Teacher(model)
        self.lab = self.teacher.lab
        self.horizon = L - 1
        self.sc = np.asarray(scores(model, L))
        self.R = E.continuation_sets(self.sc, L - 1)

    def own_label(self, p):
        p = np.array(p); a = p[-1]; dist = len(p) - 1 - np.arange(len(p))
        w = [p[int(np.argmax(self.sc[h, a, p, dist]))] for h in range(2)]
        return int(self.lab[a, w[0], w[1]])


def build_capped(es, L):
    reps, delta, index = [], {}, {}
    def add(p, parts):
        c = len(reps); reps.append(p)
        for k in range(len(parts)):
            index.setdefault((k, b"".join(parts[: k + 1])), c)
        return c
    add((E.BOS,), es.signature((E.BOS,)))
    i = 0
    while i < len(reps):
        r = reps[i]
        if len(r) < L:
            for x in range(3):
                p = r + (x,); parts = es.signature(p)
                c = index.get((len(parts) - 1, b"".join(parts)))
                delta[i, x] = add(p, parts) if c is None else c
                if len(reps) > MAX_CLASSES:
                    return reps, False
        i += 1
    return reps, True


if __name__ == "__main__":
    kind, seeds = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    Ls = [int(x) for x in sys.argv[3].split(",")]
    for seed in seeds:
        _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
        for L in Ls:
            t0 = time.time()
            es = LongSig(m, L)
            reps, done = build_capped(es, L)
            lens = [len(r) for r in reps]
            print(f"{kind} seed {seed}, L {L}: {'' if done else '> '}{len(reps)} classes, longest representative "
                  f"{max(lens)}; {time.time() - t0:.0f} s", flush=True)
