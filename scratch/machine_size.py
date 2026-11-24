import sys, time, numpy as np
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.reads import scores
BOS, T = 3, 24; MAXD = T - 1
seed = int(sys.argv[1])
_, m, _ = load_one_layer("flipflop", seed, 100_000, 128, 0.1)
sc = scores(m, MAXD)
P = [(x, d) for x in range(4) for d in range(T)]
idx = {p: i for i, p in enumerate(P)}
dom = np.zeros((2, len(P), len(P)), bool)       # dom[h, i, j]: in head h pair i beats pair j forever
for i, (x1, d1) in enumerate(P):
    for j, (x0, d0) in enumerate(P):
        if i == j: continue
        k = MAXD - max(d0, d1) + 1
        dom[:, i, j] = (sc[:, :3, x1, d1:d1 + k] > sc[:, :3, x0, d0:d0 + k]).all(axis=(1, 2))
def prune(pairs):
    ii = [idx[p] for p in pairs]
    keep = [p for p, j in zip(pairs, ii) if not dom[:, ii, j].any(axis=1).all()]
    return tuple(sorted(keep))
level = {((BOS, 0),)}
t0 = time.time()
for t in range(T - 1):
    nxt = set()
    for s in level:
        for x in range(3):
            nxt.add(prune(tuple((y, d + 1) for y, d in s) + ((x, 0),)))
    level = nxt
    sizes = [len(s) for s in level]
    print(f"t={t+1}: {len(level)} states, pairs per state max {max(sizes)}  ({time.time()-t0:.1f}s)", flush=True)
    if len(level) > 200000: print("too many, stop"); break
