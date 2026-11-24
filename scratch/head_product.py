"""Is the model's machine a quotient of the product of per-head machines?

Per-head teacher: label at a position = (current token, token that head h reads there), from
the score table only. Each head's machine is extracted by the same BFS, with tau from that
head's own erasure horizon. Then the reachable part of the product (head-0 class, head-1 class)
is walked from the start, and for each reachable pair we check that every prefix leading to it
falls into one class of the model's machine (built as in extract_weights_only)."""
import sys, itertools, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.reads import scores
from extract_bfs import Teacher, T, BOS, signature
from erasure_horizon import erasure_horizon


class HeadTeacher(Teacher):
    def __init__(self, model, head):
        super().__init__(model); self.h = head

    def labels(self, strings, start):
        N, L = strings.shape
        out = np.zeros((N, L - start), dtype=np.int64)
        for t in range(start, L):
            a = strings[:, t]; past = strings[:, : t + 1]; dist = t - np.arange(t + 1)
            w = past[np.arange(N), self.sc[self.h][a[:, None], past, dist[None, :]].argmax(1)]
            out[:, t - start] = 4 * a + w
        return out


def bfs(teacher, tau):
    """Classes, transitions, and a classifier for arbitrary prefixes."""
    reps, sig2cls, delta = [], {}, {}
    def cls(p):
        s = signature(teacher, p, tau)
        if s not in sig2cls:
            sig2cls[s] = len(reps); reps.append(p)
        return sig2cls[s]
    cls((BOS,)); i = 0
    while i < len(reps):
        if len(reps[i]) < T - tau:
            for x in range(3):
                delta[i, x] = cls(reps[i] + (x,))
        i += 1
    return reps, delta, cls


kind = sys.argv[1] if len(sys.argv) > 1 else "flipflop"
for seed in (0, 1, 2):
    _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
    per = erasure_horizon(np.asarray(scores(m, T)))
    L_head = [max(per[h, a] for a in range(3)) for h in range(2)]
    heads = [bfs(HeadTeacher(m, h), L_head[h] - 1) for h in range(2)]
    model_reps, _, model_cls = bfs(Teacher(m), max(L_head) - 1)
    # walk the reachable product, one representative prefix per pair
    start = (0, 0); rep = {start: (BOS,)}; queue = [start]; bad = 0
    pair2model = {start: model_cls((BOS,))}
    while queue:
        pq = queue.pop(0); p = rep[pq]
        if len(p) >= T - max(L_head) + 1:
            continue
        for x in range(3):
            if any((pq[h], x) not in heads[h][1] for h in range(2)):
                continue
            nxt = (heads[0][1][pq[0], x], heads[1][1][pq[1], x])
            c = model_cls(p + (x,))
            if nxt not in rep:
                rep[nxt] = p + (x,); pair2model[nxt] = c; queue.append(nxt)
            elif pair2model[nxt] != c:
                bad += 1           # same pair of head states, different model class
    sizes = [len(hd[0]) for hd in heads]
    print(f"{kind} seed {seed}: head machines {sizes[0]} and {sizes[1]} states "
          f"(tau {L_head[0]-1}, {L_head[1]-1}); product {sizes[0]*sizes[1]}, reachable pairs {len(rep)}; "
          f"model machine {len(model_reps)} classes; pairs mapped to two model classes: {bad}", flush=True)
