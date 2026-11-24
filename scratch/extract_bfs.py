"""Weights -> finite automaton for a one-layer model, through the hard-attention abstraction.

Teacher: the output label of a string at each position, computed from weights only
(argmax of the score table per head, then the finite 4x4x4 output table). The network is
never run.

Classes are built on the fly (L*-style):
  1. Signature of a prefix p = teacher labels at the last position of p and at every
     position of p.e, for all e in {0,1,2}^tau.
  2. BFS from "BOS": extend each class representative by each symbol; a known signature
     gives a transition to that class, a new one gives a new class with this representative.
  3. Consistency: every prefix that was sent to class c must, after each symbol y, land in the
     same class as c's representative does. A violation means tau is too small: tau += 1.
  4. Verification (data allowed here): run the machine and the teacher on sampled strings and
     count positions where the labels differ.
"""
import sys, itertools, numpy as np, torch
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.reads import scores, token_tables
from nerode.automata import make_automaton, sample
BOS, T = 3, 24  # strings are BOS + up to 23 symbols (training length 24 incl. BOS)


@torch.no_grad()
def output_table(model):
    blk = model.blocks[0]; _, _, v = token_tables(model)
    tab = np.zeros((4, 4, 4, 4))
    for a, w0, w1 in itertools.product(range(4), repeat=3):
        heads = np.concatenate([v[0, w0], v[1, w1]])
        x = model.tok_emb.weight[a] + blk.attn.W_O(torch.tensor(heads, dtype=torch.float32))
        x = x + blk.mlp(blk.norm2(x))
        tab[a, w0, w1] = torch.softmax(model.lm_head(model.norm_f(x)).double(), 0).numpy()
    return tab


class Teacher:
    def __init__(self, model):
        self.sc = np.asarray(scores(model, T))
        self.tab = output_table(model)
        _, lab = np.unique(np.round(self.tab, 6).reshape(64, 4), axis=0, return_inverse=True)
        self.lab = lab.reshape(4, 4, 4)

    def labels(self, strings, start):
        """strings (N, L) incl. BOS -> labels at positions start..L-1."""
        N, L = strings.shape
        out = np.zeros((N, L - start), dtype=np.int64)
        for t in range(start, L):
            a = strings[:, t]; past = strings[:, : t + 1]; dist = t - np.arange(t + 1)
            w = [past[np.arange(N), self.sc[h][a[:, None], past, dist[None, :]].argmax(1)]
                 for h in range(2)]
            out[:, t - start] = self.lab[a, w[0], w[1]]
        return out


def signature(teacher, p, tau):
    suf = np.array(list(itertools.product(range(3), repeat=tau)), dtype=np.int64).reshape(3 ** tau, tau)
    s = np.concatenate([np.tile(np.array(p), (len(suf), 1)), suf], 1)
    return teacher.labels(s, len(p) - 1).tobytes()


def build(teacher, tau):
    max_len = T - tau                      # p.e must fit into the training length
    reps, sig2cls, delta, members = [], {}, {}, {}
    def classify(p):
        s = signature(teacher, p, tau)
        if s not in sig2cls:
            sig2cls[s] = len(reps); reps.append(p)
        members.setdefault(sig2cls[s], []).append(p)
        return sig2cls[s]
    classify((BOS,))
    i = 0
    while i < len(reps):
        r = reps[i]
        if len(r) < max_len:
            for x in range(3):
                delta[i, x] = classify(r + (x,))
        i += 1
    return reps, delta, members


def inconsistent(teacher, tau, reps, delta, members):
    """Pairs (q, class c, y) with sig(q.y) != class of rep(c).y, checked one step ahead."""
    bad = []
    for c, ps in members.items():
        for q in ps:
            if q == reps[c] or len(q) >= T - tau:
                continue
            for y in range(3):
                if (c, y) in delta and signature(teacher, q + (y,), tau) != signature(teacher, reps[delta[c, y]], tau):
                    bad.append((q, c, y))
    return bad


def verify(teacher, reps, delta, kind, n=4000, seed=11):
    syms, _ = sample(make_automaton(kind, 0), n, T - 1, np.random.default_rng(seed))
    strings = np.concatenate([np.full((n, 1), BOS), syms], 1)
    truth = teacher.labels(strings, 0)
    out_lab = {c: teacher.labels(np.array([r]), len(r) - 1)[0, 0] for c, r in enumerate(reps)}
    pred = np.full_like(truth, -1)
    for j in range(n):
        c = 0; pred[j, 0] = out_lab[c]
        for t in range(1, T):
            if (c, strings[j, t]) not in delta:
                break
            c = delta[c, strings[j, t]]; pred[j, t] = out_lab[c]
    known = pred >= 0
    return (pred[known] != truth[known]).mean(), known.mean()


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "flipflop"
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
    for seed in seeds:
        _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
        teacher = Teacher(m)
        for tau in range(0, 11):
            reps, delta, members = build(teacher, tau)
            bad = inconsistent(teacher, tau, reps, delta, members)
            err, cov = verify(teacher, reps, delta, kind)
            print(f"{kind} seed {seed} tau={tau}: {len(reps)} classes, {len(bad)} one-step "
                  f"inconsistencies, label mismatch on sampled strings {err:.4f}", flush=True)
            if not bad and err == 0:
                break
        print("  representatives:", ["".join(map(str, r[1:])) or "ε" for r in reps])
