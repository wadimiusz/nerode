"""Weights-only stopping criterion for extract_bfs: the erasure horizon.

Claim. Suppose that for every head h, every query token a and every tau >= L:
    m_h(a, tau) := max( s_h(a, a, 0),  max_{d=1..tau-1} min_y s_h(a, y, d) )
                 >  max_{y in {0,1,2,BOS}, d >= tau} s_h(a, y, d).
Then after any continuation w of length tau >= L, every head's winner lies inside w, so the
output at that position depends on w only, not on the prefix. Hence two prefixes that agree on
all continuations of length <= L-1 agree on all continuations: signatures with tau = L-1 give
exact classes.

Why m is the worst case: at time tau the query is the last token of w (= a), and the
continuation token at distance d contributes s(a, w_d, d). The continuation that keeps the
best continuation score as low as possible picks, at each distance, the token with the lowest
score (the choices are independent); the current token itself is always there (distance 0).
Every string is allowed (flipflop emits all symbols in every state).

Second way for the prefix not to matter: (B) BOS beats every other token at every distance, so
the head's winner is BOS for every string; BOS is the same token for every prefix, and the
output depends only on the winner's identity, not on its distance."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.reads import scores

T = 24
def erasure_horizon(sc, queries=(0, 1, 2)):
    H, _, _, D = sc.shape
    per = {}
    for h in range(H):
        for a in queries:
            ok = []
            for tau in range(1, D):
                cont = max([sc[h, a, a, 0]] + [sc[h, a, :3, d].min() for d in range(1, tau)])
                pre = sc[h, a, :, tau:].max()          # tokens 0,1,2 and BOS at distance >= tau
                # (A) the continuation always beats the whole prefix, or
                # (B) BOS (at any distance >= tau) beats every other token at any distance,
                #     so the winner is always BOS whatever the prefix and continuation.
                bos_wins = sc[h, a, 3, tau:].min() > sc[h, a, :3, :].max()
                ok.append(cont > pre or bos_wins)
            # smallest L such that ok holds for every tau >= L (within the training length)
            L = next((t for t in range(1, D) if all(ok[t - 1:])), None)
            per[h, a] = L
    return per

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "flipflop"
    for seed in (0, 1, 2):
        _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
        per = erasure_horizon(np.asarray(scores(m, T)))
        Ls = [v for v in per.values()]
        L = None if None in Ls else max(Ls)
        print(f"{kind} seed {seed}: erasure horizon L = {L}  (per head/query: "
              + ", ".join(f"h{h}a{a}:{v}" for (h, a), v in per.items()) + ")")
