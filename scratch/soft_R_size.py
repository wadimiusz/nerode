"""How big is the soft analogue of R? (weights only; exploratory.)

A head's value depends on the key token only, so a soft read is a mix of the 4 token values with
weights pi_x proportional to the token masses M_x = sum over occurrences of x of exp(score). A
continuation w of length tau ending in a enters a future read through its masses, 3 tokens x 2 heads
(BOS never occurs in a continuation); the prefix adds its own masses to them.

Soft R_tau(a) = the set of reachable continuation log-masses, built one distance at a time exactly
like the hard R: extend every element by each token y at the new (farthest) distance,
L[h, y] <- logaddexp(L[h, y], s_h(a, y, d)), then deduplicate after quantising log-masses to a grid
of width eta. A coordinate more than CLIP nats below the continuation's own largest mass in that head
is dropped (set to -inf): the total mass is at least that largest mass, so such a coordinate moves pi
by less than e^-CLIP whatever the prefix adds.
"""
import sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from exact_classes import ExactSig

CLIP = 15.0
CAP = 2_000_000


def soft_R_sizes(sc, a, eta, tau_max):
    L = np.full((1, 2, 3), -np.inf)
    for h in range(2):
        L[0, h, a] = sc[h, a, a, 0]
    sizes = [1]
    for tau in range(2, tau_max + 1):
        d = tau - 1
        new = np.repeat(L[:, None], 3, 1)                     # (n, y, h, x)
        for y in range(3):
            for h in range(2):
                new[:, y, h, y] = np.logaddexp(new[:, y, h, y], sc[h, a, y, d])
        new = new.reshape(-1, 2, 3)
        top = new.max(2, keepdims=True)
        new = np.where(new < top - CLIP, -np.inf, new)
        key = np.where(np.isfinite(new), np.round(new / eta), -10**9).astype(np.int64).reshape(len(new), -1)
        _, first = np.unique(key, axis=0, return_index=True)
        L = new[np.sort(first)]
        sizes.append(len(L))
        if len(L) * 3 > CAP:
            sizes += [None] * (tau_max - tau)
            break
    return sizes


if __name__ == "__main__":
    tau_max = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    models = [("flipflop", 0), ("flipflop", 2), ("flipflop_long", 1),
              ("parity_hand_reset", 0), ("parity_hand_reset", 1), ("parity_hand", 0)]
    for kind, seed in models:
        _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
        es = ExactSig(m)
        sc = es.sc
        hard = [max(len(es.R[t, a][0]) for a in range(3)) for t in range(1, tau_max + 1)]
        print(f"{kind} seed {seed}: score range {sc[:, :3, :3].min():.0f}..{sc[:, :3, :3].max():.0f}")
        print(f"  hard R (max over a):      {hard}")
        for eta in (0.3, 0.1):
            s = [soft_R_sizes(sc, a, eta, tau_max) for a in range(3)]
            mx = [None if any(x[t] is None for x in s) else max(x[t] for x in s) for t in range(tau_max)]
            print(f"  soft R, eta={eta:<4} (max over a): {mx}", flush=True)
        print(f"  all continuations:        {[3 ** (t - 1) for t in range(1, tau_max + 1)]}")
