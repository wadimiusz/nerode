"""extract_bfs with tau fixed in advance by the erasure horizon (weights only): tau = L - 1.
Sampled strings are used only for the final check."""
import sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reads_spectrum import load_one_layer
from nerode.reads import scores
from extract_bfs import Teacher, build, verify, T
from erasure_horizon import erasure_horizon

kind = sys.argv[1] if len(sys.argv) > 1 else "flipflop"
seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
for seed in seeds:
    _, m, _ = load_one_layer(kind, seed, 100_000, 128, 0.1)
    per = erasure_horizon(np.asarray(scores(m, T)))
    if None in per.values():
        print(f"{kind} seed {seed}: no erasure horizon within the training length; criterion silent")
        continue
    tau = max(per.values()) - 1
    teacher = Teacher(m)
    reps, delta, _ = build(teacher, tau)
    err, cov = verify(teacher, reps, delta, kind)
    print(f"{kind} seed {seed}: L = {tau + 1}, tau = {tau}: {len(reps)} classes; "
          f"check on sampled strings: mismatch {err:.4f}", flush=True)
