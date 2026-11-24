import sys, numpy as np, torch
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.automata import flipflop_pdfa, sample
from nerode.probes import SEQ_LEN
BOS = 3; E = flipflop_pdfa().emit
def kl(p, q): return float(np.sum(p * np.log(p / q)))
rng = np.random.default_rng(123)
syms, states = sample(flipflop_pdfa(), 20000, SEQ_LEN, rng)
x = torch.tensor(np.concatenate([np.full((len(syms), 1), BOS), syms], 1))
for seed in (0, 1, 2):
    _, m, _ = load_one_layer("flipflop", seed, 100_000, 128, 0.1)
    with torch.no_grad():
        logits = m(x[:, :SEQ_LEN]).double()
    pr = torch.softmax(logits[..., :3], -1).numpy()
    lp_all = torch.log_softmax(logits, -1).numpy()           # over all 4 tokens, as in training
    true = E[states[:, :SEQ_LEN]]                             # (n, T, 3)
    exact_gap = np.mean(np.sum(true * (np.log(true) - lp_all[..., :3]), -1))  # E[KL] incl. mass on BOS
    # the same, measured the noisy way: sample loss minus entropy
    nll = -np.take_along_axis(lp_all, syms[:, :SEQ_LEN, None], -1)[..., 0].mean()
    ent = np.mean(-np.sum(true * np.log(true), -1))
    print(f"seed {seed}: exact gap (mean KL over positions, 20000 seqs) {exact_gap:.4f};"
          f"  noisy estimate loss-entropy {nll - ent:+.4f}")
# noise of the noisy estimate with 4000 sequences, as in train_parity
lp = -np.log(E)
var = np.mean([np.sum(E[s] * (lp[s] - np.sum(E[s]*lp[s]))**2) for s in (0, 1)])
print(f"std of loss-entropy estimate on 4000 x {SEQ_LEN} positions: {np.sqrt(var / (4000*SEQ_LEN)):.4f}")
# corrected weights-only prediction for seed 1: exact fractions from the automaton
# position t (0-based) predicts symbol t; current token is symbol t-1 (BOS at t=0)
# lost <=> current token 0 and some reset seen before it; "no reset yet" with current 0 -> state is s0 for sure
q = np.array([0.6, 0.214, 0.186])   # seed 1 output when head 0 reads a zero (from weights)
f_lost = f_none = 0.0
for t in range(1, SEQ_LEN):
    p_cur0 = 0.6
    p_noreset_and_cur0 = 0.6 ** t   # all of symbols 0..t-1 are 0 (from start state s0: P(0)=0.6)
    f_none += p_noreset_and_cur0
    f_lost += p_cur0 - p_noreset_and_cur0
f_lost /= SEQ_LEN; f_none /= SEQ_LEN
pred = f_lost * 0.5 * (kl(E[0], q) + kl(E[1], q)) + f_none * kl(E[0], q)
print(f"seed 1 corrected prediction: lost on {f_lost:.3f} of positions, 'zeros only so far' on {f_none:.3f}; gap ≈ {pred:.4f}")
