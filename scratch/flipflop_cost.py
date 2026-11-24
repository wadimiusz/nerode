import sys, numpy as np, torch
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.reads import token_tables
from nerode.automata import flipflop_pdfa, sample
from nerode.probes import SEQ_LEN
BOS = 3
E = flipflop_pdfa().emit
def kl(p, q): return float(np.sum(p * np.log(p / q)))

@torch.no_grad()
def predict(model, a, winners):
    blk = model.blocks[0]; _, _, v = token_tables(model)
    heads = np.concatenate([v[h, winners[h]] for h in range(v.shape[0])])
    x = model.tok_emb.weight[a] + blk.attn.W_O(torch.tensor(heads, dtype=torch.float32))
    x = x + blk.mlp(blk.norm2(x))
    return torch.softmax(model.lm_head(model.norm_f(x))[:3], 0).numpy()

# weights-only description per seed: memory head, window D*, other head's winner
desc = {0: dict(mem=1, other=BOS, Dstar=6), 1: dict(mem=0, other=BOS, Dstar=0), 2: dict(mem=0, other=BOS, Dstar=8)}
# for seed 1 the memory head (head 0) reads the current token itself; with current
# token 0 it reads a zero -> state lost at every 0 (D* = 0)
for seed, d in desc.items():
    p, m, _ = load_one_layer("flipflop", seed, 100_000, 128, 0.1)
    q_lost = predict(m, 0, [0 if d["mem"] == 0 else d["other"], 0 if d["mem"] == 1 else d["other"]])
    # positions in the training distribution where the last reset is more than D* back
    # (current token 0 and the last D*+1 tokens all 0): exact stationary probability 0.6^(D*+1);
    # the state there is s0 / s1 with prob 1/2 each
    frac = 0.6 ** (d["Dstar"] + 1)
    cost = frac * 0.5 * (kl(E[0], q_lost) + kl(E[1], q_lost))
    print(f"seed {seed}: weights say state is lost after {d['Dstar']} zeros; "
          f"lost-state prediction {np.round(q_lost,3)}; predicted loss gap ≈ {cost:.4f}")
# debugging check on data: model's KL to the true next-symbol distribution by zeros since last reset
from nerode.training import DEVICE
rng = np.random.default_rng(123)
syms, states = sample(flipflop_pdfa(), 4000, SEQ_LEN, rng)
x = torch.tensor(np.concatenate([np.full((len(syms), 1), BOS), syms], 1))
for seed in (0, 1, 2):
    _, m, _ = load_one_layer("flipflop", seed, 100_000, 128, 0.1)
    with torch.no_grad():
        pr = torch.softmax(m(x[:, :SEQ_LEN])[..., :3].float(), -1).numpy()  # pr[:, t] predicts symbol t
    # state before emitting symbol t is states[:, t]; zeros since last reset before position t
    out = {}
    for i in range(len(syms)):
        run = 0; seen = False
        for t in range(SEQ_LEN):
            k = run if seen else -1
            out.setdefault(min(k, 12), []).append(kl(E[states[i, t]], pr[i, t]))
            s = syms[i, t]
            if s == 0: run += 1
            else: run, seen = 0, True
    keys = sorted(out)
    print(f"seed {seed} data check, mean KL(true || model) by zeros since last reset (-1 = no reset yet, 12 = 12+):")
    print("   " + "  ".join(f"{k}:{np.mean(out[k]):.3f}" for k in keys))
