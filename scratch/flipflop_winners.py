import sys, numpy as np, torch
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.reads import scores, token_tables
from nerode.automata import flipflop_pdfa
BOS, MAXD, M = 3, 24, 10.0
NAME = {0: "0", 1: "1", 2: "2", 3: "BOS"}

@torch.no_grad()
def predict(model, a, winners):
    blk = model.blocks[0]
    _, _, v = token_tables(model)
    heads = np.concatenate([v[h, winners[h]] for h in range(v.shape[0])])
    x = model.tok_emb.weight[a] + blk.attn.W_O(torch.tensor(heads, dtype=torch.float32))
    x = x + blk.mlp(blk.norm2(x))
    return torch.softmax(model.lm_head(model.norm_f(x))[:3], 0).numpy()

def winner_query0(s, D, r):
    """Head score table s[x, d] for query 0; last reset r at distance D >= 1,
    zeros at 0..D-1 (the query itself is the zero at 0), anything older.
    Returns the margin by which the reset beats every possible competitor."""
    between = s[0, :D].max()
    older = s[:, D + 1:].max() if D < MAXD else -np.inf
    return s[r, D] - max(between, older)

for seed in (0, 1, 2):
    p, m, _ = load_one_layer("flipflop", seed, 100_000, 128, 0.1)
    sc = scores(m, MAXD)
    print(f"=== flipflop seed {seed} ===")
    for h in range(2):
        marg = [[winner_query0(sc[h, 0], D, r) for D in range(1, MAXD + 1)] for r in (1, 2)]
        marg = np.min(marg, axis=0)
        good = [D for D, mg in zip(range(1, MAXD + 1), marg) if mg >= M]
        upto = 0
        while upto + 1 <= MAXD and marg[upto] >= M: upto += 1
        print(f"  head {h}: query 0 — last reset wins by >= {M:.0f} for D = 1..{upto}"
              f"   (margins D=1..10: {np.round(marg[:10]).astype(int).tolist()})")
        # queries 1, 2: does the query's own token (d = 0) win outright?
        for a in (1, 2):
            s = sc[h, a]
            mg = s[a, 0] - np.delete(s.reshape(-1), a * (MAXD + 1)).max()
            print(f"          query {a} — itself (d=0) beats everything by {mg:.0f}")
        # no reset yet: BOS vs zeros, query 0
        s = sc[h, 0]
        print(f"          query 0, no reset yet — BOS minus best zero (any d): {s[BOS].min() - s[0].max():.0f}")
print()
print("truth: s0 -> [0.6 0.28 0.12], s1 -> [0.6 0.12 0.28]")
