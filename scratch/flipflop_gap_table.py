"""Loss gap of the one-layer flipflop models: exact (network run) vs predicted
from weights (hard-attention abstraction), with 95% intervals.

The abstraction uses only the score table, the value table and the layers after
attention (one vector per combination of winners); strings come from the
automaton itself, the network is never run on them."""
import sys, itertools, numpy as np, torch
sys.path.insert(0, "scripts")
from reads_spectrum import load_one_layer
from nerode.reads import scores, token_tables
from nerode.automata import flipflop_pdfa, sample
from nerode.probes import SEQ_LEN
BOS = 3; E = flipflop_pdfa().emit

@torch.no_grad()
def output_table(model):
    """P(next token, all 4) for current token a and winner tokens (w0, w1)."""
    blk = model.blocks[0]; _, _, v = token_tables(model)
    tab = np.zeros((4, 4, 4, 4))
    for a, w0, w1 in itertools.product(range(4), repeat=3):
        heads = np.concatenate([v[0, w0], v[1, w1]])
        x = model.tok_emb.weight[a] + blk.attn.W_O(torch.tensor(heads, dtype=torch.float32))
        x = x + blk.mlp(blk.norm2(x))
        tab[a, w0, w1] = torch.softmax(model.lm_head(model.norm_f(x)).double(), 0).numpy()
    return tab

def per_seq_kl(true, logp):
    """Mean over positions of KL(true || model) for each sequence."""
    return np.sum(true * (np.log(true) - logp[..., :3]), -1).mean(1)

def ci(per_seq):
    m, se = per_seq.mean(), per_seq.std(ddof=1) / np.sqrt(len(per_seq))
    return m, m - 1.96 * se, m + 1.96 * se

rng = np.random.default_rng(2026)
N = 20000
syms, states = sample(flipflop_pdfa(), N, SEQ_LEN, rng)
inp = np.concatenate([np.full((N, 1), BOS), syms], 1)[:, :SEQ_LEN]   # position t sees inp[:, :t+1]
true = E[states[:, :SEQ_LEN]]
for seed in (0, 1, 2):
    _, m, _ = load_one_layer("flipflop", seed, 100_000, 128, 0.1)
    # exact: run the network
    with torch.no_grad():
        logp = torch.log_softmax(m(torch.tensor(inp)).double(), -1).numpy()
    exact = ci(per_seq_kl(true, logp))
    # predicted from weights: hard attention (each head reads the token with the top score)
    sc = scores(m, SEQ_LEN); tab = output_table(m)
    win = np.zeros((N, SEQ_LEN, 2), dtype=int)
    for t in range(SEQ_LEN):
        a = inp[:, t]                                       # current token
        past = inp[:, : t + 1]                              # tokens at positions 0..t
        dist = t - np.arange(t + 1)
        for h in range(2):
            s = sc[h][a[:, None], past, dist[None, :]]      # (N, t+1)
            win[:, t, h] = past[np.arange(N), s.argmax(1)]
    probs = tab[inp, win[..., 0], win[..., 1]]               # (N, T, 4)
    pred = ci(per_seq_kl(true, np.log(probs)))
    agree = np.mean(np.abs(np.exp(logp) - probs).max(-1) < 0.01)
    print(f"seed {seed}: exact {exact[0]:.4f} [{exact[1]:.4f}, {exact[2]:.4f}]   "
          f"from weights {pred[0]:.4f} [{pred[1]:.4f}, {pred[2]:.4f}]   "
          f"positions where abstraction and network agree within 0.01: {agree:.4f}")
