"""Do prefixes in the same Myhill–Nerode class get similar internal states?

Sanity check for the geometric hypothesis behind the grant project:
"prefixes with the same future map to nearby points in the residual stream".

Setup
-----
1. Generate a random probabilistic deterministic finite automaton (PDFA):
   deterministic transitions delta(q, a), emission probabilities pi(a | q).
   For a PDFA the Myhill–Nerode classes of prefixes (prefixes with identical
   distributions over all continuations) are exactly the states of the
   *minimal* PDFA, so we minimise it and use minimal states as ground truth.
2. Train the nanogpt `GPT` as an ordinary language model on samples from it.
3. Read the residual stream after the embedding, after every block and after
   the final norm, at every position, and ask how well it separates classes.

Metrics (per layer, on held-out sequences)
------------------------------------------
- probe_acc:    linear probe (softmax regression) for the class.
- centroid_acc: nearest class centroid, no trained map: pure geometry.
- p_hid, c_hid: the same two, but only for telling apart classes that share
                their next-symbol distribution (chance 0.5 for pairs). Such
                classes differ only in the more distant future, so the model is
                not forced to separate them at the current position; it could
                recompute the difference later through attention.
                Only meaningful for the "shared-emission" automata.
- dist_ratio:   mean within-class distance / mean between-class distance,
                for pairs at the same position AND with the same last token
                (removes position and current-token confounds).
                < 1 means classes are clustered; 1 means no structure.
- behav_rho:    Spearman correlation between distances of class centroids and
                behavioural distances between classes (total variation between
                their next-two-symbol distributions). The "continuous" version
                of the hypothesis: more different futures, further apart.

Baselines
---------
- the same architecture with random weights (is the structure learned?);
- predicting the class from (position, last token) alone (is it trivial?);
- the optimal loss: entropy rate of the PDFA.

Run from the repo root:  uv run python experiments/nerode_geometry.py
(matplotlib is optional: uv run --with matplotlib python experiments/...)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from nanogpt.config import Config
from nanogpt.model import GPT

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- PDFA


@dataclass
class PDFA:
    delta: np.ndarray  # (n_states, n_symbols) -> next state
    emit: np.ndarray  # (n_states, n_symbols) probabilities, rows sum to 1

    @property
    def n_states(self) -> int:
        return self.delta.shape[0]

    @property
    def n_symbols(self) -> int:
        return self.delta.shape[1]


def random_pdfa(
    n_states: int, n_symbols: int, rng: np.random.Generator, shared: bool = False
) -> PDFA:
    """If `shared`, states come in pairs (2k, 2k+1) with the SAME emission
    distribution and different transitions: they predict the next symbol
    identically and differ only in the more distant future."""
    delta = rng.integers(0, n_states, size=(n_states, n_symbols))
    emit = rng.dirichlet(np.full(n_symbols, 0.7), size=n_states)
    emit = 0.9 * emit + 0.1 / n_symbols  # every symbol possible in every state
    if shared:
        emit[1::2] = emit[0::2][: n_states // 2]
    return PDFA(delta, emit / emit.sum(1, keepdims=True))


def minimise(p: PDFA, start: int = 0, tol: float = 1e-9) -> np.ndarray:
    """Map each reachable state to its minimal-PDFA class (Moore refinement).

    Unreachable states get -1. Two states are equivalent iff they emit the same
    distribution and their successors are equivalent for every symbol, i.e.
    iff they induce the same distribution over all futures.
    """
    reach, frontier = {start}, [start]
    while frontier:
        q = frontier.pop()
        for a in range(p.n_symbols):
            r = int(p.delta[q, a])
            if r not in reach:
                reach.add(r)
                frontier.append(r)
    states = sorted(reach)

    def key0(q: int) -> Hashable:
        return tuple(np.round(p.emit[q] / tol).astype(np.int64).tolist())

    labels: dict[int, Hashable] = {q: key0(q) for q in states}
    while True:
        sig: dict[int, Hashable] = {
            q: (
                labels[q],
                tuple(labels[int(p.delta[q, a])] for a in range(p.n_symbols)),
            )
            for q in states
        }
        sig_ids = {v: i for i, v in enumerate(sorted(set(sig.values()), key=str))}
        new: dict[int, Hashable] = {q: sig_ids[sig[q]] for q in states}
        if len(set(new.values())) == len(set(labels.values())):
            break
        labels = new
    final_ids = {v: i for i, v in enumerate(sorted(set(labels.values()), key=str))}
    out = np.full(p.n_states, -1)
    for q in states:
        out[q] = final_ids[labels[q]]
    return out


def sample(p: PDFA, n_seq: int, length: int, rng: np.random.Generator):
    """Return symbols (n_seq, length) and states (n_seq, length + 1).

    states[:, t] is the state after reading the first t symbols, i.e. the state
    that generates symbol t. Position t of the model input (BOS + symbols) has
    seen t symbols after BOS... see `to_model_inputs`.
    """
    syms = np.zeros((n_seq, length), dtype=np.int64)
    states = np.zeros((n_seq, length + 1), dtype=np.int64)
    cum = p.emit.cumsum(1)
    for t in range(length):
        q = states[:, t]
        u = rng.random(n_seq)[:, None]
        a = (u > cum[q]).sum(1)
        syms[:, t] = a
        states[:, t + 1] = p.delta[q, a]
    return syms, states


def entropy_rate(p: PDFA, states: np.ndarray) -> float:
    h = -(p.emit * np.log(p.emit)).sum(1)
    return float(h[states[:, :-1]].mean())


def next_two_dist(p: PDFA, q: int) -> np.ndarray:
    d = np.zeros((p.n_symbols, p.n_symbols))
    for a in range(p.n_symbols):
        d[a] = p.emit[q, a] * p.emit[int(p.delta[q, a])]
    return d.ravel()


# ---------------------------------------------------------------- model


def make_model(n_symbols: int, seq_len: int, d_model: int, n_layers: int, seed: int):
    cfg = Config(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=2,
        vocab_size=n_symbols + 1,  # last id is BOS
        max_seq_len=seq_len + 1,
        seed=seed,
    )
    torch.manual_seed(seed)
    return GPT(cfg).to(DEVICE)


Monitor = Callable[[GPT], dict[str, str]]


def _train_loop(
    model: GPT,
    next_batch: Callable[[], torch.Tensor],
    steps: int,
    lr: float,
    desc: str | None = None,
    h_opt: float | None = None,
    monitor: Monitor | None = None,
    monitor_every: int = 5000,
) -> float:
    """The training loop shared by `train` and `train_fresh`, with a progress
    bar: smoothed training loss, its gap to the optimal loss `h_opt`, learning
    rate, and whatever `monitor(model)` reports every `monitor_every` steps.
    The bar and the monitor only read the model, so results do not change."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    bar = tqdm(
        range(steps),
        desc=desc or "training",
        unit="step",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    info: dict[str, str] = {}
    ema, n_ema = 0.0, 0  # bias-corrected moving average over ~10 readings
    loss = torch.tensor(0.0)
    for step in bar:
        xb = next_batch()
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            n_ema += 1
            ema = 0.9 * ema + 0.1 * float(loss.detach())
            smooth = ema / (1 - 0.9**n_ema)
            info["loss"] = f"{smooth:.4f}"
            if h_opt is not None:
                info["gap"] = f"{smooth - h_opt:+.4f}"
            info["lr"] = f"{sched.get_last_lr()[0]:.1e}"
            bar.set_postfix(info, refresh=False)
        if monitor is not None and (step + 1) % monitor_every == 0:
            info.update(monitor(model))
            model.train()
            bar.set_postfix(info, refresh=False)
    bar.close()
    return float(loss.detach())


def train(
    model: GPT,
    x: torch.Tensor,
    steps: int,
    lr: float,
    batch: int = 128,
    **progress,
) -> float:
    """Train on the finite dataset x (the pilot setup)."""
    gen = torch.Generator().manual_seed(0)

    def next_batch() -> torch.Tensor:
        idx = torch.randint(0, x.shape[0], (batch,), generator=gen)
        return x[idx].to(DEVICE)

    return _train_loop(model, next_batch, steps, lr, **progress)


def train_fresh(
    model: GPT,
    p: PDFA,
    steps: int,
    lr: float,
    seq_len: int,
    seed: int,
    batch: int = 128,
    **progress,
) -> float:
    """Like `train`, but every batch is a fresh sample from the automaton, so
    the model cannot overfit a finite dataset and can be trained much longer."""
    bos = p.n_symbols
    rng = np.random.default_rng(10_000 + seed)

    def next_batch() -> torch.Tensor:
        syms, _ = sample(p, batch, seq_len, rng)
        xb = np.concatenate([np.full((batch, 1), bos), syms], 1)
        return torch.tensor(xb).to(DEVICE)

    return _train_loop(model, next_batch, steps, lr, **progress)


CKPT_DIR = Path(__file__).with_name("nerode_out") / "ckpt"


def fit(
    model: GPT,
    p: PDFA,
    x_tr: torch.Tensor,
    *,
    steps: int,
    seed: int,
    seq_len: int,
    lr: float = 3e-3,
    batch: int = 128,
    fresh: bool = False,
    cache: bool = True,
    desc: str | None = None,
    h_opt: float | None = None,
    monitor: Monitor | None = None,
) -> GPT:
    """Train `model` in place, or load it from the checkpoint cache.

    fresh=False is the original setup (`train` on the finite dataset x_tr).
    fresh=True trains on new samples every batch (`train_fresh`).
    The cache key covers the automaton itself and every training setting, so a
    changed automaton or setting never picks up a stale checkpoint."""
    cfg = model.cfg
    h = hashlib.sha1()
    for part in (
        p.delta.tobytes(),
        np.round(p.emit, 12).tobytes(),
        repr((steps, seed, seq_len, lr, batch, fresh)).encode(),
        repr((cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.vocab_size)).encode(),
    ):
        h.update(part)
    if not fresh:
        h.update(x_tr.numpy().tobytes())
    path = CKPT_DIR / f"{h.hexdigest()[:16]}.pt"
    if cache and path.exists():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        tqdm.write(f"  {desc or 'model'}: loaded from cache ({path.name})")
        return model
    progress = dict(desc=desc, h_opt=h_opt, monitor=monitor)
    t0 = time.time()
    if fresh:
        loss = train_fresh(model, p, steps, lr, seq_len, seed, batch, **progress)
    else:
        loss = train(model, x_tr, steps=steps, lr=lr, batch=batch, **progress)
    msg = f"  {desc or 'model'}: trained {steps} steps in {time.time() - t0:.0f}s"
    msg += f", last batch loss {loss:.4f}"
    if h_opt is not None:
        msg += f" (optimal {h_opt:.4f})"
    tqdm.write(msg)
    if cache:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def eval_loss(model: GPT, x: torch.Tensor) -> float:
    model.eval()
    x = x.to(DEVICE)
    logits = model(x[:, :-1])
    return float(
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    )


@torch.no_grad()
def residuals(model: GPT, x: torch.Tensor) -> list[tuple[str, np.ndarray]]:
    """Residual stream at every position: after embedding, each block, final norm."""
    model.eval()
    h = model.tok_emb(x.to(DEVICE))
    out = [("embed", h)]
    for i, block in enumerate(model.blocks):
        h = block(h)
        out.append((f"block{i}", h))
    out.append(("norm_f", model.norm_f(h)))
    return [(name, t.float().cpu().numpy()) for name, t in out]


# ---------------------------------------------------------------- metrics


def softmax_probe(xtr, ytr, xte, yte, n_cls: int, epochs: int = 300) -> np.ndarray:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr_t = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
    xte_t = torch.tensor((xte - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    lin = torch.nn.Linear(xtr.shape[1], n_cls)
    opt = torch.optim.LBFGS(
        lin.parameters(), max_iter=epochs, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(xtr_t), ytr_t) + 1e-3 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return lin(xte_t).argmax(1).numpy()


def centroid_pred(xtr, ytr, xte, n_cls: int) -> np.ndarray:
    cents = np.stack([xtr[ytr == c].mean(0) for c in range(n_cls)])
    d = ((xte[:, None, :] - cents[None]) ** 2).sum(-1)
    return d.argmin(1)


def same_position_ratio(
    h: np.ndarray, cls: np.ndarray, last: np.ndarray, rng, n_pairs: int = 20000
) -> float:
    """h: (n_seq, T, d); cls, last: (n_seq, T). Within/between-class distance for
    pairs at the same position that also share the last token, so that neither
    position nor the current token can explain the clustering."""
    n_seq, T, _ = h.shape
    within, between = [], []
    for t in range(2, T):
        i = rng.integers(0, n_seq, n_pairs // T + 1)
        j = rng.integers(0, n_seq, n_pairs // T + 1)
        keep = (last[i, t] == last[j, t]) & (i != j)
        i, j = i[keep], j[keep]
        d = np.linalg.norm(h[i, t] - h[j, t], axis=1)
        same = cls[i, t] == cls[j, t]
        within.extend(d[same])
        between.extend(d[~same])
    if not within or not between:
        return float("nan")
    return float(np.mean(within) / np.mean(between))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def behavioural_rho(h: np.ndarray, cls: np.ndarray, behav: np.ndarray) -> float:
    n_cls = behav.shape[0]
    cents = np.stack([h[cls == c].mean(0) for c in range(n_cls)])
    iu = np.triu_indices(n_cls, 1)
    dc = np.linalg.norm(cents[:, None] - cents[None], axis=-1)[iu]
    return spearman(dc, behav[iu]) if len(iu[0]) > 2 else float("nan")


def trivial_pred(syms_tr, cls_tr, syms_te, bos: int) -> np.ndarray:
    """Best accuracy predicting the class from (position, last token) alone."""

    def feats(syms):
        last = np.concatenate([np.full((syms.shape[0], 1), bos), syms], 1)[:, :-1]
        pos = np.broadcast_to(np.arange(last.shape[1]), last.shape)
        return pos * 1000 + last

    ftr, fte = feats(syms_tr), feats(syms_te)
    table: dict[int, int] = {}
    for f in np.unique(ftr):
        vals = cls_tr[ftr == f]
        table[int(f)] = int(np.bincount(vals).argmax())
    return np.array([table.get(int(f), 0) for f in fte.ravel()]).reshape(fte.shape)


def within_group_acc(xtr, ytr, xte, yte, groups: list[list[int]], kind: str) -> float:
    """Accuracy of telling apart classes that share a next-symbol distribution,
    given that the group is known (chance = 1 / group size, i.e. 0.5 for pairs).
    kind: "centroid" (pure geometry) or "probe" (linear, per group)."""
    correct, total = 0, 0
    for g in groups:
        tr, te = np.isin(ytr, g), np.isin(yte, g)
        if te.sum() == 0 or len(np.unique(ytr[tr])) < 2:
            continue
        remap = {c: i for i, c in enumerate(g)}
        ytr_g = np.array([remap[c] for c in ytr[tr]])
        yte_g = np.array([remap[c] for c in yte[te]])
        if kind == "centroid":
            pred = centroid_pred(xtr[tr], ytr_g, xte[te], len(g))
        else:
            pred = softmax_probe(xtr[tr], ytr_g, xte[te], yte_g, len(g))
        correct += int((pred == yte_g).sum())
        total += int(te.sum())
    return correct / total if total else float("nan")


def within_group_trivial(ftr, ytr, fte, yte, groups: list[list[int]]) -> float:
    """Same, but predicting from (position, last token) features."""
    correct, total = 0, 0
    for g in groups:
        tr, te = np.isin(ytr, g), np.isin(yte, g)
        table = {}
        for f in np.unique(ftr[tr]):
            table[int(f)] = int(np.bincount(ytr[tr][ftr[tr] == f]).argmax())
        pred = np.array([table.get(int(f), g[0]) for f in fte[te]])
        correct += int((pred == yte[te]).sum())
        total += int(te.sum())
    return correct / total if total else float("nan")


# ---------------------------------------------------------------- experiment


def run_one(
    n_states: int,
    n_symbols: int,
    seed: int,
    shared: bool = False,
    seq_len: int = 24,
    d_model: int = 64,
    n_layers: int = 2,
    steps: int = 1500,
    plot_dir=None,
    fresh: bool = False,
    batch: int = 128,
):
    rng = np.random.default_rng(seed)
    p = random_pdfa(n_states, n_symbols, rng, shared=shared)
    cls_of_state = minimise(p)
    n_cls = int(cls_of_state.max()) + 1
    rep = {c: int(np.where(cls_of_state == c)[0][0]) for c in range(n_cls)}

    # classes that share their next-symbol distribution with another class:
    # telling them apart requires information about the more distant future
    emit_key = {c: tuple(np.round(p.emit[rep[c]], 9)) for c in range(n_cls)}
    group_size = np.array(
        [sum(emit_key[c] == emit_key[d] for d in range(n_cls)) for c in range(n_cls)]
    )
    groups = [
        [d for d in range(n_cls) if emit_key[d] == k]
        for k in sorted(set(emit_key.values()))
    ]
    groups = [g for g in groups if len(g) > 1]

    syms, states = sample(p, 6000, seq_len, rng)
    bos = n_symbols
    x = torch.tensor(np.concatenate([np.full((len(syms), 1), bos), syms], 1))
    # model position t has read BOS + t symbols -> state states[:, t]
    cls = cls_of_state[states[:, :seq_len]]
    last = x[:, :seq_len].numpy()  # token at model position t
    n_tr = 5000
    x_tr, x_te = x[:n_tr], x[n_tr:]
    cls_tr, cls_te = cls[:n_tr], cls[n_tr:]

    behav = np.zeros((n_cls, n_cls))
    for c1 in range(n_cls):
        for c2 in range(n_cls):
            behav[c1, c2] = (
                0.5
                * np.abs(next_two_dist(p, rep[c1]) - next_two_dist(p, rep[c2])).sum()
            )

    m = slice(1, None)  # skip position 0 (BOS, always the start class)
    yte2 = cls_te[:, m].ravel()

    def accs(pred):
        return float((pred == yte2).mean())

    triv = trivial_pred(syms[:n_tr, :seq_len], cls_tr, syms[n_tr:, :seq_len], bos)
    triv_acc = accs(triv[:, m].ravel())
    feat = (np.arange(seq_len)[None, :] * 1000 + last)[:, m]
    triv_hidden = within_group_trivial(
        feat[:n_tr].ravel(), cls_tr[:, m].ravel(), feat[n_tr:].ravel(), yte2, groups
    )

    h_opt = entropy_rate(p, states[n_tr:])
    rows = []
    t0 = time.time()
    for kind in ("random", "trained"):
        model = make_model(n_symbols, seq_len, d_model, n_layers, seed)
        if kind == "trained":
            tag = "shared" if shared else "plain"
            fit(
                model,
                p,
                x_tr,
                steps=steps,
                seed=seed,
                seq_len=seq_len,
                batch=batch,
                fresh=fresh,
                desc=f"{tag} s{n_states} seed {seed}",
                h_opt=h_opt,
            )
        loss = eval_loss(model, x_te)
        res_tr = dict(residuals(model, x_tr[:, :seq_len]))
        res_te = dict(residuals(model, x_te[:, :seq_len]))
        for name, htr in res_tr.items():
            hte = res_te[name]
            xtr2 = htr[:, m].reshape(-1, htr.shape[-1])
            xte2 = hte[:, m].reshape(-1, hte.shape[-1])
            ytr2 = cls_tr[:, m].ravel()
            probe = accs(softmax_probe(xtr2, ytr2, xte2, yte2, n_cls))
            cent = accs(centroid_pred(xtr2, ytr2, xte2, n_cls))
            probe_h = within_group_acc(xtr2, ytr2, xte2, yte2, groups, "probe")
            cent_h = within_group_acc(xtr2, ytr2, xte2, yte2, groups, "centroid")
            rows.append(
                dict(
                    model=kind,
                    layer=name,
                    loss=loss,
                    probe_acc=probe,
                    probe_hidden=probe_h,
                    centroid_acc=cent,
                    centroid_hidden=cent_h,
                    dist_ratio=same_position_ratio(
                        hte, cls_te, last[n_tr:], np.random.default_rng(0)
                    ),
                    behav_rho=behavioural_rho(xte2, yte2, behav),
                )
            )
        if plot_dir is not None and kind == "trained":
            tag = "shared" if shared else "plain"
            plot_pca(
                res_te,
                cls_te,
                n_cls,
                plot_dir / f"pca_{tag}_s{n_states}_a{n_symbols}_seed{seed}.png",
                title=f"{tag} PDFA, {n_cls} Nerode classes, {n_symbols} symbols",
            )
    return dict(
        n_states=n_states,
        n_cls=n_cls,
        n_symbols=n_symbols,
        seed=seed,
        shared=shared,
        n_hidden_cls=int((group_size > 1).sum()),
        h_opt=h_opt,
        trivial=triv_acc,
        trivial_hidden=triv_hidden,
        rows=rows,
        secs=time.time() - t0,
    )


def plot_pca(res: dict, cls: np.ndarray, n_cls: int, path: Path, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(res)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    y = cls[:, 1:].ravel()
    for ax, name in zip(axes, names):
        h = res[name][:, 1:].reshape(-1, res[name].shape[-1])[:3000]
        yy = y[:3000]
        hc = h - h.mean(0)
        _, _, vt = np.linalg.svd(hc, full_matrices=False)
        z = hc @ vt[:2].T
        ax.scatter(z[:, 0], z[:, 1], c=yy, cmap="tab10", s=3, vmin=0, vmax=9)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title + " — colour = Nerode class", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--fresh", action="store_true", help="train on fresh samples every batch"
    )
    a = ap.parse_args()
    out_dir = Path(os.environ.get("NERODE_OUT", "nerode_out"))
    out_dir.mkdir(exist_ok=True)
    # (n_states, n_symbols, seed, shared emissions)
    settings = [
        (4, 3, 0, False),
        (6, 3, 1, False),
        (8, 4, 2, False),
        (6, 3, 3, True),
        (8, 3, 4, True),
    ]
    lines = []

    def emit(line: str):
        tqdm.write(line)
        lines.append(line)

    for n_states, n_symbols, seed, shared in tqdm(
        settings, desc="automata", unit="automaton", dynamic_ncols=True
    ):
        r = run_one(
            n_states,
            n_symbols,
            seed,
            shared=shared,
            plot_dir=out_dir,
            steps=a.steps,
            fresh=a.fresh,
            batch=a.batch,
        )
        emit(
            f"\n{'SHARED-emission' if shared else 'plain'} PDFA: {r['n_states']} states "
            f"-> {r['n_cls']} Nerode classes ({r['n_hidden_cls']} share their next-symbol "
            f"distribution), {r['n_symbols']} symbols, seed {r['seed']} ({r['secs']:.0f}s)"
        )
        emit(f"  training: {a.steps} steps, batch {a.batch}, fresh data {a.fresh}")
        emit(
            f"  optimal loss {r['h_opt']:.3f} nats | class from (position, last token): "
            f"acc {r['trivial']:.3f}, within same-next-symbol groups "
            f"{r['trivial_hidden']:.3f} (chance 0.5)"
        )
        emit(
            f"  {'model':8s} {'layer':7s} {'loss':>6s} {'probe':>6s} {'p_hid':>6s} "
            f"{'centr':>6s} {'c_hid':>6s} {'ratio':>6s} {'rho':>6s}"
        )
        for row in r["rows"]:
            emit(
                f"  {row['model']:8s} {row['layer']:7s} {row['loss']:6.3f} "
                f"{row['probe_acc']:6.3f} {row['probe_hidden']:6.3f} "
                f"{row['centroid_acc']:6.3f} {row['centroid_hidden']:6.3f} "
                f"{row['dist_ratio']:6.3f} {row['behav_rho']:6.3f}"
            )
    (out_dir / "results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
