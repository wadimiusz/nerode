"""Do prefixes in the same Myhill–Nerode class get similar internal states?

Version 1 of the experiment: read the residual stream after the embedding,
after every block and after the final norm, at every position, and ask how well
it separates the Nerode classes of the prefix read so far.

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

Command line: scripts/nerode_geometry.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nerode.automata import entropy_rate, minimise, next_two_dist, random_pdfa, sample
from nerode.readout import residuals
from nerode.stats import spearman
from nerode.training import eval_loss, fit, make_model

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
    """PCA of every layer coloured by Nerode class. Needs matplotlib
    (the `plots` extra); silently does nothing without it."""
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
