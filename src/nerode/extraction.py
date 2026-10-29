"""Extracting a model's automaton from its behaviour.

Two prefixes are Myhill–Nerode equivalent for a model when every continuation
is equally likely after both. This module approximates that relation in two
ways, continuations are cut to length k and "equally likely" is relaxed to
"within eps in total variation", and builds the automaton it defines:

1. prefixes: all prefixes of sequences sampled from the automaton, each
   weighted by how often it occurs (so by its probability);
2. signature of a prefix u: the model's distribution over all |Sigma|^k
   continuations w, P(w | u);
3. states: leader clustering of the signatures at radius eps. The heaviest
   prefix not yet in a group starts a new group and takes every unassigned
   prefix within eps of it; at the end every prefix moves to its nearest
   leader. Frequent prefixes therefore define the states;
4. the extracted automaton: one state per group. Its emission is the mean
   next-symbol distribution of the group's prefixes; its transition on a is
   the group into which most of the extensions u·a of its prefixes fall;
5. comparison with the true minimal automaton: number of states, purity,
   how deterministic the transitions are, and, on held-out sequences,
   KL(model || extracted) against KL(model || true automaton).

Run on the automaton's own probabilities instead of a model's
(`signatures_from_pdfa`), the same pipeline must return the true minimal
automaton: that checks the pipeline itself (tests/test_extraction.py).

Command line: scripts/extract_automaton.py.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from centigpt.model import GPT

from nerode.automata import PDFA, minimise, sample
from nerode.readout import future_logprob
from nerode.training import DEVICE


@dataclass
class Prefixes:
    seqs: list[tuple[int, ...]]  # distinct prefixes, sorted by length
    weight: np.ndarray  # (n,) share of all sampled prefixes, sums to 1
    state: np.ndarray  # (n,) state of the true automaton after the prefix
    index: dict[tuple[int, ...], int]  # prefix -> row


def sample_prefixes(
    p: PDFA, n_seq: int, max_len: int, rng: np.random.Generator
) -> Prefixes:
    """Every prefix (lengths 0..max_len) of n_seq sampled sequences, counted
    once per sequence it occurs in."""
    syms, states = sample(p, n_seq, max_len, rng)
    count: dict[tuple[int, ...], int] = {}
    state_of: dict[tuple[int, ...], int] = {}
    for i in range(n_seq):
        row = syms[i].tolist()
        for t in range(max_len + 1):
            u = tuple(row[:t])
            count[u] = count.get(u, 0) + 1
            state_of[u] = int(states[i, t])
    seqs = sorted(count, key=lambda u: (len(u), u))
    w = np.array([count[u] for u in seqs], dtype=np.float64)
    return Prefixes(
        seqs=seqs,
        weight=w / w.sum(),
        state=np.array([state_of[u] for u in seqs]),
        index={u: i for i, u in enumerate(seqs)},
    )


def all_futures(n_sym: int, k: int) -> np.ndarray:
    """(n_sym**k, k): every continuation of length k, first symbol slowest, so
    that a signature reshaped to (n_sym, n_sym**(k-1)) has the next symbol on
    the first axis."""
    return np.array(list(itertools.product(range(n_sym), repeat=k)), dtype=np.int64)


def signatures_from_model(model: GPT, pre: Prefixes, n_sym: int, k: int) -> np.ndarray:
    """(n, n_sym**k): P_model(w | u) for every prefix u and continuation w.
    The model's distributions are renormalised over the real symbols (BOS is
    never a prediction), so every row sums to 1."""
    futures = all_futures(n_sym, k)
    bos = n_sym
    out = np.zeros((len(pre.seqs), len(futures)))
    lengths = np.array([len(u) for u in pre.seqs])
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        ctx = np.array([(bos, *pre.seqs[i]) for i in rows], dtype=np.int64)
        out[rows] = np.exp(future_logprob(model, ctx, futures, n_sym))
    return out


def state_signatures(p: PDFA, k: int) -> np.ndarray:
    """(n_states, n_sym**k): P(w | state) for every state of the automaton."""
    futures = all_futures(p.n_symbols, k)
    per_state = np.ones((p.n_states, len(futures)))
    for q0 in range(p.n_states):
        q = np.full(len(futures), q0)
        for t in range(k):
            a = futures[:, t]
            per_state[q0] *= p.emit[q, a]
            q = p.delta[q, a]
    return per_state


def signatures_from_pdfa(p: PDFA, pre: Prefixes, k: int) -> np.ndarray:
    """The same signatures computed from the true automaton: P(w | state)."""
    return state_signatures(p, k)[pre.state]


def model_error(sig_model: np.ndarray, sig_true: np.ndarray) -> np.ndarray:
    """(n,): total variation between the model's and the automaton's
    distributions over continuations, for every prefix."""
    return 0.5 * np.abs(sig_model - sig_true).sum(1)


def wrong_bit_weight(
    sig_model: np.ndarray, sig_right: np.ndarray, sig_wrong: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Parity automata: write the model's signature as a mixture of the true
    state's signature and that of the state with the same group and the other
    bit, sig_model ~ (1 - lam) * sig_right + lam * sig_wrong.

    Returns lam (least squares, clipped to [0, 1]): how much of its prediction
    the model puts on the wrong bit; and the residual, the total variation
    between the model and the best mixture: how much of the model's error the
    mixture does not explain."""
    d = sig_wrong - sig_right
    denom = (d * d).sum(1)
    lam = np.where(denom > 0, ((sig_model - sig_right) * d).sum(1) / denom, 0.0)
    lam = np.clip(lam, 0.0, 1.0)
    mix = sig_right + lam[:, None] * d
    return lam, model_error(sig_model, mix)


def _project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row of x onto the probability simplex
    (Duchi et al. 2008)."""
    u = -np.sort(-x, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, x.shape[1] + 1)
    rho = (u - css / idx > 0).sum(1)
    theta = css[np.arange(len(x)), rho - 1] / rho
    return np.maximum(x - theta[:, None], 0.0)


def belief_weights(
    sig_model: np.ndarray, per_state: np.ndarray, iters: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Write every model signature as a mixture of the automaton's states:
    sig_model ~ b @ per_state with b on the probability simplex, b chosen by
    least squares (projected gradient with Nesterov momentum). If the model's
    predictions are a belief over hidden states, as for an observer unsure
    which state it is in, the residual is small and b is that belief.

    Returns b (n, n_states) and the residual: total variation between the
    model and the best mixture."""
    n, s = len(sig_model), len(per_state)
    gram = per_state @ per_state.T
    step = 1.0 / np.linalg.eigvalsh(gram).max()
    target = sig_model @ per_state.T
    b = np.full((n, s), 1.0 / s)
    prev, y = b, b
    for t in range(iters):
        b = _project_to_simplex(y - step * (y @ gram - target))
        y = b + (t / (t + 3)) * (b - prev)
        prev = b
    return b, model_error(sig_model, b @ per_state)


def next_symbol(sig: np.ndarray, n_sym: int) -> np.ndarray:
    """(n, n_sym): the next-symbol distribution, a marginal of the signature."""
    return sig.reshape(len(sig), n_sym, -1).sum(-1)


def _tv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Total variation between every row of a and every row of b."""
    return 0.5 * torch.cdist(a, b, p=1)


def cluster(
    sig: np.ndarray, weight: np.ndarray, eps: float, mem_bytes: int = 64 << 20
) -> np.ndarray:
    """Leader clustering at radius eps in total variation; returns a label per
    prefix, labels ordered by when their leader was chosen (heaviest first).

    The heaviest unassigned prefix becomes a leader and takes every
    unassigned prefix within eps. Then every prefix moves to its nearest
    leader, which is never farther than the one it was assigned to, so every
    prefix ends within eps of its state's leader.

    The L1 `cdist` builds a (rows, leaders, dim) tensor, so the final step goes
    in chunks of rows that keep it under `mem_bytes` (at small eps there can be
    thousands of leaders, and a 4 GB GPU runs out of memory otherwise)."""
    x = torch.tensor(sig, dtype=torch.float64, device=DEVICE)
    unassigned = torch.ones(len(x), dtype=torch.bool, device=DEVICE)
    w = torch.tensor(weight, device=DEVICE)
    leaders: list[int] = []
    while bool(unassigned.any()):
        cand = torch.where(unassigned, w, torch.full_like(w, -1.0))
        lead = int(cand.argmax())
        leaders.append(lead)
        near = _tv(x[lead : lead + 1], x)[0] <= eps
        unassigned &= ~near
        unassigned[lead] = False
    lead_x = x[leaders]
    labels = torch.empty(len(x), dtype=torch.long, device=DEVICE)
    chunk = max(1, mem_bytes // (len(leaders) * x.shape[1] * x.element_size()))
    for i in range(0, len(x), chunk):
        labels[i : i + chunk] = _tv(x[i : i + chunk], lead_x).argmin(1)
    return labels.cpu().numpy()


@dataclass
class Extracted:
    emit: np.ndarray  # (n_states, n_sym)
    delta: np.ndarray  # (n_states, n_sym), -1 where no member had that extension
    start: int  # state of the empty prefix
    mass: np.ndarray  # (n_states,) share of sampled prefixes in each state
    determinism: float  # vote share of the chosen transitions, weighted


def build_automaton(
    pre: Prefixes, labels: np.ndarray, next_dist: np.ndarray, n_sym: int
) -> Extracted:
    n_states = int(labels.max()) + 1
    w = pre.weight
    mass = np.bincount(labels, weights=w, minlength=n_states)
    emit = np.zeros((n_states, n_sym))
    np.add.at(emit, labels, w[:, None] * next_dist)
    emit /= mass[:, None]
    # votes[(s, a)][s'] = weight of extensions u·a in state s', for u in state s
    votes: dict[tuple[int, int], dict[int, float]] = {}
    for i, u in enumerate(pre.seqs):
        for a in range(n_sym):
            j = pre.index.get((*u, a))
            if j is None:
                continue
            row = votes.setdefault((int(labels[i]), a), {})
            row[int(labels[j])] = row.get(int(labels[j]), 0.0) + float(w[j])
    delta = np.full((n_states, n_sym), -1, dtype=np.int64)
    agree = total = 0.0
    for (s, a), row in votes.items():
        best = max(row, key=lambda t: row[t])
        delta[s, a] = best
        agree += row[best]
        total += sum(row.values())
    return Extracted(
        emit=emit,
        delta=delta,
        start=int(labels[pre.index[()]]),
        mass=mass,
        determinism=agree / total if total else 1.0,
    )


def purity(pre: Prefixes, labels: np.ndarray, true_cls: np.ndarray) -> float:
    """Share of prefix mass whose true minimal class is the most common one in
    its extracted state: 1 when no state mixes true classes."""
    cls = true_cls[pre.state]
    n_states = int(labels.max()) + 1
    table = np.zeros((n_states, int(cls.max()) + 1))
    np.add.at(table, (labels, cls), pre.weight)
    return float(table.max(1).sum())


def majority_state(pre: Prefixes, labels: np.ndarray, n_states: int) -> np.ndarray:
    """For every extracted state, the true automaton state that holds most of
    its mass (not the minimal class: for parity automata the state 2g + b
    names the group and the bit, which is easier to read)."""
    table = np.zeros((n_states, int(pre.state.max()) + 1))
    np.add.at(table, (labels, pre.state), pre.weight)
    return table.argmax(1)


@torch.no_grad()
def heldout_kl(
    model: GPT, p: PDFA, ext: Extracted, n_seq: int, seq_len: int, rng
) -> dict[str, float]:
    """Run the extracted automaton along held-out sequences from p. At every
    position it can follow (no missing transition so far), compare the model's
    next-symbol distribution with the extracted state's emission and with the
    true state's emission. KL in nats per position."""
    model.eval()
    n_sym = p.n_symbols
    syms, states = sample(p, n_seq, seq_len, rng)
    x = torch.tensor(np.concatenate([np.full((n_seq, 1), n_sym), syms], 1))
    logits = model(x[:, :seq_len].to(DEVICE))[..., :n_sym].float()
    pm = torch.softmax(logits, -1).double().cpu().numpy()  # (n, seq_len, n_sym)
    pt = p.emit[states[:, :seq_len]]
    s = np.full(n_seq, ext.start)
    pe = np.zeros_like(pm)
    covered = np.zeros((n_seq, seq_len), dtype=bool)
    for t in range(seq_len):
        ok = s >= 0
        covered[:, t] = ok
        pe[ok, t] = ext.emit[s[ok]]
        nxt = np.full(n_seq, -1)
        nxt[ok] = ext.delta[s[ok], syms[ok, t]]
        s = nxt

    def kl(q: np.ndarray) -> np.ndarray:
        q = np.clip(q, 1e-12, None)
        return (pm * (np.log(np.clip(pm, 1e-12, None)) - np.log(q))).sum(-1)

    return {
        "coverage": float(covered.mean()),
        "kl_ext": float(kl(np.where(covered[..., None], pe, pm))[covered].mean()),
        "kl_true": float(kl(pt)[covered].mean()),
        "kl_true_all": float(kl(pt).mean()),
    }


def symbols_since_last(u: tuple[int, ...], symbol: int) -> int:
    """Number of symbols after the last occurrence of `symbol` in u (len(u) if
    it does not occur): for parity automata, how far back the bit was set."""
    for i in range(len(u) - 1, -1, -1):
        if u[i] == symbol:
            return len(u) - 1 - i
    return len(u)


def extract(
    sig: np.ndarray, pre: Prefixes, eps: float, n_sym: int
) -> tuple[np.ndarray, Extracted]:
    """Cluster and build: the labels and the extracted automaton at radius eps."""
    labels = cluster(sig, pre.weight, eps)
    return labels, build_automaton(pre, labels, next_symbol(sig, n_sym), n_sym)


def n_true_classes(p: PDFA) -> int:
    return int(minimise(p).max()) + 1
