"""Probabilistic deterministic finite automata (PDFA) and their Nerode classes.

A PDFA has deterministic transitions delta(q, a) and emission probabilities
pi(a | q). For a PDFA the Myhill–Nerode classes of prefixes (prefixes with
identical distributions over all continuations) are exactly the states of the
*minimal* PDFA, so `minimise` gives the ground truth for every experiment.

Three constructions:

- `random_pdfa`:        every state has its own next-symbol distribution; with
                        `shared=True` states come in pairs that emit identically
                        and differ only in the more distant future;
- `parity_pdfa`:        states are (group g, parity bit b). Symbol 0 flips b,
                        1 keeps it, 2 reads b (the next group depends on it) and
                        resets it. The bit changes nothing about the next symbol;
- `hand_parity_pdfa`:   one parity automaton chosen by hand for the demo.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np


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


def sample(
    p: PDFA, n_seq: int, length: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return symbols (n_seq, length) and states (n_seq, length + 1).

    states[:, t] is the state after reading the first t symbols, i.e. the state
    that generates symbol t. Position t of the model input (BOS + symbols) has
    read t symbols after BOS, so it corresponds to states[:, t].
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
    """The optimal loss: mean entropy of the emission at the visited states."""
    h = -(p.emit * np.log(p.emit)).sum(1)
    return float(h[states[:, :-1]].mean())


def next_two_dist(p: PDFA, q: int) -> np.ndarray:
    """Distribution over the next two symbols from state q, flattened."""
    d = np.zeros((p.n_symbols, p.n_symbols))
    for a in range(p.n_symbols):
        d[a] = p.emit[q, a] * p.emit[int(p.delta[q, a])]
    return d.ravel()


# ---------------------------------------------------------------- parity automata


def group_separation(emit_g: np.ndarray) -> float:
    """Smallest KL divergence (either direction) between the emission
    distributions of two groups. If it is near 0, two groups predict the next
    symbol almost identically, so the hidden bit (which only decides the group
    after a 2) barely matters for prediction."""
    n = len(emit_g)
    return min(
        float(np.sum(emit_g[i] * np.log(emit_g[i] / emit_g[j])))
        for i in range(n)
        for j in range(n)
        if i != j
    )


def parity_pdfa(n_groups: int, rng: np.random.Generator, min_sep: float = 0.0) -> PDFA:
    """States 2g + b. Alphabet {0, 1, 2}: 0 flips b, 1 keeps it, 2 reads and
    resets it. Emissions depend on the group only.

    min_sep > 0 redraws the emissions until every two groups differ by at least
    that much (see `group_separation`), so the hidden bit always matters.
    min_sep = 0 reproduces the original automata exactly."""
    n_sym = 3
    group_next = rng.integers(0, n_groups, size=(n_groups, n_sym))
    # where symbol 2 leads when b = 1: always a different group than for b = 0
    alt = (group_next[:, 2] + rng.integers(1, n_groups, size=n_groups)) % n_groups
    for _ in range(10_000):
        emit_g = rng.dirichlet(np.full(n_sym, 0.7), size=n_groups)
        emit_g = 0.85 * emit_g + 0.15 / n_sym
        if group_separation(emit_g) >= min_sep:
            break
    else:
        raise ValueError(f"no emissions with group separation >= {min_sep}")
    return _parity_from_groups(group_next, alt, emit_g)


def hand_parity_pdfa() -> PDFA:
    """A parity automaton chosen by hand for the demo: three groups with
    clearly different emissions (every symbol likely in every group, so long
    stretches without a 2 still occur), a cyclic group structure, and the bit
    sending a 2 to a different group than it would go to otherwise."""
    emit_g = np.array(
        [
            [0.55, 0.25, 0.20],  # group 0 likes 0
            [0.20, 0.55, 0.25],  # group 1 likes 1
            [0.25, 0.20, 0.55],  # group 2 likes 2
        ]
    )
    group_next = np.array(
        [
            [1, 2, 1],  # from group 0: after 0 -> 1, after 1 -> 2, after 2 (b=0) -> 1
            [2, 0, 2],
            [0, 1, 0],
        ]
    )
    alt = np.array([2, 0, 1])  # after 2 with b = 1
    return _parity_from_groups(group_next, alt, emit_g)


def _parity_from_groups(
    group_next: np.ndarray, alt: np.ndarray, emit_g: np.ndarray
) -> PDFA:
    n_groups, n_sym = emit_g.shape
    n_states = 2 * n_groups
    delta = np.zeros((n_states, n_sym), dtype=np.int64)
    emit = np.zeros((n_states, n_sym))
    for g in range(n_groups):
        for b in range(2):
            q = 2 * g + b
            emit[q] = emit_g[g]
            delta[q, 0] = 2 * group_next[g, 0] + (1 - b)
            delta[q, 1] = 2 * group_next[g, 1] + b
            delta[q, 2] = 2 * (group_next[g, 2] if b == 0 else alt[g])
    return PDFA(delta, emit / emit.sum(1, keepdims=True))
