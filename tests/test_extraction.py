"""The extraction pipeline, checked without any model: run on the automaton's
own probabilities it must return the true minimal automaton."""

from __future__ import annotations

import numpy as np
import pytest

from nerode.automata import hand_parity_pdfa, minimise, parity_pdfa
from nerode.extraction import (
    all_futures,
    belief_weights,
    extract,
    next_symbol,
    purity,
    sample_prefixes,
    signatures_from_model,
    signatures_from_pdfa,
    state_signatures,
    symbols_since_last,
    wrong_bit_weight,
)
from nerode.training import make_model


def test_oracle_recovers_the_hand_parity_automaton() -> None:
    p = hand_parity_pdfa()
    pre = sample_prefixes(p, 300, 12, np.random.default_rng(0))
    sig = signatures_from_pdfa(p, pre, k=3)
    labels, ext = extract(sig, pre, eps=0.01, n_sym=3)
    true_cls = minimise(p)
    assert len(ext.mass) == int(true_cls.max()) + 1 == 6
    assert purity(pre, labels, true_cls) == pytest.approx(1.0)
    assert ext.determinism == pytest.approx(1.0)
    # every extracted state's emission is its true state's emission
    for s in range(len(ext.mass)):
        q = pre.state[labels == s][0]
        assert np.allclose(ext.emit[s], p.emit[q])
    # and its transitions follow the true ones
    for s in range(len(ext.mass)):
        q = pre.state[labels == s][0]
        for a in range(3):
            if ext.delta[s, a] >= 0:
                r = pre.state[labels == ext.delta[s, a]][0]
                assert true_cls[r] == true_cls[p.delta[q, a]]


def test_oracle_on_random_parity_automata() -> None:
    for seed in range(3):
        p = parity_pdfa(3, np.random.default_rng(seed), min_sep=0.1)
        pre = sample_prefixes(p, 300, 12, np.random.default_rng(seed))
        sig = signatures_from_pdfa(p, pre, k=3)
        labels, ext = extract(sig, pre, eps=0.005, n_sym=3)
        assert len(ext.mass) == int(minimise(p).max()) + 1
        assert purity(pre, labels, minimise(p)) == pytest.approx(1.0)


def test_signatures_are_distributions() -> None:
    p = hand_parity_pdfa()
    pre = sample_prefixes(p, 20, 6, np.random.default_rng(1))
    model = make_model(3, 24, 16, 1, seed=0)  # untrained: only shapes and sums
    sig = signatures_from_model(model, pre, n_sym=3, k=2)
    assert sig.shape == (len(pre.seqs), 9)
    assert np.allclose(sig.sum(1), 1.0, atol=1e-5)
    assert np.allclose(next_symbol(sig, 3).sum(1), 1.0, atol=1e-5)
    assert all_futures(3, 2)[1].tolist() == [0, 1]  # first symbol slowest


def test_symbols_since_last() -> None:
    assert symbols_since_last((), 2) == 0
    assert symbols_since_last((0, 1), 2) == 2
    assert symbols_since_last((0, 2), 2) == 0
    assert symbols_since_last((2, 0, 1), 2) == 2


def test_wrong_bit_weight_recovers_a_mixture() -> None:
    p = hand_parity_pdfa()
    per_state = state_signatures(p, 3)
    right, wrong = per_state[[0, 2, 4]], per_state[[1, 3, 5]]
    lam_true = np.array([0.0, 0.3, 0.8])
    model = (1 - lam_true[:, None]) * right + lam_true[:, None] * wrong
    lam, resid = wrong_bit_weight(model, right, wrong)
    assert np.allclose(lam, lam_true)
    assert np.allclose(resid, 0.0)


def test_belief_weights_recover_a_mixture_of_states() -> None:
    p = hand_parity_pdfa()
    per_state = state_signatures(p, 3)
    rng = np.random.default_rng(0)
    b_true = rng.dirichlet(np.ones(6), size=20)
    b_true[0] = np.eye(6)[2]  # a pure state too
    b, resid = belief_weights(b_true @ per_state, per_state)
    assert np.allclose(b, b_true, atol=1e-3)
    assert np.all(resid < 1e-3)
