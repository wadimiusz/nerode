"""Smoke tests: the automata, the checkpoint cache key, and the quick path of
the continuations experiment, end to end on a CPU (well under a minute)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from nerode.automata import hand_parity_pdfa, minimise, random_pdfa, sample
from nerode.training import ckpt_path, make_model

ROOT = Path(__file__).resolve().parents[1]


def test_sample_follows_the_automaton() -> None:
    rng = np.random.default_rng(0)
    p = random_pdfa(5, 3, rng)
    syms, states = sample(p, 10, 8, rng)
    assert syms.shape == (10, 8) and states.shape == (10, 9)
    for i in range(10):
        q = 0
        for t in range(8):
            assert states[i, t] == q
            q = int(p.delta[q, syms[i, t]])
        assert states[i, 8] == q


def test_parity_automaton_semantics() -> None:
    """States are 2g + b: 0 flips the bit, 1 keeps it, 2 reads and resets it,
    and all six states are distinct Nerode classes."""
    p = hand_parity_pdfa()
    assert int(minimise(p).max()) + 1 == 6
    for q in range(6):
        g, b = divmod(q, 2)
        assert p.delta[q, 0] % 2 == 1 - b
        assert p.delta[q, 1] % 2 == b
        assert p.delta[q, 2] % 2 == 0
        assert np.allclose(p.emit[q], p.emit[2 * g])  # emissions depend on g only


def test_ckpt_key_matches_the_pilot_cache() -> None:
    """The checkpoint cache is keyed by `ckpt_path`. The pilot's trained models
    (results/continuations_fixed.txt, results/bit_probe_concat.*) sit in the
    cache under these keys, so a drift here would silently retrain them."""
    p = hand_parity_pdfa()
    model = make_model(3, 24, 64, 2, 0)
    dummy = torch.zeros((1, 25), dtype=torch.long)
    path = ckpt_path(
        model, p, dummy, steps=100_000, seed=0, seq_len=24, batch=128, fresh=True
    )
    assert path.name == "2469b26d8af5dab4.pt"


def test_quick_continuations_end_to_end(tmp_path: Path) -> None:
    """`scripts/nerode_continuations.py --quick`: one seed, 200 steps, 60
    prefixes, both automaton kinds; on the CPU, into a temporary directory."""
    out = tmp_path / "continuations.txt"
    # outputs redirected; CUDA hidden so the test measures the CPU path everywhere
    env = {
        **os.environ,
        "NERODE_OUT": str(tmp_path / "out"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    script = ROOT / "scripts" / "nerode_continuations.py"
    subprocess.run(
        [sys.executable, str(script), "--quick", "--out", str(out)],
        check=True,
        cwd=ROOT,
        env=env,
        timeout=600,
        capture_output=True,
    )
    text = out.read_text()
    assert "plain PDFA, seed 0" in text
    assert "parity PDFA, seed 0" in text
    assert "trained model tracks the hidden bit" in text
    # 2 kinds x (random, trained) x 4 layers (embed, block0, block1, norm_f)
    rows = [
        line.split()
        for line in text.splitlines()
        if line.strip().startswith(("random", "trained"))
    ]
    assert len(rows) == 16
    for row in rows:
        assert row[1] in {"embed", "block0", "block1", "norm_f"}
        loss, rho_beh = float(row[2]), float(row[3])
        assert 0 < loss < 10 and -1 <= rho_beh <= 1
    # the trained models were cached, in the redirected directory only
    assert len(list((tmp_path / "out" / "ckpt").glob("*.pt"))) == 2
