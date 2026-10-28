"""Rank correlations and pairwise distance matrices."""

from __future__ import annotations

import numpy as np


def ranks(x: np.ndarray) -> np.ndarray:
    return x.argsort().argsort().astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank correlation of x and y controlling for z."""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    c = np.corrcoef(np.stack([rx, ry, rz]))
    rxy, rxz, ryz = c[0, 1], c[0, 2], c[1, 2]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2)))


def cosine_dist_matrix(h: np.ndarray) -> np.ndarray:
    """h: (n, d) -> (n, n)."""
    hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-9)
    return 1.0 - hn @ hn.T


def tv_matrix(p: np.ndarray) -> np.ndarray:
    """p: (n, k) distributions -> (n, n) total variation."""
    return 0.5 * np.abs(p[:, None, :] - p[None, :, :]).sum(-1)
