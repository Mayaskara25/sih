"""PLAN.md §2.3."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def global_rx(cube: np.ndarray, *, reg: float = 1e-6) -> np.ndarray:
    """Global Reed-Xiaoli detector.

    r(x) = (x - mu)^T @ inv(Sigma + reg*I) @ (x - mu)
    mu, Sigma estimated over ALL valid pixels.
    Returns [H, W] float32, NaN where the input pixel was NaN.
    Uses scipy.linalg.cho_factor / cho_solve -- never an explicit inverse.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    valid = ~np.any(np.isnan(flat), axis=-1)

    mu = flat[valid].mean(axis=0)
    centered = flat[valid] - mu
    sigma = (centered.T @ centered) / valid.sum()
    sigma = sigma + reg * np.eye(b, dtype=sigma.dtype)

    c, lower = cho_factor(sigma)

    scores = np.full(flat.shape[0], np.nan, dtype=np.float64)
    dv = flat[valid] - mu
    solved = cho_solve((c, lower), dv.T)
    scores[valid] = np.einsum("ij,ji->i", dv, solved)

    return scores.reshape(h, w).astype(np.float32)
