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
    # float64 accumulation (D24). The cube is float32, and a float32
    # `centered.T @ centered` over 10 000+ pixels loses precision and quietly
    # biases the covariance -- measured max relative error 1.15e-4 against a
    # float64 reference on abu-urban-3. This is the exact hazard PLAN.md
    # §3A.5 writes into the STREAMING spec ("float32 co-moment accumulation
    # loses precision over 20 000+ pixels and quietly biases the
    # covariance"); the warning was simply never applied to the reference
    # implementation it was meant to protect.
    flat = cube.reshape(-1, b).astype(np.float64, copy=False)
    valid = ~np.any(np.isnan(flat), axis=-1)

    mu = flat[valid].mean(axis=0)
    centered = flat[valid] - mu
    sigma = (centered.T @ centered) / valid.sum()
    # SCALE-RELATIVE ridge (D22). `reg` is a FRACTION of mean band variance,
    # not an absolute number of units. An absolute `reg * I` is meaningless
    # against data whose scale we do not control: ABU ships native radiance
    # with covariance diagonals of 1e4-1e6, where reg=1e-6 sits ten to twelve
    # orders of magnitude below the quantity it is supposed to condition, and
    # cho_factor raised LinAlgError on 3 of 13 ABU scenes. Scaling by
    # trace(sigma)/b makes `reg` mean the same thing on radiance, reflectance
    # and standardized input alike.
    sigma = sigma + reg * (np.trace(sigma) / b) * np.eye(b, dtype=sigma.dtype)

    c, lower = cho_factor(sigma)

    scores = np.full(flat.shape[0], np.nan, dtype=np.float64)
    dv = flat[valid] - mu
    solved = cho_solve((c, lower), dv.T)
    scores[valid] = np.einsum("ij,ji->i", dv, solved)

    return scores.reshape(h, w).astype(np.float32)
