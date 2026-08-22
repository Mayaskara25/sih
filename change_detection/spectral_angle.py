"""PLAN.md §3C.2 -- Spectral Angle Mapper change signal (the primary arm)."""
from __future__ import annotations

import numpy as np


def spectral_angle(cube_t1: np.ndarray, cube_t2: np.ndarray) -> np.ndarray:
    """Spectral Angle Mapper between co-registered epoch cubes, per pixel.

        SAM(x1, x2) = arccos( <x1,x2> / (||x1|| ||x2||) )  ∈ [0, π]

    Invariant to uniform brightness scaling by construction, which is exactly
    what makes it robust to the illumination/seasonal shifts that generate
    most pseudo-change (§3C.2). Returns [H, W] float32 radians.

    A pixel where EITHER epoch is all-zero (or otherwise has zero norm) has
    no direction to compare -- the angle is undefined there and NaN is
    emitted, not a fabricated 0 or π. The cosine argument is clipped to
    [-1, 1]: floating-point round-off can push <x1,x2>/(||x1||||x2||) a hair
    past ±1 for near-collinear spectra, and arccos of that is NaN -- the same
    clip-guard rationale as `ace_score` in anomaly/scoring.py.

    NaN = nodata propagates POSITIONALLY: a NaN band in either cube makes the
    dot product NaN, so the output is NaN at exactly the [H, W] locations
    where either input pixel was NaN.

    Parameters
    ----------
    cube_t1, cube_t2 : [H, W, B] float32, same shape, NaN = nodata.

    Returns
    -------
    [H, W] float32 radians in [0, π]; NaN at all-zero/norm-0/NaN pixels.

    Raises
    ------
    ValueError
        If the two cubes differ in shape or are not 3-D.
    """
    t1 = np.asarray(cube_t1)
    t2 = np.asarray(cube_t2)
    if t1.shape != t2.shape:
        raise ValueError(f"spectral_angle: shape mismatch {t1.shape} vs {t2.shape}")
    if t1.ndim != 3:
        raise ValueError(f"spectral_angle: expected [H, W, B] cubes, got shape {t1.shape}")

    # float64 accumulation (D24): the per-pixel dot products over B bands lose
    # precision in float32 for radiance-scale data, and a silently biased
    # cosine near ±1 turns into spurious exact-0/exact-π angles after arccos.
    a = t1.astype(np.float64)
    c = t2.astype(np.float64)

    dot = np.einsum("ijk,ijk->ij", a, c)
    n1 = np.sqrt(np.einsum("ijk,ijk->ij", a, a))
    n2 = np.sqrt(np.einsum("ijk,ijk->ij", c, c))

    denom = n1 * n2
    # denom == 0 covers the all-zero pixel of either epoch; a NaN denominator
    # fails the `> 0` test and stays NaN, so nodata needs no separate mask.
    safe = denom > 0
    cos = np.divide(dot, denom, out=np.full(dot.shape, np.nan), where=safe)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos).astype(np.float32)
