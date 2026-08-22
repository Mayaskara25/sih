"""PLAN.md §3C.3 -- magnitude difference (the baseline comparison arm)."""
from __future__ import annotations

import numpy as np


def magnitude_difference(cube_t1: np.ndarray, cube_t2: np.ndarray, *,
                         norm: str = "l2") -> np.ndarray:
    """Per-pixel magnitude of the temporal difference, ||x2 - x1||.

    Kept as the classical comparison arm, NOT as the primary signal -- it is
    what SAM is being measured against (§3C.3). Unlike SAM it responds to
    pure brightness shifts, which is precisely the pseudo-change failure mode
    the primary arm exists to suppress.

    Parameters
    ----------
    cube_t1, cube_t2 : [H, W, B] float32, same shape, NaN = nodata.
    norm : "l2" (default) for the Euclidean norm over bands, or "l1" for the
        sum of absolute per-band differences. Keyword-only.

    Returns
    -------
    [H, W] float32. NaN = nodata propagates POSITIONALLY: any NaN band in
    either input pixel yields NaN at that [H, W] location.

    Raises
    ------
    ValueError
        On shape mismatch, non-3-D input, or an unknown `norm`.
    """
    t1 = np.asarray(cube_t1)
    t2 = np.asarray(cube_t2)
    if t1.shape != t2.shape:
        raise ValueError(f"magnitude_difference: shape mismatch {t1.shape} vs {t2.shape}")
    if t1.ndim != 3:
        raise ValueError(f"magnitude_difference: expected [H, W, B] cubes, got shape {t1.shape}")
    if norm not in ("l2", "l1"):
        raise ValueError(f"magnitude_difference: unknown norm {norm!r}, expected 'l2' or 'l1'")

    # float64 accumulation (D24) before the norm collapses the band axis.
    diff = t2.astype(np.float64) - t1.astype(np.float64)

    # Both norms propagate NaN naturally: a NaN band makes the squared/abs
    # sum NaN, so no explicit mask is needed -- NaN lands at exactly the
    # input-NaN [H, W] positions.
    if norm == "l2":
        out = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    else:
        out = np.abs(diff).sum(axis=-1)
    return out.astype(np.float32)
