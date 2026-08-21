"""D3 -- score normalization. Nothing else may define a normalization."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def percentile_normalize(score: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.9,
                          valid: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Storage / thresholding normalization -- invertible.

    v_lo/v_hi/p_lo/p_hi are meant to be written into the GeoTIFF tags (C2) so
    the raw score is recoverable from the normalized product.
    """
    if valid is None:
        valid = ~np.isnan(score)
    v_lo = float(np.percentile(score[valid], p_lo))
    v_hi = float(np.percentile(score[valid], p_hi))
    span = v_hi - v_lo if v_hi != v_lo else 1.0
    norm = np.clip((score - v_lo) / span, 0, 1)
    norm = np.where(np.isnan(score), np.nan, norm).astype(np.float32)
    return norm, v_lo, v_hi


def rank_normalize(score: np.ndarray, *, valid: np.ndarray | None = None) -> np.ndarray:
    """Fusion normalization -- scale-free, within-scene only.

    Never stored as a product (D3): only used inside anomaly/fusion.py to put
    heterogeneous detector outputs on a common scale before combining them.
    """
    if valid is None:
        valid = ~np.isnan(score)
    out = np.full(score.shape, np.nan, dtype=np.float32)
    n = int(valid.sum())
    if n > 1:
        ranks = rankdata(score[valid], method="average")
        out[valid] = (ranks - 1) / (n - 1)
    elif n == 1:
        out[valid] = 0.5
    return out


def threshold_by_percentile(norm_score: np.ndarray, *, pct: float) -> np.ndarray:
    """norm_score in [0,1] -> C3 uint8 mask, background=0, target=1."""
    valid = ~np.isnan(norm_score)
    mask = np.zeros(norm_score.shape, dtype=np.uint8)
    if valid.any():
        cutoff = np.percentile(norm_score[valid], pct)
        mask[valid & (norm_score >= cutoff)] = 1
    return mask
