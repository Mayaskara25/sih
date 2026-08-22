"""PLAN.md §3C.4 -- difference-space structure statistics and physics fusion."""
from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from anomaly.scoring import rank_normalize
from core.contracts import validate_mask

_DEFAULT_WEIGHTS: dict[str, float] = {
    "sam": 0.50, "variance": 0.20, "entropy": 0.15, "coherence": 0.15,
}
_ENTROPY_BINS = 16
_STRUCTURE_KEYS = ("variance", "entropy", "coherence")


def difference_structure(cube_t1: np.ndarray, cube_t2: np.ndarray, *,
                         patch: int = 7, bins: int = _ENTROPY_BINS
                         ) -> dict[str, np.ndarray]:
    """Patch-wise statistics of the difference space (§3C.4).

    Genuine change and pseudo-change have measurably different structure in
    the difference cube d = t2 - t1:

      variance  -- local variance of the per-band difference, AVERAGED OVER
                   BANDS (documented simplification: a single [H, W] map is
                   required; the band mean keeps spectral-localisation while
                   staying comparable across sensors).
      entropy   -- Shannon entropy (log2) of the histogram of the BAND-MEAN
                   difference within each patch. Fixed bin count `bins=16`
                   over the scene-wide finite range of the band-mean
                   difference -- a fixed global range (not per-patch ranges)
                   is what makes entropies comparable BETWEEN patches.
      coherence -- mean absolute OFF-DIAGONAL correlation of the B x B
                   correlation matrix of the difference vectors inside each
                   patch: how tightly the bands move together. Computed per
                   patch from sufficient statistics via einsum, never by a
                   Python loop over patches.

    Edge handling (consistent for all three maps): REFLECT padding of the
    difference cube and valid mask by patch//2 on every side, so every output
    pixel gets a full centred patch×patch window and the output grid equals
    the input grid -- border patches mirror inward rather than shrink.

    NaN policy: pixels with any NaN band are EXCLUDED from the statistics of
    every window they fall into (invalid entries are zeroed before any
    arithmetic -- NaN * 0 is NaN), but the OUTPUT is NaN at exactly those
    positions (positional propagation). A degenerate patch (constant
    difference, or fewer than 2 valid pixels) has no correlation evidence and
    yields coherence 0.0 there -- benign for fusion, not missing data.

    Parameters
    ----------
    cube_t1, cube_t2 : [H, W, B] float32, same shape, NaN = nodata.
    patch : odd positive side length of the analysis window.
    bins : histogram bin count for the entropy map.

    Returns
    -------
    dict with keys 'variance', 'entropy', 'coherence', each [H, W] float32.
    """
    t1 = np.asarray(cube_t1)
    t2 = np.asarray(cube_t2)
    if t1.shape != t2.shape:
        raise ValueError(f"difference_structure: shape mismatch {t1.shape} vs {t2.shape}")
    if t1.ndim != 3:
        raise ValueError(f"difference_structure: expected [H, W, B] cubes, got shape {t1.shape}")
    if patch < 1 or patch % 2 == 0:
        raise ValueError(f"difference_structure: patch must be odd and >= 1, got {patch}")
    if bins < 2:
        raise ValueError(f"difference_structure: bins must be >= 2, got {bins}")

    # float64 accumulation (D24) before variance/covariance squaring.
    diff = t2.astype(np.float64) - t1.astype(np.float64)
    h, w, b = diff.shape
    valid = ~np.isnan(diff).any(axis=-1)                      # [H, W]

    pad = patch // 2
    dpad = np.pad(diff, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    vpad = np.pad(valid, pad, mode="edge")

    # Windows: [H, W, P, P, B] differences and [H, W, P, P] validity masks.
    win_d = np.moveaxis(sliding_window_view(dpad, (patch, patch), axis=(0, 1)), 2, -1)
    win_v = sliding_window_view(vpad, (patch, patch), axis=(0, 1))

    cnt = win_v.sum(axis=(-2, -1)).astype(np.float64)          # valid px per patch

    # Zero out invalid pixels BEFORE any arithmetic -- NaN * 0 is NaN, so
    # multiplying by the mask would poison every window touching nodata.
    win_d = np.where(win_v[..., None], win_d, 0.0)

    # --- variance --------------------------------------------------------
    s1 = win_d.sum(axis=(2, 3))                                # [H, W, B]
    s2 = (win_d * win_d).sum(axis=(2, 3))                      # [H, W, B]
    mean_b = s1 / cnt[..., None]
    var_b = np.clip(s2 / cnt[..., None] - mean_b * mean_b, 0.0, None)
    variance_map = var_b.mean(axis=-1)

    # --- entropy ---------------------------------------------------------
    band_mean = np.where(valid, np.nanmean(diff, axis=-1), np.nan)
    finite = band_mean[np.isfinite(band_mean)]
    if finite.size == 0:
        entropy_map = np.full((h, w), np.nan)
    else:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi <= lo:
            hi = lo + 1.0                    # constant scene -> one occupied bin
        edges = np.linspace(lo, hi, bins + 1)[1:-1]
        idx = np.digitize(band_mean, edges)                     # NaN -> out of range
        onehot = np.zeros((h, w, bins), dtype=np.float64)
        for k in range(bins):
            onehot[..., k] = ((idx == k) & valid).astype(np.float64)
        ohpad = np.pad(onehot, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        win_oh = np.moveaxis(sliding_window_view(ohpad, (patch, patch), axis=(0, 1)), 2, -1)
        bin_counts = win_oh.sum(axis=(2, 3))                    # [H, W, bins]
        p = bin_counts / cnt[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log2(p), 0.0)
        entropy_map = -terms.sum(axis=-1)

    # --- coherence -------------------------------------------------------
    flat_d = win_d.reshape(h, w, patch * patch, b)
    flat_m = win_v.reshape(h, w, patch * patch).astype(np.float64)
    sum1 = np.einsum("ijp,ijpr->ijr", flat_m, flat_d)                # [H, W, B]
    sum2 = np.einsum("ijp,ijpr,ijps->ijrs", flat_m, flat_d, flat_d)  # [H, W, B, B]
    mu = sum1 / cnt[..., None]
    cov = sum2 / cnt[..., None, None] - mu[..., :, None] * mu[..., None, :]
    var = np.einsum("ijbb->ijb", cov)
    std = np.sqrt(np.clip(var, 0.0, None))
    denom = std[..., :, None] * std[..., None, :]
    corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)

    off_diag = ~np.eye(b, dtype=bool)
    abs_corr = np.abs(np.where(off_diag, corr, np.nan))
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            coherence_map = np.nanmean(abs_corr, axis=(-2, -1))
    # Degenerate patches -- constant difference (zero variance in every band)
    # or <2 valid pixels -- carry NO co-movement evidence. That is a benign
    # zero for fusion, not missing data: emit 0.0 so static interiors stay
    # finite and scoreable (positional NaN is still restored at the end).
    degenerate = ~np.isfinite(coherence_map) | (cnt < 2)
    coherence_map[degenerate] = 0.0

    nan_out = ~valid
    maps = {"variance": variance_map, "entropy": entropy_map,
            "coherence": coherence_map}
    return {k: np.where(nan_out, np.nan, v).astype(np.float32)
            for k, v in maps.items()}


def fuse_change_signals(sam: np.ndarray, structure: dict[str, np.ndarray],
                        cloud_mask: np.ndarray,
                        weights: dict[str, float] | None = None) -> np.ndarray:
    """Fuse SAM and difference-structure signals into one change score.

    Each signal is rank-normalized across its valid pixels (D3's
    `anomaly.scoring.rank_normalize` -- scale-free, within-scene only; never
    reimplemented here), then combined as a weighted sum. The weights are
    used AS GIVEN (no renormalization): the defaults form a convex combination
        {sam: 0.50, variance: 0.20, entropy: 0.15, coherence: 0.15}
    and a caller passing non-convex weights owns that decision.

    Components given weight 0 are excluded entirely -- neither summed nor
    consulted for NaN -- so e.g. all-weight-on-sam reproduces
    `rank_normalize(sam)` exactly even where the structure maps are NaN.

    cloud_mask follows C3 (uint8 {0, 1}) and is validated via
    `core.contracts.validate_mask`. Output is ZERO where cloud_mask == 1;
    NaN (nodata) takes precedence over zeroing wherever ANY active input
    signal was NaN.

    Returns [H, W] float32.
    """
    mask = np.asarray(cloud_mask)
    validate_mask(mask)
    w = dict(_DEFAULT_WEIGHTS if weights is None else weights)
    unknown = set(w) - set(_DEFAULT_WEIGHTS)
    missing = set(_DEFAULT_WEIGHTS) - set(w)
    if unknown or missing:
        raise ValueError(
            f"fuse_change_signals: bad weights keys (unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}), expected {sorted(_DEFAULT_WEIGHTS)}")
    negative = {k: v for k, v in w.items() if v < 0}
    if negative:
        raise ValueError(f"fuse_change_signals: negative weights {negative}")

    signals = {"sam": sam}
    for key in _STRUCTURE_KEYS:
        if key not in structure:
            raise ValueError(f"fuse_change_signals: structure missing {key!r}")
        signals[key] = structure[key]

    active = {k: v for k, v in w.items() if v != 0.0}

    normed: dict[str, np.ndarray] = {}
    nan_any = np.zeros(np.shape(sam), dtype=bool)
    for key in active:
        sig = np.asarray(signals[key], dtype=np.float32)
        normed[key] = rank_normalize(sig)
        nan_any |= np.isnan(sig)

    fused = np.zeros(np.shape(sam), dtype=np.float64)
    for key, weight in active.items():
        fused += weight * normed[key]
    fused = np.where(nan_any, np.nan, fused).astype(np.float32)

    cloudy = mask == 1
    fused[cloudy & ~nan_any] = 0.0
    return fused
