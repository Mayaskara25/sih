"""D3 -- score normalization. Nothing else may define a normalization."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import median_filter
from scipy.stats import rankdata

from core.contracts import SceneMeta
from preprocessing.bands import select_band


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


# --- S3A.8 -- fusion inputs: target signature, ACE, spectral indices, spatial context -----

def estimate_target_signature(cube: np.ndarray, base_score: np.ndarray, *,
                                top_frac: float = 0.001) -> np.ndarray:
    """Unsupervised target signature: mean spectrum of the top `top_frac` pixels
    by `base_score`.

    ACE (`ace_score`) needs a target signature to test pixels against, but
    unsupervised anomaly detection has none by definition -- there is no
    ground-truth target library available at scoring time. It is bootstrapped
    instead from whatever base detector already ran (typically `global_rx`):
    its highest-scoring pixels are the best unsupervised guess at "what a
    target looks like" in this scene.

    A pixel only counts if it is valid in BOTH `cube` (no NaN band) and
    `base_score` (not NaN) -- a NaN `base_score` pixel (background stats
    could not be computed there, or it fell outside a mask) must never be
    ranked into the top set just because its cube values happen to be finite.

    Parameters
    ----------
    cube : [H, W, B] float32, NaN = nodata.
    base_score : [H, W] float, NaN = no score (e.g. RX output).
    top_frac : fraction of valid pixels to average; at least 1 pixel always.

    Returns
    -------
    [B] float32 mean spectrum of the top-scoring pixels.
    """
    h, w, b = cube.shape
    flat_cube = cube.reshape(-1, b)
    flat_score = np.asarray(base_score, dtype=np.float64).reshape(-1)

    valid = ~np.any(np.isnan(flat_cube), axis=-1) & ~np.isnan(flat_score)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("estimate_target_signature: no pixel is valid in both cube and base_score")

    n_top = max(1, int(round(top_frac * n_valid)))
    valid_idx = np.flatnonzero(valid)
    # Descending stable sort by score -- ties resolved by ascending flat
    # index, which is an arbitrary but deterministic and reproducible choice.
    order = np.argsort(flat_score[valid_idx], kind="stable")[::-1]
    top_idx = valid_idx[order[:n_top]]

    return flat_cube[top_idx].mean(axis=0).astype(np.float32)


def ace_score(cube: np.ndarray, signature: np.ndarray, *, mean: np.ndarray | None = None,
              cov: np.ndarray | None = None, reg: float = 1e-6) -> np.ndarray:
    """Adaptive Cosine Estimator.

        ACE(x) = (s'^T Sigma^-1 x')^2 / ((s'^T Sigma^-1 s')(x'^T Sigma^-1 x'))

    where `x' = x - mu` and `s' = signature - mu`. This is the squared cosine
    of the angle between the centered pixel and the centered target signature
    in the whitened (Sigma^-1) space -- by Cauchy-Schwarz it is bounded in
    [0, 1] whenever Sigma is positive definite, which the `reg * I` term
    guarantees. That bound is asserted, not just hoped for: floating-point
    round-off can push a value a hair past 1.0 (e.g. a pixel exactly equal to
    `signature`), so the result is clipped.

    `mean`/`cov` let a caller reuse background statistics already computed
    elsewhere (e.g. `global_rx`'s mu/Sigma) instead of re-estimating them from
    this cube; if omitted, both are estimated from this cube's valid pixels
    (population covariance, matching `anomaly.rx.global_rx`'s convention).
    Never an explicit inverse -- `cho_factor`/`cho_solve`, per CONTRIBUTING.md.

    Parameters
    ----------
    cube : [H, W, B] float32, NaN = nodata.
    signature : [B] target spectrum, e.g. from `estimate_target_signature`.
    mean : [B] optional background mean; estimated from `cube` if None.
    cov : [B, B] optional background covariance; estimated from `cube` if None.
    reg : Tikhonov regularization added to the covariance diagonal.

    Returns
    -------
    [H, W] float32 in [0, 1], NaN wherever the input pixel was NaN.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    valid = ~np.any(np.isnan(flat), axis=-1)
    data = flat[valid].astype(np.float64)

    mu = np.asarray(mean, dtype=np.float64) if mean is not None else data.mean(axis=0)
    if cov is not None:
        sigma = np.asarray(cov, dtype=np.float64)
    else:
        centered = data - mu
        sigma = (centered.T @ centered) / data.shape[0]
    sigma = sigma + reg * np.eye(b, dtype=sigma.dtype)

    c, lower = cho_factor(sigma)

    s = np.asarray(signature, dtype=np.float64).reshape(-1) - mu
    s_inv = cho_solve((c, lower), s)               # Sigma^-1 s'
    s_dot = float(s @ s_inv)                        # s'^T Sigma^-1 s'

    dv = data - mu                                  # x' for every valid pixel
    dv_inv = cho_solve((c, lower), dv.T)             # Sigma^-1 x', [B, N]
    numer = (dv @ s_inv) ** 2                        # (s'^T Sigma^-1 x')^2, [N]
    denom_x = np.einsum("ij,ji->i", dv, dv_inv)      # x'^T Sigma^-1 x', [N]

    denom = s_dot * denom_x
    scores_valid = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    scores_valid = np.clip(scores_valid, 0.0, 1.0)

    scores = np.full(flat.shape[0], np.nan, dtype=np.float64)
    scores[valid] = scores_valid
    return scores.reshape(h, w).astype(np.float32)


# Named bands used by the D6 index set. Wavelengths (nm) are the conventional
# multispectral definitions (Landsat/Sentinel-2-like band centers), looked up
# per-scene by nearest wavelength (preprocessing.bands) -- never by index,
# which differs per sensor (D9).
_INDEX_BAND_WAVELENGTHS_NM = {
    "blue": 480.0,
    "green": 560.0,
    "red": 660.0,
    "nir": 860.0,
    "swir1": 1650.0,
    "swir2": 2200.0,
}

# name -> (required named bands, formula over {band_name: reflectance array})
_INDEX_DEFINITIONS: dict[str, tuple[tuple[str, ...], "callable"]] = {
    "ndbi": (("swir1", "nir"),
             lambda r: (r["swir1"] - r["nir"]) / (r["swir1"] + r["nir"])),
    "iron_oxide_ratio": (("red", "blue"),
                          lambda r: r["red"] / r["blue"]),
    "clay_ratio": (("swir1", "swir2"),
                   lambda r: r["swir1"] / r["swir2"]),
    "brightness": (("blue", "green", "red", "nir"),
                   lambda r: (r["blue"] + r["green"] + r["red"] + r["nir"]) / 4.0),
    "ndvi": (("nir", "red"),
             lambda r: (r["nir"] - r["red"]) / (r["nir"] + r["red"])),
    "ndwi": (("green", "nir"),
             lambda r: (r["green"] - r["nir"]) / (r["green"] + r["nir"])),
    "nbr": (("nir", "swir2"),
            lambda r: (r["nir"] - r["swir2"]) / (r["nir"] + r["swir2"])),
    "bsi": (("swir1", "red", "nir", "blue"),
            lambda r: ((r["swir1"] + r["red"]) - (r["nir"] + r["blue"]))
                       / ((r["swir1"] + r["red"]) + (r["nir"] + r["blue"]))),
}


def spectral_index_score(cube: np.ndarray, meta: SceneMeta, index_names: list[str]) -> np.ndarray:
    """Fuse a set of named spectral indices into one [0,1] anomaly-oriented score.

    The index SET comes from `configs/target_profile.yaml` (D6) -- the caller
    reads `object.indices` or `landcover.indices` from that config and passes
    it in; nothing here hardcodes which indices belong to which profile.

    Band selection is by nearest wavelength via `preprocessing.bands.select_band`
    -- never by band index, which differs per sensor (D9). That is also why
    this function is unavailable on ABU, HYDICE and Indian Pines: none of
    them ship a wavelength array (D13.4), so `select_band` raises `ValueError`
    rather than fabricate a band. **This is deliberate, not a bug** -- D20's
    `fuse_scores` catches exactly this `ValueError` and drops the `index`
    component from the fused score, renormalizing the remaining weights,
    instead of ever silently guessing a band.

    Each requested index is computed on its own [0,1]-ish native scale (an
    NDVI-style ratio, or a raw reflectance-ratio for `iron_oxide_ratio` /
    `clay_ratio`), then rank-normalized (D3's `rank_normalize`, already
    defined in this module -- not reimplemented) before being averaged, so
    that indices on incompatible native scales do not silently dominate one
    another inside this function, before it is ever handed to
    `anomaly.fusion.fuse_scores` as a single component.

    Parameters
    ----------
    cube : [H, W, B] float32, NaN = nodata.
    meta : SceneMeta; must carry `wavelengths` (see above).
    index_names : names drawn from `configs/target_profile.yaml`'s `indices`
        list, e.g. `["ndbi", "iron_oxide_ratio", "clay_ratio", "brightness"]`.

    Returns
    -------
    [H, W] float32 in [0, 1], NaN wherever the input pixel was NaN.

    Raises
    ------
    ValueError
        If `meta.wavelengths` is None (propagated from `select_band`), or an
        unknown index name is requested.
    """
    if not index_names:
        raise ValueError("spectral_index_score: index_names is empty")

    unknown = [n for n in index_names if n not in _INDEX_DEFINITIONS]
    if unknown:
        raise ValueError(
            f"spectral_index_score: unknown index name(s) {unknown}, expected one of "
            f"{sorted(_INDEX_DEFINITIONS)}")

    # select_band raises ValueError itself when meta.wavelengths is None
    # (D13.4/D20) -- deliberately not caught here, so the caller sees exactly
    # why this component is unavailable rather than a fabricated array.
    band_cache: dict[str, int] = {}

    def _band_idx(name: str) -> int:
        if name not in band_cache:
            band_cache[name] = select_band(meta, _INDEX_BAND_WAVELENGTHS_NM[name])
        return band_cache[name]

    valid = ~np.any(np.isnan(cube), axis=-1)

    per_index = []
    for name in index_names:
        needed, formula = _INDEX_DEFINITIONS[name]
        bands = {n: cube[..., _band_idx(n)].astype(np.float64) for n in needed}
        with np.errstate(divide="ignore", invalid="ignore"):
            idx_val = formula(bands).astype(np.float32)
        idx_val = np.where(valid, idx_val, np.nan).astype(np.float32)
        per_index.append(rank_normalize(idx_val, valid=valid))

    # Every per_index entry shares the same `valid` mask (each was built from
    # it above), so a plain mean -- not nanmean -- is exactly right: valid
    # pixels are finite in every entry and average normally; invalid pixels
    # are NaN in every entry and a plain mean propagates that NaN, with no
    # "empty slice" case to warn about.
    stacked = np.stack(per_index, axis=0)
    out = np.mean(stacked, axis=0).astype(np.float32)
    return out


def spatial_context_score(score: np.ndarray, *, k: int = 7) -> np.ndarray:
    """Local contrast: `score - median_filter(score, k)`, clipped at 0.

    Suppresses broad regional trends (a slowly varying illumination or
    terrain gradient) while keeping compact deviations (a small anomalous
    cluster of pixels) -- a median filter tracks the slow trend but is not
    pulled toward a small spike the way a mean filter would be, so the
    residual after subtracting it is large exactly where a compact anomaly
    sits and near zero across a smooth ramp.

    NaN pixels are filled with the scene's valid-pixel median before
    filtering (not left as NaN): `scipy.ndimage.median_filter` does not
    special-case NaN, and a NaN inside a window can silently poison
    neighbouring outputs depending on how it sorts -- the same class of
    `0.0 * NaN == NaN` surprise D15 hit in `harmonize()`, just via median
    instead of a matmul. Filling with a constant keeps the filter numerically
    well-defined everywhere; the result is masked back to NaN at genuinely
    invalid pixels afterward, so this never fabricates a score for a nodata
    pixel, only keeps it from contaminating its valid neighbours' filtering.

    Parameters
    ----------
    score : [H, W] float, NaN = no score.
    k : median filter window size (odd, per `scipy.ndimage.median_filter`).

    Returns
    -------
    [H, W] float32, >= 0, NaN wherever the input pixel was NaN.
    """
    valid = ~np.isnan(score)
    fill_value = float(np.median(score[valid])) if valid.any() else 0.0
    filled = np.where(valid, score, fill_value).astype(np.float64)

    smoothed = median_filter(filled, size=k, mode="reflect")
    context = np.clip(filled - smoothed, 0.0, None)

    out = np.where(valid, context, np.nan)
    return out.astype(np.float32)
