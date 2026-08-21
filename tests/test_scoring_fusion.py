"""S3A.8 -- estimate_target_signature, ace_score, spectral_index_score,
spatial_context_score. D20's guard (spectral_index_score raises rather than
guessing a band on a wavelength-less scene) is the single most important
test in this file -- it is what makes D20 structural rather than merely
documented.
"""
import affine
import numpy as np
import pytest
import rasterio.crs

from core.contracts import SceneMeta
from anomaly.scoring import (
    ace_score,
    estimate_target_signature,
    rank_normalize,
    spatial_context_score,
    spectral_index_score,
)


def _meta(wavelengths, *, source="had100", scene_id="s1", bad_bands=None):
    b = 0 if wavelengths is None else len(wavelengths)
    return SceneMeta(
        scene_id=scene_id,
        crs=rasterio.crs.CRS.from_epsg(32615),
        transform=affine.Affine(10.0, 0, 0, 0, -10.0, 0),
        wavelengths=wavelengths,
        bad_bands=np.zeros(b, dtype=bool) if bad_bands is None else bad_bands,
        gsd_m=10.0,
        source=source,
        georef="real",
    )


# --- estimate_target_signature ----------------------------------------------

def test_estimate_target_signature_hand_built():
    """10 pixels, spectrum[i] = [i, i], base_score = i (strictly increasing).
    top_frac=0.3 -> top 3 pixels are indices 7,8,9 -> mean spectrum [8,8]."""
    cube = np.zeros((1, 10, 2), dtype=np.float32)
    for i in range(10):
        cube[0, i] = [i, i]
    base_score = np.arange(10, dtype=np.float64).reshape(1, 10)

    sig = estimate_target_signature(cube, base_score, top_frac=0.3)
    np.testing.assert_allclose(sig, [8.0, 8.0])


def test_estimate_target_signature_at_least_one_pixel():
    cube = np.zeros((1, 5, 2), dtype=np.float32)
    for i in range(5):
        cube[0, i] = [i, i]
    base_score = np.arange(5, dtype=np.float64).reshape(1, 5)
    sig = estimate_target_signature(cube, base_score, top_frac=0.001)
    np.testing.assert_allclose(sig, [4.0, 4.0])  # single highest-scoring pixel


def test_estimate_target_signature_excludes_pixels_invalid_in_either_input():
    """A pixel with the highest base_score but a NaN band must not be picked,
    even though its base_score alone looks like the top choice."""
    cube = np.zeros((1, 5, 2), dtype=np.float32)
    for i in range(5):
        cube[0, i] = [i, i]
    cube[0, 4] = np.nan               # highest-score pixel is cube-invalid
    base_score = np.array([0.0, 1.0, 2.0, 3.0, 100.0]).reshape(1, 5)

    sig = estimate_target_signature(cube, base_score, top_frac=0.25)
    # excluding index 4, the remaining top-1 (25% of 4 valid) is index 3 -> [3,3]
    np.testing.assert_allclose(sig, [3.0, 3.0])


def test_estimate_target_signature_raises_when_nothing_valid():
    cube = np.full((1, 3, 2), np.nan, dtype=np.float32)
    base_score = np.array([1.0, 2.0, 3.0]).reshape(1, 3)
    with pytest.raises(ValueError):
        estimate_target_signature(cube, base_score)


# --- ace_score ----------------------------------------------------------------

def test_ace_score_strictly_in_unit_interval():
    rng = np.random.default_rng(0)
    h, w, b = 24, 24, 6
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[10, 10] += rng.normal(size=b) * 15.0   # implanted outlier

    base_score = np.linalg.norm(cube - cube.reshape(-1, b).mean(axis=0), axis=-1)
    sig = estimate_target_signature(cube, base_score, top_frac=0.01)
    scores = ace_score(cube, sig)

    valid = ~np.isnan(scores)
    assert np.all(scores[valid] >= 0.0)
    assert np.all(scores[valid] <= 1.0)


def test_ace_score_pixel_equal_to_signature_scores_near_one():
    rng = np.random.default_rng(1)
    h, w, b = 16, 16, 5
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    signature = np.array([3.0, -2.0, 1.5, 0.5, 2.0], dtype=np.float32)

    cube2 = cube.copy()
    cube2[4, 4] = signature
    scores = ace_score(cube2, signature)
    assert scores[4, 4] == pytest.approx(1.0, abs=1e-4)


def test_ace_score_nan_locality():
    rng = np.random.default_rng(2)
    h, w, b = 10, 10, 4
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[2, 3, :] = np.nan
    signature = rng.normal(size=b).astype(np.float32)

    scores = ace_score(cube, signature)
    assert np.isnan(scores[2, 3])
    assert not np.any(np.isnan(np.delete(scores.ravel(), 2 * w + 3)))


def test_ace_score_accepts_precomputed_mean_and_cov():
    rng = np.random.default_rng(3)
    h, w, b = 12, 12, 4
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    signature = rng.normal(size=b).astype(np.float32)

    flat = cube.reshape(-1, b).astype(np.float64)
    mu = flat.mean(axis=0)
    sigma = np.cov(flat.T, bias=True)

    scores_reused = ace_score(cube, signature, mean=mu, cov=sigma)
    scores_default = ace_score(cube, signature)
    np.testing.assert_allclose(scores_reused, scores_default, rtol=1e-5, atol=1e-6)


# --- spectral_index_score -------------------------------------------------

def test_spectral_index_score_raises_on_wavelength_less_scene():
    """THE guard: ABU/HYDICE/Indian Pines ship no wavelength array (D13.4).
    spectral_index_score must refuse rather than fabricate a band-index-based
    array -- this is what makes D20's fuse_scores fallback structural."""
    b = 20
    cube = np.zeros((4, 4, b), dtype=np.float32)
    meta = _meta(None, source="abu")
    with pytest.raises(ValueError, match="wavelengths"):
        spectral_index_score(cube, meta, ["ndvi"])


def test_spectral_index_score_raises_on_unknown_index_name():
    wl = np.arange(400, 2501, 10, dtype=np.float32)
    cube = np.zeros((2, 2, len(wl)), dtype=np.float32)
    meta = _meta(wl)
    with pytest.raises(ValueError, match="unknown"):
        spectral_index_score(cube, meta, ["not_a_real_index"])


def test_spectral_index_score_object_profile_shape_range_and_nan_locality():
    rng = np.random.default_rng(4)
    wl = np.arange(400, 2501, 10, dtype=np.float32)   # 211 bands, canonical grid
    b = len(wl)
    h, w = 12, 12
    cube = rng.uniform(0.01, 0.5, size=(h, w, b)).astype(np.float32)
    cube[3, 3, :] = np.nan
    meta = _meta(wl)

    out = spectral_index_score(cube, meta, ["ndbi", "iron_oxide_ratio", "clay_ratio", "brightness"])
    assert out.shape == (h, w)
    assert np.isnan(out[3, 3])
    valid = ~np.isnan(out)
    assert valid.sum() == h * w - 1
    assert np.all(out[valid] >= 0.0) and np.all(out[valid] <= 1.0)


def test_spectral_index_score_landcover_profile_runs():
    rng = np.random.default_rng(5)
    wl = np.arange(400, 2501, 10, dtype=np.float32)
    b = len(wl)
    cube = rng.uniform(0.01, 0.5, size=(10, 10, b)).astype(np.float32)
    meta = _meta(wl)
    out = spectral_index_score(cube, meta, ["ndvi", "ndwi", "nbr", "bsi"])
    assert out.shape == (10, 10)
    assert not np.any(np.isnan(out))


# --- spatial_context_score --------------------------------------------------

def test_spatial_context_score_clipped_at_zero():
    rng = np.random.default_rng(6)
    score = rng.normal(size=(30, 30)).astype(np.float32)
    out = spatial_context_score(score, k=7)
    assert np.nanmin(out) >= 0.0


def test_spatial_context_score_suppresses_ramp_but_keeps_spike():
    """A broad linear ramp is local-median-invariant in its interior (median
    of an odd symmetric window on a monotonic ramp equals the center value),
    so its residual is ~0 there; an isolated spike is untouched by the
    median (a single outlier never becomes the window median), so its
    residual survives at full amplitude. This is the whole point of the
    function -- suppress broad trend, keep compact deviation."""
    h, w = 50, 50
    rows = np.arange(h, dtype=np.float32).reshape(-1, 1)
    score = np.tile(rows, (1, w))
    score[25, 25] += 20.0

    out = spatial_context_score(score, k=7)

    interior_far_from_spike = out[10:40, 10:40].copy()
    interior_far_from_spike[10:21, 10:21] = np.nan   # excise the spike's neighbourhood
    ramp_residual = np.nanmax(interior_far_from_spike)
    spike_residual = out[25, 25]

    assert ramp_residual < 0.5
    assert spike_residual > 15.0
    assert spike_residual > 10 * max(ramp_residual, 1e-6)


def test_spatial_context_score_nan_locality():
    score = np.zeros((10, 10), dtype=np.float32)
    score[5, 5] = np.nan
    out = spatial_context_score(score, k=5)
    assert np.isnan(out[5, 5])
    assert not np.any(np.isnan(np.delete(out.ravel(), 5 * 10 + 5)))
