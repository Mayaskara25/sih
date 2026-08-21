"""§2.2 -- NaN-safe on a cube with 5% NaN pixels; no band's output variance is 0."""
import numpy as np

from preprocessing.normalize import l2_normalize, standardize


def _nan_cube(seed: int, frac: float = 0.05):
    rng = np.random.default_rng(seed)
    h, w, b = 40, 40, 6
    cube = rng.normal(loc=5.0, scale=2.0, size=(h, w, b)).astype(np.float32)
    nan_pixels = rng.random((h, w)) < frac
    cube[nan_pixels] = np.nan
    return cube, nan_pixels


def test_standardize_is_nan_safe_and_preserves_nan_positions():
    cube, nan_pixels = _nan_cube(0)
    out = standardize(cube)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert np.array_equal(np.isnan(out).any(axis=-1), nan_pixels)
    valid = ~nan_pixels
    for band in range(out.shape[-1]):
        assert np.nanvar(out[..., band][valid]) > 0


def test_l2_normalize_is_nan_safe_and_preserves_nan_positions():
    cube, nan_pixels = _nan_cube(1)
    out = l2_normalize(cube)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert np.array_equal(np.isnan(out).any(axis=-1), nan_pixels)
    valid = ~nan_pixels
    for band in range(out.shape[-1]):
        assert np.nanvar(out[..., band][valid]) > 0
