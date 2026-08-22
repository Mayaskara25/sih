"""Tests for change_detection/spectral_angle.py (PLAN.md §3C.2)."""
from __future__ import annotations

import numpy as np
import pytest

from change_detection.spectral_angle import spectral_angle


def _cube(spectra: np.ndarray, h: int = 2, w: int = 2) -> np.ndarray:
    """[n_spectra, B] rows tiled into an [H, W, B] float32 cube."""
    b = spectra.shape[-1]
    return np.broadcast_to(spectra.astype(np.float32), (h, w, b)).copy()


def test_identical_cubes_give_near_zero_angle():
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(4, 3, 5)).astype(np.float32)
    ang = spectral_angle(cube, cube)
    assert ang.shape == (4, 3)
    assert ang.dtype == np.float32
    assert np.allclose(ang, 0.0, atol=1e-6)


def test_invariant_to_uniform_brightness_scaling():
    rng = np.random.default_rng(1)
    t1 = rng.uniform(0.1, 10.0, size=(3, 3, 6)).astype(np.float32)
    t2 = (2.5 * t1).astype(np.float32)
    ang = spectral_angle(t1, t2)
    # float64 accumulation keeps the scaled dot product collinear to ~1e-6 rad
    assert np.allclose(ang, 0.0, atol=1e-5)


def test_orthogonal_spectra_give_pi_over_two():
    spectra = np.array([[1.0, 0.0], [0.0, 1.0]])
    t1 = _cube(spectra[:1])
    t2 = _cube(spectra[1:])
    ang = spectral_angle(t1, t2)
    assert np.allclose(ang, np.pi / 2, atol=1e-6)


def test_opposite_spectra_give_pi():
    t1 = _cube(np.array([[1.0, 1.0]]))
    t2 = _cube(np.array([[-1.0, -1.0]]))
    ang = spectral_angle(t1, t2)
    assert np.allclose(ang, np.pi, atol=1e-6)


def test_nan_propagates_positionally():
    t1 = _cube(np.array([[1.0, 0.0]]), h=2, w=3)
    t2 = _cube(np.array([[0.0, 1.0]]), h=2, w=3)
    t1[1, 2, 0] = np.nan          # NaN in one band of one pixel, epoch 1
    t2[0, 1, 1] = np.nan          # different pixel, epoch 2
    ang = spectral_angle(t1, t2)
    assert np.isnan(ang[1, 2])
    assert np.isnan(ang[0, 1])
    assert np.isfinite(ang).sum() == 4


def test_all_zero_pixel_gives_nan():
    t1 = _cube(np.array([[1.0, 1.0]]), h=2, w=2)
    t2 = _cube(np.array([[1.0, 1.0]]), h=2, w=2)
    t2[0, 0] = 0.0                # zero-norm pixel -> undefined angle
    ang = spectral_angle(t1, t2)
    assert np.isnan(ang[0, 0])
    assert np.allclose(ang[1, 1], 0.0, atol=1e-6)


def test_shape_mismatch_raises():
    t1 = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        spectral_angle(t1, np.zeros((2, 2, 4), dtype=np.float32))
