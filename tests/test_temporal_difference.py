"""Tests for change_detection/temporal_difference.py (PLAN.md §3C.3)."""
from __future__ import annotations

import numpy as np
import pytest

from change_detection.temporal_difference import magnitude_difference


def test_l2_matches_direct_numpy():
    rng = np.random.default_rng(2)
    t1 = rng.normal(size=(3, 4, 5)).astype(np.float32)
    t2 = rng.normal(size=(3, 4, 5)).astype(np.float32)
    out = magnitude_difference(t1, t2, norm="l2")
    ref = np.linalg.norm((t2 - t1).astype(np.float64), axis=-1)
    assert out.shape == (3, 4)
    assert out.dtype == np.float32
    assert np.allclose(out, ref, atol=1e-5)


def test_l1_matches_direct_numpy():
    t1 = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t2 = (t1 * 0.5 - 3.0).astype(np.float32)
    out = magnitude_difference(t1, t2, norm="l1")
    ref = np.abs(t2.astype(np.float64) - t1.astype(np.float64)).sum(axis=-1)
    assert np.allclose(out, ref, atol=1e-4)


def test_l2_default():
    t1 = np.zeros((1, 1, 3), dtype=np.float32)
    t2 = np.array([[[3.0, 4.0, 0.0]]], dtype=np.float32)
    assert magnitude_difference(t1, t2)[0, 0] == pytest.approx(5.0, abs=1e-5)


def test_unknown_norm_raises():
    t = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        magnitude_difference(t, t, norm="linf")


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        magnitude_difference(np.zeros((2, 2, 3), dtype=np.float32),
                             np.zeros((2, 3, 3), dtype=np.float32))


def test_nan_propagates_positionally():
    t1 = np.ones((2, 2, 3), dtype=np.float32)
    t2 = np.zeros((2, 2, 3), dtype=np.float32)
    t1[0, 1, 1] = np.nan          # one NaN band poisons only that pixel
    t2[1, 0, 2] = np.nan
    out = magnitude_difference(t1, t2)
    assert np.isnan(out[0, 1])
    assert np.isnan(out[1, 0])
    # untouched pixels are sqrt(3) away (l2 of the all-ones vs all-zeros diff)
    assert np.allclose(out[[0, 1], [0, 1]], np.sqrt(3.0))
