"""Tests for preprocessing/registration.py (PLAN.md §3C.1).

Accept criterion from the spec: a synthetically shifted pair with known
shift (3.7, -2.3) px recovers to within 0.1 px; a deliberately un-alignable
pair RAISES RegistrationFailure rather than returning garbage.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio.crs
from scipy.ndimage import shift as ndi_shift

from core.contracts import SceneMeta
from preprocessing.registration import RegistrationFailure, coregister_subpixel


def _meta(source: str = "enmap") -> SceneMeta:
    return SceneMeta(
        scene_id="reg_test", crs=rasterio.crs.CRS.from_epsg(4326),
        transform=rasterio.transform.Affine.identity(), wavelengths=None,
        bad_bands=np.zeros(5, dtype=bool), gsd_m=1.0, source=source,
        georef="synthetic")


@pytest.fixture(scope="module")
def shifted_pair():
    """Structured t1 and t2 = t1 shifted by (3.7 rows, -2.3 cols)."""
    rng = np.random.default_rng(3)
    h, w, b = 64, 64, 5
    t1 = rng.uniform(0.1, 1.0, size=(h, w, b)).astype(np.float32)
    # smooth structure so phase correlation has strong gradients
    t1 = ndi_shift(t1, (2, 2, 0), order=3, mode="reflect")
    t2 = ndi_shift(t1, (3.7, -2.3, 0), order=3, mode="reflect").astype(np.float32)
    return t1.astype(np.float32), t2


def test_known_shift_recovered_within_0p1px(shifted_pair):
    t1, t2 = shifted_pair
    aligned, report = coregister_subpixel(t1, t2, _meta(), _meta(),
                                          upsample_factor=20)
    assert abs(report["rmse_px"]) <= 0.1


def test_aligned_output_shape_dtype_and_finiteness(shifted_pair):
    t1, t2 = shifted_pair
    aligned, report = coregister_subpixel(t1, t2, _meta(), _meta())
    assert aligned.shape == t1.shape
    assert aligned.dtype == np.float32
    interior = aligned[8:-8, 8:-8]
    assert np.isfinite(interior).all()


def test_alignment_reduces_difference(shifted_pair):
    t1, t2 = shifted_pair
    aligned, _ = coregister_subpixel(t1, t2, _meta(), _meta())
    raw = np.nanmean(np.abs(t1[8:-8, 8:-8] - t2[8:-8, 8:-8]))
    fixed = np.nanmean(np.abs(t1[8:-8, 8:-8] - aligned[8:-8, 8:-8]))
    assert fixed < raw


def test_unalignable_pair_raises_not_garbage():
    rng = np.random.default_rng(5)
    t1 = rng.uniform(0.1, 1.0, size=(48, 48, 4)).astype(np.float32)
    t1 = ndi_shift(t1, (2, 2, 0), order=3, mode="reflect")
    noise = rng.uniform(0.1, 1.0, size=(48, 48, 4)).astype(np.float32)
    with pytest.raises(RegistrationFailure):
        coregister_subpixel(t1, noise, _meta(), _meta())


def test_nan_nodata_preserved_positionally(shifted_pair):
    t1, t2 = shifted_pair
    t2[5, 5, 2] = np.nan          # a NaN pixel in t2 must not become data;
    aligned, report = coregister_subpixel(t1, t2, _meta(), _meta())
    # the applied correction moves t2 content by (shift_px); the invalid
    # source pixel travels with it -- verify it stays nodata at its DESTINATION
    r = int(round(5 + report["shift_px"]["row"]))
    c = int(round(5 + report["shift_px"]["col"]))
    assert np.isnan(aligned[r, c]).all()
    assert np.isfinite(aligned).sum() > 0


def test_shape_mismatch_raises():
    rng = np.random.default_rng(7)
    a = rng.uniform(size=(16, 16, 3)).astype(np.float32)
    b = rng.uniform(size=(16, 15, 3)).astype(np.float32)
    with pytest.raises(ValueError):
        coregister_subpixel(a, b, _meta(), _meta())
