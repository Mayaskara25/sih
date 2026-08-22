"""Plan §3C.7 -- cloud_shadow_mask: SCL mapping, spectral thresholds,
NaN policy and contract validation.
"""
from __future__ import annotations

import affine
import numpy as np
import pytest
import rasterio.crs

from core.contracts import SceneMeta, validate_mask
from preprocessing.cloud_mask import cloud_shadow_mask


# 450..1650 nm @ 10 nm step -- contains exact 450/1600 bands; nearest to
# red_nm=665 is 660, nearest to nir_nm=842 is 840 (both well within any
# hyperspectral sensor's sampling).
_WL = np.arange(450.0, 1651.0, 10.0, dtype=np.float32)


def _meta(wavelengths=_WL, *, source="enmap", scene_id="test_scene"):
    b = 0 if wavelengths is None else len(wavelengths)
    return SceneMeta(
        scene_id=scene_id,
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=affine.Affine.identity(),
        wavelengths=wavelengths,
        bad_bands=np.zeros(b, dtype=bool),
        gsd_m=30.0,
        source=source,
        georef="synthetic",
    )


def _cube(spectral_values):
    """[H,W,B] float32 cube where every pixel gets one of the named spectra."""
    spectra = {
        "cloud":   {"blue": 0.90, "red": 0.70, "nir": 0.75, "swir": 0.05},
        "veget":   {"blue": 0.05, "red": 0.05, "nir": 0.45, "swir": 0.15},
        "soil":    {"blue": 0.15, "red": 0.25, "nir": 0.30, "swir": 0.35},
    }

    def fill(spec):
        band = np.full(len(_WL), spec["blue"], dtype=np.float32)
        band[int(np.argmin(np.abs(_WL.astype(np.float64) - 660)))] = spec["red"]
        band[int(np.argmin(np.abs(_WL.astype(np.float64) - 840)))] = spec["nir"]
        band[int(np.argmin(np.abs(_WL.astype(np.float64) - 1600)))] = spec["swir"]
        return band

    h = w = len(spectral_values)
    cube = np.zeros((h, w, len(_WL)), dtype=np.float32)
    for i, name in enumerate(spectral_values):
        cube[i, :] = fill(spectra[name])
    return cube


def test_spectral_flags_cloud_and_clears_vegetation():
    # one cloud row above one vegetation row
    names = ["cloud", "veget"]
    mask = cloud_shadow_mask(_cube(names), _meta())
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask[0], np.ones(2, dtype=np.uint8))
    np.testing.assert_array_equal(mask[1], np.zeros(2, dtype=np.uint8))


def test_spectral_soil_not_cloud():
    # bright-ish but high SWIR / moderate NDVI -> clear
    mask = cloud_shadow_mask(_cube(["soil"] * 4), _meta())
    assert not mask.any()


def test_missing_wavelengths_raises_naming_the_problem():
    meta = _meta(wavelengths=None, source="abu")
    with pytest.raises(ValueError, match="wavelengths"):
        cloud_shadow_mask(_cube(["cloud"] * 2), meta)


def test_sentinel2_without_scl_raises():
    with pytest.raises(ValueError, match="SCL"):
        cloud_shadow_mask(
            _cube(["cloud"] * 2), _meta(source="sentinel2"), method="spectral")


def test_auto_sentinel2_without_scl_also_raises():
    with pytest.raises(ValueError, match="SCL"):
        cloud_shadow_mask(_cube(["cloud"] * 2), _meta(source="sentinel2"))


def test_scl_class_mapping_correct():
    scl = np.array([[3, 8, 0], [9, 10, 1], [4, 2, 11]], dtype=np.uint8)
    meta = _meta(source="sentinel2")
    mask = cloud_shadow_mask(_cube(["veget"] * 3), meta, scl=scl)
    expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_output_passes_validate_mask_all_paths():
    validate_mask(cloud_shadow_mask(_cube(["cloud", "veget"]), _meta()))
    validate_mask(cloud_shadow_mask(_cube(["veget"] * 2), _meta(source="sentinel2"),
                                    scl=np.zeros((2, 2), dtype=np.uint8)))


def test_nan_pixels_are_clear_per_documented_policy():
    cube = _cube(["cloud", "veget"])
    cube[0, 0] = np.nan
    mask = cloud_shadow_mask(cube, _meta())
    assert mask[0, 0] == 0          # nodata is NOT flagged as cloud
    assert mask[0, 1] == 1
    assert mask[1, 0] == 0


def test_contradictory_combo_spectral_plus_scl_raises():
    with pytest.raises(ValueError, match="contradictory"):
        cloud_shadow_mask(_cube(["veget"] * 2), _meta(),
                          method="spectral",
                          scl=np.zeros((2, 2), dtype=np.uint8))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        cloud_shadow_mask(_cube(["veget"] * 2), _meta(), method="ndvi")


def test_scl_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        cloud_shadow_mask(_cube(["veget"] * 2), _meta(source="sentinel2"),
                          scl=np.zeros((3, 3), dtype=np.uint8))
