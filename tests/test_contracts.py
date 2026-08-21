"""C1-C6 validators: one passing and one failing case each (§1.5)."""
import json

import affine
import numpy as np
import pytest
import rasterio
import rasterio.crs

from core.contracts import (
    ContractViolation,
    ROIRecord,
    SceneMeta,
    validate_geojson,
    validate_mask,
    validate_roi,
    validate_scene,
    validate_score_raster,
)


def _meta(**overrides):
    defaults = dict(
        scene_id="indian_pines_1",
        crs=rasterio.crs.CRS.from_epsg(32616),
        transform=affine.Affine(20.0, 0, 500_000.0, 0, -20.0, 4_480_000.0),
        wavelengths=None,
        bad_bands=np.zeros(4, dtype=bool),
        gsd_m=20.0,
        source="indian_pines",
        georef="synthetic",
    )
    defaults.update(overrides)
    return SceneMeta(**defaults)


# --- C1 ----------------------------------------------------------------------

def test_validate_scene_passes():
    cube = np.zeros((3, 3, 4), dtype=np.float32)
    validate_scene(cube, _meta())


def test_validate_scene_rejects_wrong_dtype():
    cube = np.zeros((3, 3, 4), dtype=np.float64)
    with pytest.raises(ContractViolation):
        validate_scene(cube, _meta())


def test_validate_scene_rejects_non_ascending_wavelengths():
    cube = np.zeros((3, 3, 3), dtype=np.float32)
    bad = _meta(wavelengths=np.array([500.0, 400.0, 600.0], dtype=np.float32),
                bad_bands=np.zeros(3, dtype=bool))
    with pytest.raises(ContractViolation):
        validate_scene(cube, bad)


def test_validate_scene_rejects_sentinel_inf():
    cube = np.zeros((3, 3, 4), dtype=np.float32)
    cube[0, 0, 0] = np.inf
    with pytest.raises(ContractViolation):
        validate_scene(cube, _meta())


def test_validate_scene_rejects_bad_source_enum():
    cube = np.zeros((3, 3, 4), dtype=np.float32)
    with pytest.raises(ContractViolation):
        validate_scene(cube, _meta(source="hydice_urban"))  # not "_anomaly" -- D13.3


# --- C2 ------------------------------------------------------------------

def test_validate_score_raster_passes(tmp_path):
    path = tmp_path / "scene_anom_norm.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
    ) as ds:
        ds.write(np.full((4, 4), 0.5, dtype=np.float32), 1)
        ds.update_tags(
            NORM_METHOD="percentile_clip", NORM_P_LO="1.0", NORM_P_HI="99.9",
            NORM_V_LO="0.0", NORM_V_HI="1.0", SCORE_METHOD="global_rx",
            SCENE_ID="scene", GEOREF="synthetic",
        )
    validate_score_raster(path)


def test_validate_score_raster_rejects_missing_tags(tmp_path):
    path = tmp_path / "scene_anom_norm.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
    ) as ds:
        ds.write(np.full((4, 4), 0.5, dtype=np.float32), 1)
    with pytest.raises(ContractViolation):
        validate_score_raster(path)


# --- C3 ------------------------------------------------------------------

def test_validate_mask_passes():
    validate_mask(np.array([[0, 1], [1, 0]], dtype=np.uint8))


def test_validate_mask_rejects_non_binary_values():
    with pytest.raises(ContractViolation):
        validate_mask(np.array([[0, 2]], dtype=np.uint8))


# --- C5 ------------------------------------------------------------------

def _roi(**overrides):
    defaults = dict(
        roi_id="scene_1:anomaly:0000",
        source_branch="anomaly",
        target_profile="object",
        bbox=(0, 0, 2, 2),
        mask=np.ones((2, 2), dtype=np.uint8),
    )
    defaults.update(overrides)
    return ROIRecord(**defaults)


def test_validate_roi_passes():
    validate_roi(_roi())


def test_validate_roi_rejects_roi_id_branch_mismatch():
    with pytest.raises(ContractViolation):
        validate_roi(_roi(roi_id="scene_1:change:0000"))  # branch says "anomaly"


# --- C6 ------------------------------------------------------------------

def _feature(**prop_overrides):
    props = dict(
        lat=40.0, lon=-86.5, area=100.0, perimeter=40.0,
        anomaly_score=0.8, change_score=None, confidence=0.8,
        timestamp="2026-08-20T11:04:32Z", source_scene="scene_1", class_="UNKNOWN",
        roi_id="scene_1:anomaly:0000", source_branch="anomaly",
        target_profile="object", linked_roi_ids=[], confidence_components=["c_anom"],
        georef="synthetic",
    )
    props["class"] = props.pop("class_")
    props.update(prop_overrides)
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]},
        "properties": props,
    }


def test_validate_geojson_passes(tmp_path):
    path = tmp_path / "rois.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [_feature()]}))
    validate_geojson(path)


def test_validate_geojson_rejects_missing_field(tmp_path):
    feat = _feature()
    del feat["properties"]["confidence"]
    path = tmp_path / "rois.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feat]}))
    with pytest.raises(ContractViolation):
        validate_geojson(path)
