"""§2.6/§2.7/§2.8 -- pixel->world round-trip; EPSG:6933 area and geodesic
perimeter tested SEPARATELY (an area-only test passes with perimeter broken);
EPSG:4326 reprojection happens only at geojson export (C7).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import affine
import numpy as np
import pyproj
import pytest
import rasterio.crs
import shapely.geometry

from core.contracts import SceneMeta
from geospatial.polygonize import mask_to_rois, rois_to_polygons
from geospatial.projections import area_m2, perimeter_m

UTM43N = rasterio.crs.CRS.from_epsg(32643)


def _meta(transform, scene_id="scene_1"):
    return SceneMeta(
        scene_id=scene_id, crs=UTM43N, transform=transform, wavelengths=None,
        bad_bands=np.zeros(1, dtype=bool), gsd_m=abs(transform.a),
        source="had100", georef="real",
    )


def test_polygon_pixel_to_world_roundtrip():
    transform = affine.Affine(10.0, 0, 500_000.0, 0, -10.0, 3_000_000.0)
    meta = _meta(transform)
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:15, 8:18] = 1   # 10x10 square at row0=5, col0=8

    rois = mask_to_rois(mask, meta, source_branch="anomaly", target_profile="object")
    assert len(rois) == 1
    polygons = rois_to_polygons(rois, meta)
    assert len(polygons) == 1

    expected_minx, expected_maxy = transform * (8, 5)
    expected_maxx, expected_miny = transform * (18, 15)
    minx, miny, maxx, maxy = polygons[0].bounds
    assert abs(minx - expected_minx) < 1e-6
    assert abs(maxy - expected_maxy) < 1e-6
    assert abs(maxx - expected_maxx) < 1e-6
    assert abs(miny - expected_miny) < 1e-6


def _square_km_at_latlon(lat: float, lon: float) -> shapely.geometry.Polygon:
    transformer = pyproj.Transformer.from_crs("EPSG:4326", UTM43N, always_xy=True)
    cx, cy = transformer.transform(lon, lat)
    half = 500.0   # metres -> 1km x 1km square
    return shapely.geometry.box(cx - half, cy - half, cx + half, cy + half)


def test_area_m2_at_8_degrees_north():
    square = _square_km_at_latlon(8.0, 77.0)
    assert area_m2(square, UTM43N) == pytest.approx(1_000_000.0, rel=0.005)


def test_area_m2_at_35_degrees_north():
    square = _square_km_at_latlon(35.0, 77.0)
    assert area_m2(square, UTM43N) == pytest.approx(1_000_000.0, rel=0.005)


def test_perimeter_m_at_8_degrees_north():
    square = _square_km_at_latlon(8.0, 77.0)
    assert perimeter_m(square, UTM43N) == pytest.approx(4_000.0, rel=0.005)


def test_perimeter_m_at_35_degrees_north():
    square = _square_km_at_latlon(35.0, 77.0)
    assert perimeter_m(square, UTM43N) == pytest.approx(4_000.0, rel=0.005)


def test_native_polygon_is_not_already_in_degrees():
    """C7: native-CRS polygons must not accidentally already look like lat/lon."""
    square = _square_km_at_latlon(20.0, 77.0)
    minx, miny, maxx, maxy = square.bounds
    assert abs(minx) > 180 or abs(maxx) > 180   # UTM metres, not degrees


# --- D33: C6 `timestamp` is the ACQUISITION time, not the run time ----------
# Regression guard. This defaulted to datetime.now() unconditionally, so every
# GeoJSON carried the run date. It went unnoticed while every source had
# acquired=None, because the wrong value and the only available value were the
# same string; EnMAP is the first source that parses a real <startTime>.

def _meta_acq(acquired, scene_id="scene_ts"):
    return SceneMeta(
        scene_id=scene_id, crs=UTM43N,
        transform=affine.Affine(30.0, 0, 500_000.0, 0, -30.0, 1_000_000.0),
        wavelengths=None, bad_bands=np.zeros(1, dtype=bool), gsd_m=30.0,
        source="enmap", georef="real", acquired=acquired,
    )


def _one_roi_geojson(tmp_path, meta, **kw):
    from geospatial.geojson import rois_to_geojson
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    rois = mask_to_rois(mask, meta, source_branch="anomaly", target_profile="object")
    out = rois_to_geojson(rois, meta, tmp_path / "ts.geojson", **kw)
    return json.loads(Path(out).read_text())["features"]


def test_timestamp_uses_scene_acquisition_time_not_run_time(tmp_path):
    """The bug itself: acquisition 2026-07-24, run date is today."""
    meta = _meta_acq("2026-07-24T05:43:07Z")
    feats = _one_roi_geojson(tmp_path, meta)
    assert feats, "fixture must produce at least one ROI"
    for f in feats:
        assert f["properties"]["timestamp"] == "2026-07-24T05:43:07Z"
        assert not f["properties"]["timestamp"].startswith(
            datetime.now(timezone.utc).strftime("%Y-%m-%d")), "emitted the RUN date"


def test_timestamp_normalizes_offset_and_subsecond_to_utc(tmp_path):
    """EnMAP's <startTime> carries sub-second precision; other sources may
    carry an offset. Both must land in the same C6 format, in UTC."""
    feats = _one_roi_geojson(tmp_path, _meta_acq("2026-07-24T11:13:07.482000+05:30"))
    assert feats[0]["properties"]["timestamp"] == "2026-07-24T05:43:07Z"


def test_timestamp_falls_back_to_now_only_when_source_has_none(tmp_path):
    """`.mat` sources genuinely have no acquisition time; C6 still requires a
    string, so the run date is the correct fallback THERE and only there."""
    feats = _one_roi_geojson(tmp_path, _meta_acq(None))
    ts = feats[0]["properties"]["timestamp"]
    assert ts.startswith(datetime.now(timezone.utc).strftime("%Y-%m-%d"))


def test_malformed_acquired_raises_rather_than_silently_using_run_date(tmp_path):
    """Silently substituting now() would restore the original bug for exactly
    the source most likely to hit it."""
    from geospatial.geojson import rois_to_geojson
    meta = _meta_acq("24/07/2026 05:43")
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    rois = mask_to_rois(mask, meta, source_branch="anomaly", target_profile="object")
    with pytest.raises(ValueError, match="not ISO-8601"):
        rois_to_geojson(rois, meta, tmp_path / "bad.geojson")


def test_explicit_timestamp_argument_still_wins(tmp_path):
    feats = _one_roi_geojson(tmp_path, _meta_acq("2026-07-24T05:43:07Z"),
                             timestamp="1999-01-01T00:00:00Z")
    assert feats[0]["properties"]["timestamp"] == "1999-01-01T00:00:00Z"
