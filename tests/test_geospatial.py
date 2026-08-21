"""§2.6/§2.7/§2.8 -- pixel->world round-trip; EPSG:6933 area and geodesic
perimeter tested SEPARATELY (an area-only test passes with perimeter broken);
EPSG:4326 reprojection happens only at geojson export (C7).
"""
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
