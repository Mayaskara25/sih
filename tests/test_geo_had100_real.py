"""§3A.1 -- the first REAL georeference check (D11.5), moved forward from
Phase 5 Level 2. HAD100 ENVI headers carry genuine UTM/WGS-84 'map info'.

Genuinely INDEPENDENT, not a re-check of GDAL against itself: this hand-derives
the pixel->world affine straight from the raw 'map info' fields (tie point,
pixel size, rotation) using basic trigonometry, with no rasterio/GDAL/spectral
involved in the derivation. D14.2 already established that HAD100's real
scenes are rotated (33 deg NG, 90 deg Classic) and that a rotation-naive
parse is measurably wrong; this is the independent check that GDAL's
rotation-aware transform (which raster_loader now delegates to) is ITSELF
correct, rather than merely self-consistent.

Then runs the actual Phase 2 pixel -> world -> EPSG:4326 -> GeoJSON path
(geospatial/polygonize.py, geospatial/projections.py) on a known pixel block
of a real scene, and confirms the exported centroid agrees with the
hand-derived coordinate to within one pixel -- the literal §3A.1 accept
criterion.
"""
import re
from pathlib import Path

import affine
import numpy as np
import pyproj
import pytest

from geospatial.polygonize import mask_to_rois, rois_to_polygons
from geospatial.projections import centroid_latlon
from preprocessing.raster_loader import load_scene

ROOT = Path(__file__).resolve().parents[1]
HAD100_NG_HDR = (ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data"
                 / "aviris_ng_normal" / "ang20191004t185054_13.hdr")
HAD100_CLASSIC_HDR = (ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data"
                       / "aviris_normal" / "f170507t01p00r10_1.hdr")
_have_had100 = HAD100_NG_HDR.exists() and HAD100_CLASSIC_HDR.exists()

_MAP_INFO_RE = re.compile(r"map info\s*=\s*\{([^}]*)\}", re.S)
_ROTATION_RE = re.compile(r"rotation\s*=\s*([-\d.]+)")


def _map_info_affine(tie_x: float, tie_y: float, tie_east: float, tie_north: float,
                      dx: float, dy: float, rotation_deg: float) -> affine.Affine:
    """Independent re-derivation of the ENVI 'map info' pixel->world affine,
    including rotation -- no rasterio/GDAL/spectral involved.

    Formula (empirically derived and cross-checked against GDAL's own ENVI
    driver on both a 33deg and a 90deg real header before this was trusted):
        a =  dx * cos(theta)     b =  dy * sin(theta)
        d =  dx * sin(theta)     e = -dy * cos(theta)
    with the tie point (1-based) fixing the constant terms c, f. Both real
    HAD100 headers have tie_x == tie_y == 1.0, which multiplies that part of
    the formula by zero -- see test_tie_point_offset_arithmetic_round_trips
    for the check that exercises it at a non-unity tie point.
    """
    theta = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(theta), np.sin(theta)
    a, b = dx * cos_r, dy * sin_r
    d, e = dx * sin_r, -dy * cos_r
    c = tie_east - (tie_x - 1) * a - (tie_y - 1) * b
    f = tie_north - (tie_x - 1) * d - (tie_y - 1) * e
    return affine.Affine(a, b, c, d, e, f)


def _hand_derived_affine(hdr_path: Path) -> affine.Affine:
    text = hdr_path.read_text(errors="replace")
    parts = [p.strip() for p in _MAP_INFO_RE.search(text).group(1).split(",")]
    tie_x, tie_y, tie_east, tie_north, dx, dy = (float(v) for v in parts[1:7])
    rot_m = _ROTATION_RE.search(text)
    rotation_deg = float(rot_m.group(1)) if rot_m else 0.0
    return _map_info_affine(tie_x, tie_y, tie_east, tie_north, dx, dy, rotation_deg)


@pytest.mark.parametrize("tie_x,tie_y,rotation_deg", [
    (1.0, 1.0, 0.0), (3.0, 2.0, 0.0), (2.5, 4.5, 33.0), (1.0, 1.0, 90.0), (7.0, 1.0, 145.0),
])
def test_tie_point_offset_arithmetic_round_trips(tie_x, tie_y, rotation_deg):
    """Both real HAD100 headers have tie_x == tie_y == 1.0, so the
    `- (tie_x-1)*a - (tie_y-1)*b` constant-term arithmetic is multiplied by
    zero in both and goes untested by test_hand_derived_rotation_affine_matches_gdal.
    This is a property that must hold for ANY tie point regardless: applying
    the resulting affine to the zero-based tie pixel (tie_x-1, tie_y-1) must
    recover the tie point's map coordinate (tie_east, tie_north) exactly,
    by construction of the formula -- independent of any real file.
    """
    tie_east, tie_north, dx, dy = 500_000.0, 4_000_000.0, 8.4, 8.4
    transform = _map_info_affine(tie_x, tie_y, tie_east, tie_north, dx, dy, rotation_deg)
    got_east, got_north = transform * (tie_x - 1, tie_y - 1)
    assert got_east == pytest.approx(tie_east, abs=1e-9)
    assert got_north == pytest.approx(tie_north, abs=1e-9)


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
@pytest.mark.parametrize("hdr_path", [HAD100_NG_HDR, HAD100_CLASSIC_HDR], ids=["ng_33deg", "classic_90deg"])
def test_hand_derived_rotation_affine_matches_gdal(hdr_path):
    """Independent verification that GDAL's ENVI driver -- which
    raster_loader now delegates to (D14.2) -- is itself correct, not merely
    internally consistent."""
    _cube, meta = load_scene(hdr_path, source="had100")
    hand = _hand_derived_affine(hdr_path)

    for got, expected in zip(meta.transform, hand):
        assert got == pytest.approx(expected, abs=1e-3), (meta.transform, hand)


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
@pytest.mark.parametrize("hdr_path", [HAD100_NG_HDR, HAD100_CLASSIC_HDR], ids=["ng_33deg", "classic_90deg"])
def test_pixel_to_wgs84_geojson_centroid_within_one_pixel_of_independent_coordinate(hdr_path):
    """The literal §3A.1 accept criterion: run the real pixel -> world ->
    EPSG:4326 path (§2.6-2.7) on a real HAD100 scene and confirm the ROI
    centroid falls within one pixel of a coordinate computed independently
    (hand-derived affine + pyproj, no shared code path with polygonize.py /
    projections.py beyond pyproj itself)."""
    cube, meta = load_scene(hdr_path, source="had100")
    h, w, _ = cube.shape

    r0, c0 = h // 2, w // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[r0:r0 + 3, c0:c0 + 3] = 1

    rois = mask_to_rois(mask, meta, source_branch="anomaly", target_profile="object")
    assert len(rois) == 1
    polygon = rois_to_polygons(rois, meta)[0]
    lat, lon = centroid_latlon(polygon, meta.crs)

    hand_transform = _hand_derived_affine(hdr_path)
    px_col, px_row = c0 + 1.5, r0 + 1.5   # pixel-space centroid of the 3x3 block
    hand_x, hand_y = hand_transform * (px_col, px_row)
    transformer = pyproj.Transformer.from_crs(meta.crs, "EPSG:4326", always_xy=True)
    hand_lon, hand_lat = transformer.transform(hand_x, hand_y)

    geod = pyproj.Geod(ellps="WGS84")
    _fwd, _back, dist_m = geod.inv(lon, lat, hand_lon, hand_lat)
    assert dist_m < meta.gsd_m, (
        f"centroid {dist_m:.3f} m from the independently-computed coordinate, "
        f"exceeds one pixel ({meta.gsd_m} m)")
