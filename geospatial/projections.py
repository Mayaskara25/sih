"""PLAN.md §2.7."""
from __future__ import annotations

import pyproj
import shapely.geometry
import shapely.ops

_WGS84 = "EPSG:4326"
_EQUAL_AREA = "EPSG:6933"

_geod = pyproj.Geod(ellps="WGS84")


def to_wgs84(geoms: list, src_crs) -> list[shapely.geometry.base.BaseGeometry]:
    transformer = pyproj.Transformer.from_crs(src_crs, _WGS84, always_xy=True)
    return [shapely.ops.transform(transformer.transform, g) for g in geoms]


def area_m2(geom, src_crs) -> float:
    """EPSG:6933 (Lambert cylindrical equal-area). NEVER degrees -- see C6."""
    transformer = pyproj.Transformer.from_crs(src_crs, _EQUAL_AREA, always_xy=True)
    projected = shapely.ops.transform(transformer.transform, geom)
    return abs(projected.area)


def perimeter_m(geom, src_crs) -> float:
    """GEODESIC: pyproj.Geod(ellps="WGS84").geometry_length on the WGS84
    geometry. NOT EPSG:6933 -- that projection preserves area, not length,
    and its scale asymmetry biases perimeter at Indian latitudes (C6).
    """
    wgs84_geom = to_wgs84([geom], src_crs)[0]
    boundary = getattr(wgs84_geom, "exterior", wgs84_geom)
    return abs(_geod.geometry_length(boundary))


def centroid_latlon(geom, src_crs) -> tuple[float, float]:
    wgs84_geom = to_wgs84([geom], src_crs)[0]
    c = wgs84_geom.centroid
    return c.y, c.x
