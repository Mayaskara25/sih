"""PLAN.md §2.8."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd

from core.contracts import ROIRecord, SceneMeta
from geospatial.polygonize import rois_to_polygons
from geospatial.projections import area_m2, centroid_latlon, perimeter_m, to_wgs84

# D4 -- weighted mean over available components, weights renormalized when a
# component is missing.
_WEIGHTS = {
    "c_anom": 0.40, "c_change": 0.20, "c_seg": 0.25, "c_clear": 0.10, "c_shape": 0.05,
}


def compute_confidence(roi: ROIRecord) -> tuple[float, list[str]]:
    components = {
        "c_anom": roi.anomaly_score,
        "c_change": roi.change_score,
        "c_seg": roi.seg_prob,
        "c_clear": roi.clear_fraction,
        "c_shape": roi.shape_plausibility,
    }
    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return 0.0, []
    total_weight = sum(_WEIGHTS[k] for k in available)
    confidence = sum(_WEIGHTS[k] * v for k, v in available.items()) / total_weight
    order = list(_WEIGHTS)
    return confidence, sorted(available, key=order.index)


def rois_to_geojson(rois: list[ROIRecord], meta: SceneMeta, out_path: str | Path,
                     *, timestamp: str | None = None) -> Path:
    """The ONLY place EPSG:4326 reprojection happens (C7). Emits every C6
    field including the D5 amendment fields. Computes `confidence` via D4
    over whatever components are non-None, and records which ones in
    `confidence_components`.
    """
    out_path = Path(out_path)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    polygons = rois_to_polygons(rois, meta)
    wgs84_polygons = to_wgs84(polygons, meta.crs)

    records = []
    for roi, native_geom, wgs84_geom in zip(rois, polygons, wgs84_polygons):
        confidence, components = compute_confidence(roi)
        lat, lon = centroid_latlon(native_geom, meta.crs)
        records.append(dict(
            geometry=wgs84_geom,
            lat=lat, lon=lon,
            area=area_m2(native_geom, meta.crs),
            perimeter=perimeter_m(native_geom, meta.crs),
            anomaly_score=roi.anomaly_score,
            change_score=roi.change_score,
            confidence=confidence,
            timestamp=timestamp,
            source_scene=meta.scene_id,
            **{"class": "UNKNOWN"},
            roi_id=roi.roi_id,
            source_branch=roi.source_branch,
            target_profile=roi.target_profile,
            linked_roi_ids=roi.linked_roi_ids,
            confidence_components=components,
            georef=meta.georef,
        ))

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path
