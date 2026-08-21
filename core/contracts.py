"""Frozen contracts C1-C6 (PLAN.md §2). No branch may redefine these locally.

Every validator raises ContractViolation naming the offending field. Called at
every module boundary in debug mode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import affine
import numpy as np
import rasterio
import rasterio.crs


class ContractViolation(Exception):
    """Raised by every validate_* function. Names the offending field."""


# Closed enum (C1). "hydice_urban" (no _anomaly) is REJECTED on purpose -- that
# name belongs to the Copperas Cove unmixing scene this project must never
# score against (D13.3).
VALID_SOURCES = frozenset({
    "indian_pines", "abu", "hydice_urban_anomaly",
    "had100", "enmap", "sentinel2", "aviris",
})
VALID_GEOREF = frozenset({"real", "synthetic"})
VALID_SOURCE_BRANCH = frozenset({"anomaly", "change", "fused"})
VALID_TARGET_PROFILE = frozenset({"object", "landcover"})


# --- C1 — preprocessed scene ------------------------------------------------

@dataclass(frozen=True)
class SceneMeta:
    scene_id: str                      # unique, stable, e.g. "abu_airport_1"
    crs: rasterio.crs.CRS              # native scene CRS
    transform: affine.Affine           # native pixel -> world
    wavelengths: np.ndarray | None     # [B] float32 nm, strictly ascending when present
                                        # None for sources that ship no wavelengths
                                        # (Indian Pines / ABU / HYDICE -- D13.1/D13.2/D13.3)
    bad_bands: np.ndarray              # [B] bool, True = excluded
    gsd_m: float                       # ground sample distance, metres
    source: str                        # closed enum, see VALID_SOURCES
    georef: str                        # "real" | "synthetic"   (D2)
    acquired: str | None = None        # ISO-8601 or None


def validate_scene(cube: np.ndarray, meta: SceneMeta) -> None:
    if cube.dtype != np.float32:
        raise ContractViolation(f"cube.dtype must be float32, got {cube.dtype}")
    if cube.ndim != 3:
        raise ContractViolation(f"cube.ndim must be 3 [H,W,B], got {cube.ndim}")
    if not cube.flags["C_CONTIGUOUS"]:
        raise ContractViolation("cube must be C-contiguous")

    _, _, b = cube.shape
    if meta.bad_bands.shape != (b,):
        raise ContractViolation(
            f"meta.bad_bands shape {meta.bad_bands.shape} != cube band count ({b},)")
    if meta.bad_bands.dtype != np.bool_:
        raise ContractViolation(f"meta.bad_bands dtype must be bool, got {meta.bad_bands.dtype}")

    if meta.wavelengths is not None:
        if meta.wavelengths.shape != (b,):
            raise ContractViolation(
                f"meta.wavelengths shape {meta.wavelengths.shape} != cube band count ({b},)")
        if not np.all(np.diff(meta.wavelengths) > 0):
            raise ContractViolation("meta.wavelengths must be strictly ascending")

    if meta.source not in VALID_SOURCES:
        raise ContractViolation(f"meta.source {meta.source!r} not in {sorted(VALID_SOURCES)}")
    if meta.georef not in VALID_GEOREF:
        raise ContractViolation(f"meta.georef {meta.georef!r} not in {sorted(VALID_GEOREF)}")

    # Nodata must be NaN, never a sentinel. +/-inf is the one sentinel shape a
    # generic check can catch -- a numeric sentinel (e.g. -9999) is indistinguishable
    # from real data without per-source knowledge, and is caught at load time instead.
    if np.any(np.isinf(cube)):
        raise ContractViolation("cube contains +/-inf; nodata must be NaN, not a sentinel")


# --- C2 — anomaly / change score raster -------------------------------------

REQUIRED_NORM_TAGS = frozenset({
    "NORM_METHOD", "NORM_P_LO", "NORM_P_HI", "NORM_V_LO", "NORM_V_HI",
    "SCORE_METHOD", "SCENE_ID", "GEOREF",
})


def validate_score_raster(path: str | Path) -> None:
    path = Path(path)
    with rasterio.open(path) as ds:
        if ds.count != 1:
            raise ContractViolation(f"{path}: expected 1 band, got {ds.count}")
        if ds.dtypes[0] != "float32":
            raise ContractViolation(f"{path}: expected float32, got {ds.dtypes[0]}")
        if path.name.endswith("_norm.tif"):
            tags = ds.tags()
            missing = REQUIRED_NORM_TAGS - tags.keys()
            if missing:
                raise ContractViolation(f"{path}: missing required tags {sorted(missing)}")
            arr = ds.read(1)
            valid = arr[~np.isnan(arr)]
            if valid.size and (valid.min() < 0.0 or valid.max() > 1.0):
                raise ContractViolation(
                    f"{path}: _norm raster values must be in [0,1], "
                    f"got [{valid.min()}, {valid.max()}]")


# --- C3 — mask ---------------------------------------------------------------

def validate_mask(mask: np.ndarray) -> None:
    if mask.dtype != np.uint8:
        raise ContractViolation(f"mask.dtype must be uint8, got {mask.dtype}")
    vals = set(np.unique(mask).tolist())
    if not vals <= {0, 1}:
        raise ContractViolation(f"mask values must be a subset of {{0,1}}, got {vals}")


# --- C5 — ROI record (in-memory, Stage 4 -> Stage 6) ------------------------

@dataclass
class ROIRecord:
    roi_id: str                        # "{scene_id}:{branch}:{index:04d}"
    source_branch: str                 # "anomaly" | "change" | "fused"        (D5)
    target_profile: str                # "object" | "landcover"                (D5)
    bbox: tuple[int, int, int, int]    # row0, col0, row1, col1 (pixel, t1 grid)
    mask: np.ndarray                   # uint8 [h, w], C3 convention, bbox-local
    anomaly_score: float | None = None
    change_score: float | None = None
    seg_prob: float | None = None
    clear_fraction: float | None = None
    shape_plausibility: float | None = None
    linked_roi_ids: list[str] = field(default_factory=list)
    parent_roi_ids: list[str] = field(default_factory=list)


def validate_roi(roi: ROIRecord) -> None:
    if roi.source_branch not in VALID_SOURCE_BRANCH:
        raise ContractViolation(
            f"roi.source_branch {roi.source_branch!r} not in {sorted(VALID_SOURCE_BRANCH)}")
    if roi.target_profile not in VALID_TARGET_PROFILE:
        raise ContractViolation(
            f"roi.target_profile {roi.target_profile!r} not in {sorted(VALID_TARGET_PROFILE)}")

    validate_mask(roi.mask)

    r0, c0, r1, c1 = roi.bbox
    if not (r1 > r0 and c1 > c0):
        raise ContractViolation(f"roi.bbox for {roi.roi_id!r} is degenerate: {roi.bbox}")
    if roi.mask.shape != (r1 - r0, c1 - c0):
        raise ContractViolation(
            f"roi.mask shape {roi.mask.shape} != bbox-implied shape "
            f"{(r1 - r0, c1 - c0)} for {roi.roi_id!r}")

    parts = roi.roi_id.split(":")
    if len(parts) != 3 or parts[1] != roi.source_branch:
        raise ContractViolation(
            f"roi_id {roi.roi_id!r} does not encode source_branch {roi.source_branch!r}")


# --- C6 — GeoJSON output -----------------------------------------------------

# Roadmap §2 (10) + D5 amendment (6) = 16, exactly.
C6_PROPERTIES = frozenset({
    "lat", "lon", "area", "perimeter", "anomaly_score", "change_score",
    "confidence", "timestamp", "source_scene", "class",
    "roi_id", "source_branch", "target_profile", "linked_roi_ids",
    "confidence_components", "georef",
})

_C6_TYPES: dict[str, tuple[type, ...]] = {
    "lat": (int, float), "lon": (int, float),
    "area": (int, float), "perimeter": (int, float), "confidence": (int, float),
    "anomaly_score": (int, float, type(None)),
    "change_score": (int, float, type(None)),
    "timestamp": (str,), "source_scene": (str,), "class": (str,),
    "roi_id": (str,), "source_branch": (str,), "target_profile": (str,),
    "linked_roi_ids": (list,), "confidence_components": (list,), "georef": (str,),
}


def validate_geojson(path: str | Path) -> None:
    path = Path(path)
    data = json.loads(path.read_text())

    if data.get("type") != "FeatureCollection":
        raise ContractViolation(f"{path}: top-level type must be FeatureCollection")

    for i, feat in enumerate(data.get("features", [])):
        if feat.get("type") != "Feature":
            raise ContractViolation(f"{path}: feature {i} type != 'Feature'")
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            raise ContractViolation(f"{path}: feature {i} geometry.type must be 'Polygon'")

        props = feat.get("properties", {})
        missing = C6_PROPERTIES - props.keys()
        if missing:
            raise ContractViolation(f"{path}: feature {i} missing properties {sorted(missing)}")
        extra = props.keys() - C6_PROPERTIES
        if extra:
            raise ContractViolation(f"{path}: feature {i} has unexpected properties {sorted(extra)}")

        for field_name, allowed_types in _C6_TYPES.items():
            if not isinstance(props[field_name], allowed_types):
                raise ContractViolation(
                    f"{path}: feature {i} property {field_name!r} has wrong type "
                    f"{type(props[field_name]).__name__}, expected one of {allowed_types}")

        if props["source_branch"] not in VALID_SOURCE_BRANCH:
            raise ContractViolation(
                f"{path}: feature {i} source_branch {props['source_branch']!r} invalid")
        if props["target_profile"] not in VALID_TARGET_PROFILE:
            raise ContractViolation(
                f"{path}: feature {i} target_profile {props['target_profile']!r} invalid")
        if props["georef"] not in VALID_GEOREF:
            raise ContractViolation(f"{path}: feature {i} georef {props['georef']!r} invalid")
