"""PLAN.md §2.5 / §3B.7. Phase 2 used morphology only; shape_plausibility and
filter_rois below are the Phase 3B profile-driven post-filter (D6).

morphological_cleanup and connected_components are used by
pipeline/run_pipeline.py RIGHT NOW -- everything below them in this file is
purely additive and does not change their behaviour or signatures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

from core.contracts import ROIRecord


def morphological_cleanup(mask: np.ndarray, *, open_radius: int = 1,
                           close_radius: int = 2) -> np.ndarray:
    out = mask.astype(bool)
    if open_radius > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, open_radius)
        out = ndimage.binary_opening(out, structure=struct)
    if close_radius > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, close_radius)
        out = ndimage.binary_closing(out, structure=struct)
    return out.astype(np.uint8)


def connected_components(mask: np.ndarray, *, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """connectivity: 8 (skimage connectivity=2) or 4 (skimage connectivity=1)."""
    conn = 2 if connectivity == 8 else 1
    labels, n = label(mask.astype(bool), connectivity=conn, return_num=True)
    return labels.astype(np.int32), n


# --- §3B.7 -- profile-driven post-filter (D6), everything below is new ------

ROOT = Path(__file__).resolve().parents[1]
TARGET_PROFILE_PATH = ROOT / "configs" / "target_profile.yaml"
CASCADE_AUDIT_DIR = ROOT / "experiments" / "cascade_recall_audit"


def load_target_profile(name: str, *, config_path: Path = TARGET_PROFILE_PATH) -> dict:
    """Reads `postfilter` thresholds for `name` ("object" | "landcover")
    straight out of configs/target_profile.yaml -- D6 says both profiles are
    maintained and config-selected; thresholds are never hardcoded here.
    Returns a fresh dict copy so a caller mutating it (resolve_profile below)
    never corrupts a value some other caller already read.
    """
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    if name not in cfg or "postfilter" not in cfg.get(name, {}):
        valid = sorted(k for k in cfg if k != "default")
        raise ValueError(f"unknown target profile {name!r}, expected one of {valid}")
    return dict(cfg[name]["postfilter"])


def resolve_profile(profile: str | dict, *, scene_px: int,
                     config_path: Path = TARGET_PROFILE_PATH) -> dict:
    """Materializes the ACTIVE profile's thresholds for ONE SPECIFIC SCENE
    (plan.md §3B.7 sanity constraint).

    `filter_rois`/`shape_plausibility` take a resolved profile dict, not a
    profile name, because `max_area_px` is not a constant: the `object`
    profile's `max_area_px = 2000` sits against 64x64 HAD100 patches
    (4096 px total) -- a target may legitimately occupy half a patch, so the
    effective ceiling is `min(max_area_px, 0.5 * scene_px)`, computed HERE,
    per scene, not baked into the YAML as one number. `ROIRecord` carries no
    scene extent (its `bbox` is scene-relative but says nothing about the
    scene's own shape), so this cannot live inside `filter_rois` itself --
    the caller resolves once per scene, before filtering that scene's ROIs.

    `profile` may be a profile name (looked up via load_target_profile) or
    an already-loaded profile dict (e.g. a synthetic one built in a test) --
    accepting both means the scene-scaling logic isn't duplicated wherever a
    caller already has a profile dict in hand.
    """
    base = load_target_profile(profile, config_path=config_path) if isinstance(profile, str) \
        else dict(profile)
    max_area = base.get("max_area_px")
    if max_area is not None:
        base["max_area_px"] = min(max_area, 0.5 * scene_px)
    return base


def _largest_region_props(mask: np.ndarray):
    """The regionprops of the largest connected component of `mask`, or None
    if `mask` has no foreground pixels at all. A ROIRecord.mask is normally
    already a single connected blob (it comes out of connected_components /
    mask_to_rois), but morphological cleanup upstream can in principle leave
    slivers; scoring the largest piece is the conservative, defensible
    choice -- it is the piece that would drive a human's read of the shape.
    """
    labeled = label(mask.astype(bool), connectivity=2)
    props = regionprops(labeled)
    if not props:
        return None
    return max(props, key=lambda p: p.area)


def _evaluate_shape(props, profile: dict) -> dict:
    """Single source of truth for the D6 gate: area / solidity / elongation
    against `profile`'s min_area_px / max_area_px / min_solidity /
    max_elongation. Used by both shape_plausibility (score) and filter_rois
    (score + WHY, for the audit trail).

    `margin` sign convention: always >= 0, magnitude = how far past the
    violated bound, in that criterion's own units (px for area, solidity
    units for solidity, ratio units for elongation). `margin == 0.0` and
    `failing_criterion is None` together mean every gate passed.
    """
    area = float(props.area)
    solidity = float(props.solidity)
    major = float(props.axis_major_length)
    minor = float(props.axis_minor_length)
    elongation = (major / minor) if minor > 1e-9 else math.inf

    # Defensive: regionprops has not been observed to return non-finite
    # solidity even for degenerate (1px / 1px-wide-line) shapes -- verified
    # empirically -- but `nan < min_solidity` is False, which would let a
    # NaN sail through the gate as "plausible". Route it to failure instead.
    if not math.isfinite(solidity):
        solidity = -math.inf

    min_area = base_min_area = profile.get("min_area_px") or 0
    max_area = profile.get("max_area_px")
    min_solidity = profile.get("min_solidity") or 0
    max_elongation = profile.get("max_elongation")
    max_elongation = math.inf if max_elongation is None else float(max_elongation)

    checks: list[tuple[str, bool, float]] = [
        ("area_below_min", area < min_area, min_area - area),
        ("area_above_max", max_area is not None and area > max_area,
         (area - max_area) if max_area is not None else 0.0),
        ("solidity_below_min", solidity < min_solidity, min_solidity - solidity),
        ("elongation_above_max", elongation > max_elongation,
         (elongation - max_elongation) if math.isfinite(max_elongation) else float(elongation)),
    ]
    failing_criterion = None
    margin = 0.0
    for name, is_bad, m in checks:
        if is_bad:
            failing_criterion = name
            margin = float(m)
            break

    return dict(
        area=area, solidity=solidity, major_axis_length=major, minor_axis_length=minor,
        elongation=(elongation if math.isfinite(elongation) else None),
        min_area_px=base_min_area, max_area_px=max_area, min_solidity=min_solidity,
        max_elongation=(max_elongation if math.isfinite(max_elongation) else None),
        failing_criterion=failing_criterion, margin=margin,
    )


def _continuous_score(metrics: dict) -> float:
    """[0,1] score for a shape that already PASSED every hard gate in
    `_evaluate_shape` (callers only reach this once failing_criterion is
    None). `profile` only specifies bounds for area, not a preferred value
    within them -- an early design scored area against the midpoint of
    [min_area_px, max_area_px], but for the `object` profile that midpoint
    is ~1000 px, which would score a perfectly ordinary 20-30 px target
    near 0 for no principled reason. Area therefore contributes only the
    hard pass/fail gate; solidity and elongation are the graded signal,
    combined by geometric mean so neither alone can carry a bad shape to a
    high score.
    """
    solidity_score = float(np.clip(metrics["solidity"], 0.0, 1.0))

    max_elongation = metrics["max_elongation"]
    elongation = metrics["elongation"] if metrics["elongation"] is not None else math.inf
    if max_elongation is not None and max_elongation > 1.0 and math.isfinite(elongation):
        elongation_score = 1.0 - (elongation - 1.0) / (max_elongation - 1.0)
    else:
        elongation_score = 1.0 if math.isfinite(elongation) else 0.0
    elongation_score = float(np.clip(elongation_score, 0.0, 1.0))

    score = math.sqrt(max(solidity_score, 1e-6) * max(elongation_score, 1e-6))
    return float(np.clip(score, 1e-6, 1.0))


def shape_plausibility(props, profile: dict) -> float:
    """[0,1] from area / solidity / elongation against the ACTIVE profile.
    0.0 = implausible -> filtered. Feeds c_shape in the D4 confidence.

    `props` is a skimage.measure.regionprops RegionProperties object (or
    anything duck-typing .area/.solidity/.axis_major_length/
    .axis_minor_length) for ONE shape -- see `_largest_region_props` for how
    filter_rois derives it from a ROIRecord.mask. `profile` is a resolved
    threshold dict as returned by `resolve_profile` (min_area_px,
    max_area_px, min_solidity, max_elongation) -- callers scoring several
    ROIs from the same scene should resolve once and reuse it.
    """
    metrics = _evaluate_shape(props, profile)
    if metrics["failing_criterion"] is not None:
        return 0.0
    return _continuous_score(metrics)


def _write_audit_record(roi: ROIRecord, metrics: dict, audit_dir: Path) -> Path:
    """Persists one dropped ROI's failing criterion and margin to
    experiments/cascade_recall_audit/ (plan.md: "a stage-2 false negative
    caused by the post-filter is traceable rather than invisible" / Phase 5
    Level 1). One JSON file per dropped ROI, named from its roi_id, so a
    later scan of the directory reconstructs exactly which ROIs were
    dropped and why without needing to replay the pipeline.
    """
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    safe_id = roi.roi_id.replace(":", "__").replace("/", "_")
    record = dict(
        stage="postfilter.filter_rois",
        roi_id=roi.roi_id, source_branch=roi.source_branch, target_profile=roi.target_profile,
        bbox=list(roi.bbox),
        area_px=metrics["area"], solidity=metrics["solidity"],
        major_axis_length=metrics["major_axis_length"], minor_axis_length=metrics["minor_axis_length"],
        elongation=metrics["elongation"],   # None means minor_axis_length ~ 0 (degenerate/line shape)
        min_area_px=metrics["min_area_px"], max_area_px=metrics["max_area_px"],
        min_solidity=metrics["min_solidity"], max_elongation=metrics["max_elongation"],
        failing_criterion=metrics["failing_criterion"], margin=metrics["margin"],
    )
    path = audit_dir / f"{safe_id}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def filter_rois(rois: list[ROIRecord], profile: dict, *,
                 audit_dir: Path = CASCADE_AUDIT_DIR) -> tuple[list[ROIRecord], list[ROIRecord]]:
    """Returns (kept, dropped). Dropped ROIs are NOT discarded -- they are
    written to experiments/cascade_recall_audit/ so a stage-2 false negative
    caused by the post-filter is traceable rather than invisible.

    `profile` must already be scene-resolved (see `resolve_profile`) -- this
    function has no scene extent to work with (ROIRecord carries none), so
    it cannot compute the per-scene `max_area_px` cap itself; that is why
    `resolve_profile` exists as a separate step the caller runs once per
    scene before calling this. Every input ROI's `shape_plausibility` field
    is set (kept or dropped alike) as a side effect, matching how
    pipeline/run_pipeline.py already mutates `roi.anomaly_score` in place.
    """
    kept: list[ROIRecord] = []
    dropped: list[ROIRecord] = []
    for roi in rois:
        props = _largest_region_props(roi.mask)
        if props is None:
            min_area = profile.get("min_area_px") or 0
            metrics = dict(
                area=0.0, solidity=0.0, major_axis_length=0.0, minor_axis_length=0.0,
                elongation=None, min_area_px=min_area, max_area_px=profile.get("max_area_px"),
                min_solidity=profile.get("min_solidity") or 0,
                max_elongation=profile.get("max_elongation"),
                failing_criterion="empty_mask", margin=float(min_area),
            )
        else:
            metrics = _evaluate_shape(props, profile)

        roi.shape_plausibility = 0.0 if metrics["failing_criterion"] is not None \
            else _continuous_score(metrics)

        if metrics["failing_criterion"] is None:
            kept.append(roi)
        else:
            dropped.append(roi)
            _write_audit_record(roi, metrics, audit_dir)

    return kept, dropped
