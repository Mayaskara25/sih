"""3D.6 -- the branch's actual claim, instrumented (plan.md §6.4).

`roi_vs_full_comparison` measures, on one scene:
  * full-scene vs ROI-only stage-2 latency,
  * pixels processed at stage 2 vs total,
  * % of pixels discarded by stage-1 screening,
  * bandwidth as full-cube bytes vs transmitted GeoJSON bytes.

ACCEPT CRITERION is NOT defined here. This branch does not own a recall
floor: the threshold comes from `anomaly.scoring.calibrate_threshold_for_
recall` at §4.2's calibrated `target_recall = 0.98`, and the <10%-pixels
criterion is only counted as MET if that recall is achieved. Two recall
numbers in two sections is how a cascade quietly ships at the weaker one.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from anomaly.scoring import calibrate_threshold_for_recall
from core.contracts import SceneMeta
from geospatial.geojson import rois_to_geojson
from geospatial.polygonize import mask_to_rois
from segmentation.infer import N_COMPONENTS, _windows_for_bbox, segment_rois

DEFAULT_TARGET_RECALL = 0.98          # §4.2 -- imported rule, not a local floor
MAX_STAGE2_PIXEL_FRACTION = 0.10      # "<10% of pixels at stage 2"


def _count_stage2_pixels(cube_shape, bboxes, *, patch: int) -> int:
    """Pixels the operational ROI path actually pushes through the model:
    every patch-aligned window `segment_rois` schedules, counted as patch**2
    (the window is what the conv sees), deduplicated exactly like infer does."""
    seen: set[tuple[int, int]] = set()
    h, w = cube_shape[:2]
    for bbox in bboxes:
        for origin in _windows_for_bbox(bbox, patch, (h, w)):
            seen.add(origin)
    return len(seen) * patch * patch


def _count_full_scene_pixels(cube_shape, *, patch: int) -> int:
    """Windows the full-scene baseline would schedule over the entire grid."""
    h, w = cube_shape[:2]
    return -(-h // patch) * -(-w // patch) * patch * patch


def roi_vs_full_comparison(cube: np.ndarray, meta: SceneMeta, gt: np.ndarray,
                           detector, seg_model, *,
                           target_recall: float = DEFAULT_TARGET_RECALL,
                           patch: int = 64, batch: int = 16,
                           transformer=None,
                           n_components: int = N_COMPONENTS,
                           geojson_dir: Path | None = None) -> dict:
    """Run stage 1 (`detector`) full-scene, calibrate the §4.2 threshold against
    ground truth `gt`, then run stage 2 (`seg_model`) ROI-only vs full-scene.

    `detector` : callable `[H,W,B] float32 -> [H,W] float32` (frozen signature).
    Labels are REQUIRED -- this function exists to test a labelled claim, not
    to fabricate one. Returns a SIMULATED-tagged report whose `"accept"` block
    states the criterion as **met / not met**, never as an aspiration.
    """
    import torch

    h, w, b = cube.shape
    total_px = h * w
    report: dict = {"measurement": "SIMULATED", "scene": meta.scene_id,
                    "target_recall": target_recall}

    # --- stage 1 -------------------------------------------------------------
    t0 = time.perf_counter()
    score = np.asarray(detector(cube), dtype=np.float32)
    stage1_latency_s = time.perf_counter() - t0
    report["stage1_detector"] = getattr(detector, "__name__", repr(detector))
    report["stage1_latency_s"] = stage1_latency_s

    valid = ~np.isnan(score) & ~np.isnan(gt.astype(np.float32))
    thr, fp_rate = calibrate_threshold_for_recall(score[valid], gt[valid],
                                                  target_recall=target_recall)
    flagged = np.zeros_like(score, dtype=bool)
    flagged[valid] = score[valid] >= thr
    pos = gt[valid].astype(bool)
    recall_achieved = float((flagged[valid] & pos).sum() / pos.sum()) if pos.sum() else float("nan")

    report.update(threshold=thr, induced_fp_rate=fp_rate, recall_achieved=recall_achieved)

    # --- stage 2, ROI-only (the operational path) ----------------------------
    mask = flagged.astype(np.uint8)
    rois = mask_to_rois(mask, meta, source_branch="anomaly", target_profile="object")
    report["n_rois"] = len(rois)

    bboxes = [roi.bbox for roi in rois]
    roi_px = _count_stage2_pixels(cube.shape, bboxes, patch=patch)
    full_px = _count_full_scene_pixels(cube.shape, patch=patch)

    # Stage-2 MODEL runs happen only when this scene is LEGAL input to the
    # seg model's fitted preprocessing (D15: the transformer is fixed at
    # training time; ABU ships 205 raw bands and no wavelengths, so it can
    # neither reach the 184-band transformer nor harmonize -- plan.md D19
    # suspended exactly this combination). Geometry-only measurement still
    # answers the accept criterion; latency fields degrade to None with a
    # named reason rather than being faked with an illegal refit.
    roi_latency_s = None
    full_latency_s = None
    stage2_notes = []

    if rois:
        try:
            # Probe on one ROI first: a preprocessing incompatibility
            # (e.g. ABU's 205 raw bands vs the 184-band fitted transformer)
            # must surface as a RECORDED note below, not abort the report.
            segment_rois(cube, meta, rois[:1], seg_model, patch=patch, batch=batch,
                         transformer=transformer, n_components=n_components)

            t0 = time.perf_counter()
            segment_rois(cube, meta, rois, seg_model, patch=patch, batch=batch,
                         transformer=transformer, n_components=n_components)
            roi_latency_s = time.perf_counter() - t0

            # --- stage 2, full-scene baseline ------------------------------------
            from segmentation.infer import _extract_window, _prepare_model_input

            def _full_scene_pass() -> None:
                wins = [(r, c)
                        for r in range(0, h, patch)
                        for c in range(0, w, patch)]
                device = next(seg_model.parameters()).device \
                    if hasattr(seg_model, "parameters") else torch.device("cpu")
                seg_model.eval()
                with torch.no_grad():
                    for i in range(0, len(wins), batch):
                        chunk = wins[i:i + batch]
                        arrs = [_prepare_model_input(_extract_window(cube, r, c, patch),
                                                     transformer, n_components)
                                for r, c in chunk]
                        batch_t = torch.from_numpy(np.stack(arrs)).to(device)
                        seg_model(batch_t)

            t0 = time.perf_counter()
            _full_scene_pass()
            full_latency_s = time.perf_counter() - t0
        except ValueError as exc:
            # Band-count mismatch etc. -- record WHY, never silently proceed.
            stage2_notes.append(
                f"stage-2 model runs skipped: {type(exc).__name__}: {str(exc)[:160]}. "
                "Pixel fractions are GEOMETRY-ONLY (window counts); the accept "
                "criterion does not depend on them."
            )
    elif not rois:
        stage2_notes.append("no ROIs survived screening -- nothing for stage 2")

    report["stage2"] = {
        "pixels_total_scene": total_px,
        "pixels_roi_path": roi_px,
        "pixels_full_scene": full_px,
        "fraction_processed_at_stage2": roi_px / total_px,
        "fraction_discarded_by_screening": 1.0 - roi_px / total_px,
        "latency_roi_s": roi_latency_s,
        "latency_full_scene_s": full_latency_s,
        "speedup_x": (full_latency_s / roi_latency_s)
                      if roi_latency_s and full_latency_s else None,
        "notes": stage2_notes,
        "measurement": "SIMULATED",
    }

    # --- bandwidth: full cube vs transmitted GeoJSON --------------------------
    out_dir = Path(geojson_dir) if geojson_dir is not None else \
        Path(tempfile.mkdtemp(prefix="edge_bw_"))
    gj_path = out_dir / f"{meta.scene_id}_roi.geojson"
    if rois:
        rois_to_geojson(rois, meta, gj_path)
        gj_bytes = gj_path.stat().st_size
    else:
        # zero-ROI is a normal outcome; the empty FeatureCollection is still
        # the transmitted product (geospatial.geojson writes it by hand).
        rois_to_geojson(rois, meta, gj_path)
        gj_bytes = gj_path.stat().st_size
    report["bandwidth"] = {
        "full_cube_bytes": int(cube.nbytes),
        "geojson_bytes": int(gj_bytes),
        "ratio_multiple": round(cube.nbytes / gj_bytes, 3) if gj_bytes else None,
        "measurement": "SIMULATED",
    }

    # --- accept criterion: reported as met/not met, never as a target --------
    report["accept"] = {
        "stage2_pixel_fraction_criterion": MAX_STAGE2_PIXEL_FRACTION,
        "stage2_pixel_fraction_met":
            report["stage2"]["fraction_processed_at_stage2"] < MAX_STAGE2_PIXEL_FRACTION,
        "recall_target_source": "plan.md §4.2 via calibrate_threshold_for_recall",
        "recall_met": recall_achieved >= target_recall,
        "criterion_met": bool(
            report["stage2"]["fraction_processed_at_stage2"] < MAX_STAGE2_PIXEL_FRACTION
            and recall_achieved >= target_recall
        ),
        "measurement": "SIMULATED",
    }
    return report