"""§3B.7 segmentation/postfilter.py -- profile-driven shape post-filter (D6).

morphological_cleanup / connected_components predate this work and are used
RIGHT NOW by pipeline/run_pipeline.py; a couple of small regression tests
here pin their existing behaviour so nothing below them in the file can
quietly change it.

The rest targets shape_plausibility / filter_rois / resolve_profile /
load_target_profile, with two things emphasized per plan.md:

* `max_area_px` is NOT a constant -- resolve_profile computes
  `min(max_area_px, 0.5 * scene_px)` per scene (§3B.7). A test that hardcodes
  one scene size would pass even if `filter_rois` baked in a constant
  ceiling; `test_max_area_px_is_per_scene_not_constant` runs the SAME roi
  through two different scene sizes and requires opposite verdicts.
* Dropped ROIs must be traceable, not silently discarded (Phase 5 L1):
  `test_filter_rois_writes_audit_record_with_failing_criterion_and_margin`
  reads the JSON `filter_rois` writes to `audit_dir` and checks the
  failing criterion and margin are actually recorded.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from skimage.measure import label, regionprops

from core.contracts import ROIRecord
from segmentation.postfilter import (
    connected_components,
    filter_rois,
    load_target_profile,
    morphological_cleanup,
    resolve_profile,
    shape_plausibility,
)


# --- fixtures -----------------------------------------------------------------

def _roi(index: int, mask: np.ndarray, *, target_profile: str = "object",
         scene_id: str = "test_scene") -> ROIRecord:
    h, w = mask.shape
    return ROIRecord(
        roi_id=f"{scene_id}:anomaly:{index:04d}",
        source_branch="anomaly",
        target_profile=target_profile,
        bbox=(0, 0, h, w),
        mask=mask.astype(np.uint8),
    )


def _props(mask: np.ndarray):
    labeled = label(mask.astype(bool), connectivity=2)
    props = regionprops(labeled)
    assert props, "fixture mask has no foreground pixels"
    return max(props, key=lambda p: p.area)


def _square(side: int) -> np.ndarray:
    return np.ones((side, side), dtype=np.uint8)


def _rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def _diagonal_line(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        m[i, i] = 1
    return m


# --- morphological_cleanup / connected_components: unchanged regression -------

def test_morphological_cleanup_opening_removes_isolated_noise_pixel():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2, 2] = 1                 # isolated single-pixel noise
    mask[5:8, 5:8] = 1             # solid 3x3 blob

    out = morphological_cleanup(mask, open_radius=1, close_radius=0)

    assert out[2, 2] == 0, "opening should remove the isolated noise pixel"
    assert out[6, 6] == 1, "opening should not erase the solid blob's core"
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_morphological_cleanup_closing_fills_interior_hole():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    mask[5, 5] = 0                 # single interior pixel, fully surrounded

    out = morphological_cleanup(mask, open_radius=0, close_radius=2)

    assert out[5, 5] == 1, "closing should fill the small interior hole"


def test_connected_components_connectivity_8_vs_4_differ_on_diagonal_touch():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2, 2] = 1                 # touches [1,1] only diagonally

    labels8, n8 = connected_components(mask, connectivity=8)
    labels4, n4 = connected_components(mask, connectivity=4)

    assert n8 == 1, "8-connectivity should merge diagonally-touching pixels"
    assert n4 == 2, "4-connectivity should keep them separate"
    assert labels8.dtype == np.int32
    assert labels8.shape == mask.shape


# --- load_target_profile / resolve_profile ------------------------------------

def test_load_target_profile_reads_configs_target_profile_yaml():
    object_profile = load_target_profile("object")
    assert object_profile == dict(
        min_area_px=4, max_area_px=2000, min_solidity=0.15, max_elongation=8.0)

    landcover_profile = load_target_profile("landcover")
    assert landcover_profile == dict(
        min_area_px=50, max_area_px=None, min_solidity=0.05, max_elongation=20.0)


def test_load_target_profile_rejects_unknown_name():
    with pytest.raises(ValueError):
        load_target_profile("not_a_real_profile")


def test_resolve_profile_accepts_dict_without_hitting_the_yaml_file():
    resolved = resolve_profile(
        dict(min_area_px=1, max_area_px=100, min_solidity=0.1, max_elongation=5.0),
        scene_px=10_000)
    assert resolved["max_area_px"] == 100   # 0.5*10000=5000 > 100, base wins


def test_resolve_profile_landcover_null_max_area_stays_unbounded():
    resolved = resolve_profile("landcover", scene_px=64 * 64)
    assert resolved["max_area_px"] is None


# --- shape_plausibility ---------------------------------------------------------

def test_shape_plausibility_zero_for_implausible_shape_object_profile():
    profile = resolve_profile("object", scene_px=64 * 64)
    props = _props(_diagonal_line(60))   # near-infinite elongation
    assert shape_plausibility(props, profile) == 0.0


def test_shape_plausibility_positive_for_plausible_shape_object_profile():
    profile = resolve_profile("object", scene_px=64 * 64)
    props = _props(_square(20))          # compact, solid, unit elongation
    score = shape_plausibility(props, profile)
    assert score > 0.0
    assert score <= 1.0


def test_shape_plausibility_verdict_depends_on_active_profile():
    """The SAME shape (elongation ~10.2, area 250) must be rejected under the
    object profile (max_elongation=8.0) and accepted under landcover
    (max_elongation=20.0, min_area_px=50) -- proof thresholds actually come
    from the active profile rather than being baked into the function."""
    props = _props(_rect(5, 50))

    object_profile = resolve_profile("object", scene_px=64 * 64)
    landcover_profile = resolve_profile("landcover", scene_px=64 * 64)

    assert shape_plausibility(props, object_profile) == 0.0
    assert shape_plausibility(props, landcover_profile) > 0.0


# --- max_area_px is per-scene, not a constant ----------------------------------

def test_max_area_px_is_per_scene_not_constant(tmp_path):
    """area=800 sits between the two scenes' effective ceilings:
    min(2000, 0.5*32*32)=512 (dropped) vs min(2000, 0.5*120*120)=2000 (kept).
    A hardcoded max_area_px=2000 would keep it in both cases and pass a
    naive test; resolve_profile must actually scale with scene_px."""
    mask = _rect(20, 40)   # area=800, solidity=1.0, elongation~2.0 -- otherwise unremarkable

    small_scene_profile = resolve_profile("object", scene_px=32 * 32)
    large_scene_profile = resolve_profile("object", scene_px=120 * 120)
    assert small_scene_profile["max_area_px"] == 512.0
    assert large_scene_profile["max_area_px"] == 2000

    # Discrepancy vs. plan.md's own worked example: the docstring's two
    # named scene sizes (64x64 -> 4096 scene_px, 120x120 -> 14400 scene_px,
    # "D11.2 raw-scene shapes") do NOT actually demonstrate a per-scene
    # difference -- 0.5*scene_px (2048 and 7200) exceeds the base
    # max_area_px=2000 at BOTH sizes, so min(2000, 0.5*scene_px) resolves to
    # the constant 2000 in both cases. The 0.5*scene_px cap only binds below
    # ~4000 scene_px (roughly 63x63 and smaller) -- it is a safety rail for
    # sub-patch scenes, not a scaling law that differentiates 64x64 from
    # 120x120. Pinned here so the no-op at the plan's own named sizes is
    # recorded in the suite, not just asserted away by picking 32x32 instead.
    assert resolve_profile("object", scene_px=64 * 64)["max_area_px"] == 2000
    assert resolve_profile("object", scene_px=120 * 120)["max_area_px"] == 2000

    roi_small = _roi(0, mask)
    roi_large = _roi(0, mask.copy())

    kept_small, dropped_small = filter_rois(
        [roi_small], small_scene_profile, audit_dir=tmp_path / "audit_small")
    kept_large, dropped_large = filter_rois(
        [roi_large], large_scene_profile, audit_dir=tmp_path / "audit_large")

    assert kept_small == [] and dropped_small == [roi_small], \
        "800px roi should be DROPPED against the 32x32-scene ceiling (512px)"
    assert kept_large == [roi_large] and dropped_large == [], \
        "the SAME roi shape should be KEPT against the 120x120-scene ceiling (2000px)"

    assert roi_small.shape_plausibility == 0.0
    assert roi_large.shape_plausibility is not None and roi_large.shape_plausibility > 0.0


# --- filter_rois: (kept, dropped) partitions the input, nothing lost ----------

def test_filter_rois_partitions_input_with_nothing_lost(tmp_path):
    profile = resolve_profile("object", scene_px=64 * 64)
    rois = [
        _roi(0, _square(20)),          # plausible
        _roi(1, _square(15)),          # plausible
        _roi(2, _diagonal_line(60)),   # implausible: elongation
        _roi(3, np.ones((1, 1), dtype=np.uint8)),   # implausible: area below min (4)
    ]

    kept, dropped = filter_rois(rois, profile, audit_dir=tmp_path / "audit")

    assert len(kept) + len(dropped) == len(rois)
    assert {r.roi_id for r in kept} | {r.roi_id for r in dropped} == {r.roi_id for r in rois}
    assert {r.roi_id for r in kept} & {r.roi_id for r in dropped} == set()

    kept_ids = {r.roi_id for r in kept}
    assert kept_ids == {rois[0].roi_id, rois[1].roi_id}

    for roi in kept:
        assert roi.shape_plausibility is not None and roi.shape_plausibility > 0.0
    for roi in dropped:
        assert roi.shape_plausibility == 0.0


def test_filter_rois_handles_empty_mask_roi_without_crashing(tmp_path):
    """A roi whose mask has no foreground pixels at all (e.g. fully cleaned
    away by morphology upstream) must be dropped, not raise."""
    profile = resolve_profile("object", scene_px=64 * 64)
    roi = _roi(0, np.zeros((5, 5), dtype=np.uint8))

    kept, dropped = filter_rois([roi], profile, audit_dir=tmp_path / "audit")

    assert kept == []
    assert dropped == [roi]
    assert roi.shape_plausibility == 0.0


# --- dropped ROIs are audited: failing criterion + margin recorded ------------

def test_filter_rois_writes_audit_record_with_failing_criterion_and_margin(tmp_path):
    profile = resolve_profile("object", scene_px=64 * 64)
    audit_dir = tmp_path / "cascade_recall_audit"
    roi = _roi(0, _diagonal_line(60), scene_id="audit_scene")

    kept, dropped = filter_rois([roi], profile, audit_dir=audit_dir)

    assert dropped == [roi]
    expected_path = audit_dir / "audit_scene__anomaly__0000.json"
    assert expected_path.exists(), f"no audit record written for dropped roi at {expected_path}"

    record = json.loads(expected_path.read_text())
    assert record["roi_id"] == roi.roi_id
    assert record["stage"] == "postfilter.filter_rois"
    assert record["failing_criterion"] == "elongation_above_max"
    assert record["margin"] > 0.0
    assert record["max_elongation"] == pytest.approx(8.0)


def test_filter_rois_does_not_write_audit_record_for_kept_rois(tmp_path):
    profile = resolve_profile("object", scene_px=64 * 64)
    audit_dir = tmp_path / "cascade_recall_audit"
    roi = _roi(0, _square(20))

    kept, dropped = filter_rois([roi], profile, audit_dir=audit_dir)

    assert kept == [roi]
    assert dropped == []
    # Kept ROIs generate no audit trail at all -- the directory may not even
    # be created if every roi in the batch is kept.
    if audit_dir.exists():
        assert list(audit_dir.iterdir()) == []


# --- both profiles exercised end-to-end via filter_rois ------------------------

def test_filter_rois_object_profile_end_to_end(tmp_path):
    profile = resolve_profile("object", scene_px=64 * 64)
    rois = [_roi(0, _square(10), target_profile="object"),
            _roi(1, _diagonal_line(30), target_profile="object")]

    kept, dropped = filter_rois(rois, profile, audit_dir=tmp_path / "audit")

    assert [r.roi_id for r in kept] == [rois[0].roi_id]
    assert [r.roi_id for r in dropped] == [rois[1].roi_id]


def test_filter_rois_landcover_profile_end_to_end(tmp_path):
    profile = resolve_profile("landcover", scene_px=64 * 64)
    rois = [
        _roi(0, _square(10), target_profile="landcover"),      # area=100 >=50, plausible
        _roi(1, np.ones((3, 3), dtype=np.uint8), target_profile="landcover"),  # area=9 <50, implausible
    ]

    kept, dropped = filter_rois(rois, profile, audit_dir=tmp_path / "audit")

    assert [r.roi_id for r in kept] == [rois[0].roi_id]
    assert [r.roi_id for r in dropped] == [rois[1].roi_id]
    record = json.loads((tmp_path / "audit" / rois[1].roi_id.replace(':', '__').replace('/', '_')
                          ).with_suffix(".json").read_text())
    assert record["failing_criterion"] == "area_below_min"
    assert record["margin"] == pytest.approx(50 - 9)
