"""plan.md Section 4.3 / D5 -- fuse_rois.

D5's rule under test: Phase-4 ROI fusion merges ONLY parents sharing
`target_profile` (result: source_branch="fused", profile inherited,
parent_roi_ids set). Different-profile spatial overlaps are NEVER merged --
both survive and each records the other in linked_roi_ids, reciprocally.
The accept criterion (one `object` anomaly ROI + one `landcover` change ROI
at IoU 0.8 -> two ROIs, reciprocally linked, neither "fused") is tested
verbatim below.

All fixtures are hand-built ROIRecords -- no real data, no disk I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.contracts import ROIRecord, validate_roi
from geospatial.roi_fusion import fuse_rois


def _solid_roi(scene_id: str, branch: str, index: int, profile: str,
                bbox: tuple[int, int, int, int], **scores) -> ROIRecord:
    """An ROI whose mask is entirely filled (1s) across its bbox -- for
    these, mask IoU and bbox IoU coincide, which keeps the geometry of
    "plain overlap" tests easy to reason about by hand.
    """
    r0, c0, r1, c1 = bbox
    mask = np.ones((r1 - r0, c1 - c0), dtype=np.uint8)
    return ROIRecord(
        roi_id=f"{scene_id}:{branch}:{index:04d}", source_branch=branch,
        target_profile=profile, bbox=bbox, mask=mask, **scores)


# --- D5 accept criterion, quoted from the spec ------------------------------

def test_accept_criterion_cross_profile_iou_0p8_yields_two_linked_rois():
    """One `object` anomaly ROI + one `landcover` change ROI at mask IoU
    0.8 -> TWO output ROIs, reciprocally linked, neither "fused".

    Two 9x9 solid squares offset by 1 column: intersection = 9*8 = 72,
    union = 81+81-72 = 90, IoU = 72/90 = 0.8 exactly.
    """
    a = _solid_roi("s1", "anomaly", 0, "object", (0, 0, 9, 9), anomaly_score=0.9)
    b = _solid_roi("s1", "change", 0, "landcover", (0, 1, 9, 10), change_score=0.7)

    out = fuse_rois([a], [b], iou_threshold=0.3)

    assert len(out) == 2
    branches = {r.source_branch for r in out}
    assert branches == {"anomaly", "change"}   # neither is "fused"

    by_id = {r.roi_id: r for r in out}
    assert a.roi_id in by_id and b.roi_id in by_id
    assert by_id[b.roi_id].roi_id in by_id[a.roi_id].linked_roi_ids
    assert by_id[a.roi_id].roi_id in by_id[b.roi_id].linked_roi_ids
    for r in out:
        validate_roi(r)


# --- same-profile merge ------------------------------------------------------

def test_same_profile_parents_above_threshold_merge():
    """Two same-`target_profile` parents (one anomaly-branch, one
    change-branch -- D6: either branch can run under either profile) at
    IoU 0.8 merge into one fused ROI. Scores are a JOIN: anomaly_score
    comes only from `a` (b's is None), change_score comes only from `b`.
    """
    a = _solid_roi("s1", "anomaly", 0, "object", (0, 0, 9, 9), anomaly_score=0.9)
    b = _solid_roi("s1", "change", 0, "object", (0, 1, 9, 10), change_score=0.4)

    out = fuse_rois([a], [b], iou_threshold=0.3)

    assert len(out) == 1
    fused = out[0]
    assert fused.source_branch == "fused"
    assert fused.target_profile == "object"
    assert fused.roi_id == "s1:fused:0000"
    assert set(fused.parent_roi_ids) == {a.roi_id, b.roi_id}
    assert fused.anomaly_score == pytest.approx(0.9)
    assert fused.change_score == pytest.approx(0.4)
    assert fused.bbox == (0, 0, 9, 10)          # union bbox
    assert fused.linked_roi_ids == []
    validate_roi(fused)


def test_same_profile_merge_takes_max_when_both_parents_carry_the_field():
    """When BOTH merging parents have a value for the same score field
    (e.g. two `object` anomaly ROIs), the join takes the max, not an
    average -- the stronger of two redundant readings survives fusion."""
    a = _solid_roi("s1", "anomaly", 0, "object", (0, 0, 9, 9), anomaly_score=0.3)
    b = _solid_roi("s1", "anomaly", 1, "object", (0, 1, 9, 10), anomaly_score=0.9)

    out = fuse_rois([a, b], [], iou_threshold=0.3)

    assert len(out) == 1
    assert out[0].anomaly_score == pytest.approx(0.9)


def test_same_profile_parents_below_threshold_do_not_merge():
    """Tiny overlap -> IoU well under threshold -> both ROIs survive
    unchanged, same-profile pair excluded from cross-profile linking."""
    a = _solid_roi("s1", "change", 0, "landcover", (0, 0, 20, 20))
    b = _solid_roi("s1", "change", 1, "landcover", (0, 19, 20, 39))  # 1-col overlap

    out = fuse_rois([], [a, b], iou_threshold=0.3)

    assert len(out) == 2
    ids = {r.roi_id for r in out}
    assert ids == {a.roi_id, b.roi_id}
    for r in out:
        assert r.source_branch == "change"
        assert r.linked_roi_ids == []
        validate_roi(r)


# --- mask IoU, not bbox IoU --------------------------------------------------

def test_uses_mask_iou_not_bbox_iou():
    """Both ROIs share the EXACT SAME bbox (bbox-only IoU would read 1.0
    and would wrongly merge them), but their masks occupy disjoint
    quadrants of that bbox -- mask IoU is 0.0. Must NOT merge."""
    bbox = (0, 0, 4, 4)
    mask_a = np.zeros((4, 4), dtype=np.uint8)
    mask_a[0:2, 0:2] = 1   # top-left 2x2
    mask_b = np.zeros((4, 4), dtype=np.uint8)
    mask_b[2:4, 2:4] = 1   # bottom-right 2x2

    a = ROIRecord(roi_id="s1:anomaly:0000", source_branch="anomaly",
                  target_profile="object", bbox=bbox, mask=mask_a)
    b = ROIRecord(roi_id="s1:anomaly:0001", source_branch="anomaly",
                  target_profile="object", bbox=bbox, mask=mask_b)

    out = fuse_rois([a, b], [], iou_threshold=0.3)

    assert len(out) == 2   # bbox IoU (1.0) would have merged; mask IoU (0.0) does not
    ids = {r.roi_id for r in out}
    assert ids == {a.roi_id, b.roi_id}
    for r in out:
        assert r.source_branch == "anomaly"
        validate_roi(r)


# --- non-overlapping pass-through -------------------------------------------

def test_non_overlapping_rois_pass_through_untouched():
    a = _solid_roi("s1", "anomaly", 0, "object", (0, 0, 5, 5))
    b = _solid_roi("s1", "change", 0, "landcover", (100, 100, 105, 105))

    out = fuse_rois([a], [b], iou_threshold=0.3)

    assert len(out) == 2
    by_id = {r.roi_id: r for r in out}
    assert by_id[a.roi_id].bbox == a.bbox
    assert by_id[b.roi_id].bbox == b.bbox
    assert np.array_equal(by_id[a.roi_id].mask, a.mask)
    assert np.array_equal(by_id[b.roi_id].mask, b.mask)
    assert by_id[a.roi_id].linked_roi_ids == []
    assert by_id[b.roi_id].linked_roi_ids == []
    for r in out:
        validate_roi(r)


# --- transitive / chained merging -------------------------------------------

def test_transitive_chain_merge_all_three_fuse_into_one():
    """A-B IoU and B-C IoU both clear threshold; direct A-C IoU does not.
    Documented decision: merging is transitive (connected components over
    the threshold graph), so all three still collapse into ONE fused ROI.

    Three 10-wide solid rows, each offset 3 columns from the last:
    A-B and B-C: intersection=10*7=70, union=200-70=130, IoU=70/130=0.538.
    A-C (offset 6): intersection=10*4=40, union=200-40=160, IoU=40/160=0.25.
    0.25 < 0.3 <= 0.538, exactly the chain-but-not-direct case.
    """
    a = _solid_roi("s1", "anomaly", 0, "landcover", (0, 0, 10, 10))
    b = _solid_roi("s1", "change", 0, "landcover", (0, 3, 10, 13))
    c = _solid_roi("s1", "change", 1, "landcover", (0, 6, 10, 16))

    from geospatial.roi_fusion import _mask_iou
    assert _mask_iou(a, b) == pytest.approx(70 / 130)
    assert _mask_iou(b, c) == pytest.approx(70 / 130)
    assert _mask_iou(a, c) == pytest.approx(40 / 160)
    assert _mask_iou(a, c) < 0.3 <= _mask_iou(a, b)

    out = fuse_rois([a], [b, c], iou_threshold=0.3)

    assert len(out) == 1
    fused = out[0]
    assert fused.source_branch == "fused"
    assert set(fused.parent_roi_ids) == {a.roi_id, b.roi_id, c.roi_id}
    assert fused.bbox == (0, 0, 10, 16)   # union of all three
    validate_roi(fused)


# --- determinism --------------------------------------------------------------

def test_determinism_same_input_same_roi_ids():
    """Two calls on freshly-built, value-identical input must yield
    identical roi_ids -- a nondeterministic id would make run manifests
    irreproducible."""
    def build():
        a1 = _solid_roi("s1", "anomaly", 0, "object", (0, 0, 9, 9))
        b1 = _solid_roi("s1", "change", 0, "object", (0, 1, 9, 10))       # merges with a1
        c1 = _solid_roi("s1", "anomaly", 1, "object", (50, 50, 55, 55))   # singleton
        d1 = _solid_roi("s1", "change", 1, "landcover", (50, 50, 55, 55))  # links with c1
        return [a1, c1], [b1, d1]

    anomaly_1, change_1 = build()
    out_1 = fuse_rois(anomaly_1, change_1, iou_threshold=0.3)

    anomaly_2, change_2 = build()
    out_2 = fuse_rois(anomaly_2, change_2, iou_threshold=0.3)

    ids_1 = [r.roi_id for r in out_1]
    ids_2 = [r.roi_id for r in out_2]
    assert ids_1 == ids_2

    links_1 = {r.roi_id: sorted(r.linked_roi_ids) for r in out_1}
    links_2 = {r.roi_id: sorted(r.linked_roi_ids) for r in out_2}
    assert links_1 == links_2


# --- empty input --------------------------------------------------------------

def test_empty_input_returns_empty_list():
    assert fuse_rois([], []) == []
