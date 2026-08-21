"""geospatial/roi_fusion.py -- plan.md Section 4.3 / D5.

Phase-4 ROI-level fusion, run once per scene after the anomaly branch (3A)
and change branch (3C) have each produced their own `ROIRecord` lists via
`geospatial.polygonize.mask_to_rois`.

D5's rule, restated: fusion is gated on `target_profile`, not on spatial
overlap alone.

    - Same `target_profile`      -> MERGE. Two (or more, transitively --
      see `fuse_rois`'s docstring) parents collapse into one
      `source_branch="fused"` ROI that inherits the shared profile and
      records every parent in `parent_roi_ids`.
    - Different `target_profile` -> NEVER merge, however high the IoU.
      Both survive as independent ROIs and each records the other's
      `roi_id` in `linked_roi_ids` (reciprocally).

Why not merge-and-take-the-higher-scoring-parent's-profile (the documented
alternative, D5)? Because `object` and `landcover` ROIs were screened by
DIFFERENT post-filters (D6: different `min_area_px`, `min_solidity`,
`max_elongation`, different index sets). A merged ROI that ended up
carrying, say, `landcover`'s loose `min_solidity=0.05` after actually having
been shaped by `object`'s strict `min_solidity=0.15` filter would silently
misrepresent what it was screened against -- exactly the "silent
integration mismatch" the frozen contracts (core/contracts.py) exist to
prevent. Non-merge is the only choice that keeps every ROI's provenance
honest; the merge-and-pick-a-profile alternative is defensible on paper but
destroys that audit trail, which is why D5 rejected it.

IoU is computed over MASKS, not bboxes. `ROIRecord.mask` is bbox-local
(C3/C5), so two masks are only comparable after both are placed on a shared
pixel canvas -- the union of the two bboxes. A bbox-only IoU would treat
two ROIs with heavily overlapping bounding boxes but near-disjoint actual
shapes (e.g. two slivers occupying opposite corners of the same bbox) as
heavily overlapping; this module never does that -- see `_mask_iou`.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.contracts import ROIRecord, validate_roi

# Every optional scalar score field on C5's ROIRecord. Iterated by name
# (rather than hardcoding five separate lines in fuse_rois) so adding a new
# scalar field to the contract only requires updating this tuple.
_SCORE_FIELDS = (
    "anomaly_score", "change_score", "seg_prob", "clear_fraction", "shape_plausibility",
)


def _scene_id(roi: ROIRecord) -> str:
    """`roi_id` is `"{scene_id}:{branch}:{index:04d}"` (D5); scene_id is field 0."""
    parts = roi.roi_id.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"fuse_rois: malformed roi_id {roi.roi_id!r}, expected 3 ':'-separated fields")
    return parts[0]


def _union_bbox(rois: list[ROIRecord]) -> tuple[int, int, int, int]:
    r0 = min(r.bbox[0] for r in rois)
    c0 = min(r.bbox[1] for r in rois)
    r1 = max(r.bbox[2] for r in rois)
    c1 = max(r.bbox[3] for r in rois)
    return r0, c0, r1, c1


def _place(roi: ROIRecord, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """`roi.mask` (bbox-local, C3) placed as a bool array on `bbox`'s canvas.

    `bbox` must enclose `roi.bbox` -- callers here always build it from
    `_union_bbox` over a set that includes `roi` itself.
    """
    r0, c0, r1, c1 = bbox
    canvas = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    rr0, rc0, rr1, rc1 = roi.bbox
    canvas[rr0 - r0:rr1 - r0, rc0 - c0:rc1 - c0] = roi.mask.astype(bool)
    return canvas


def _bboxes_disjoint(a: ROIRecord, b: ROIRecord) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0


def _mask_iou(a: ROIRecord, b: ROIRecord) -> float:
    """Mask-level IoU (deliberately NOT bbox IoU -- see module docstring).

    Two ROIs can share an identical, or heavily overlapping, bbox while
    their actual masks are near-disjoint; only resolving both masks into a
    shared coordinate frame first gives the geometrically correct answer.

    Disjoint bboxes are the common case at scene scale (most ROI pairs in a
    scene are nowhere near each other) and imply disjoint masks exactly --
    no need to allocate a shared canvas (which can be scene-sized) to learn
    IoU == 0.0 for a pair this function is going to be called on O(n^2)
    times.
    """
    if _bboxes_disjoint(a, b):
        return 0.0
    bbox = _union_bbox([a, b])
    ca, cb = _place(a, bbox), _place(b, bbox)
    union = np.count_nonzero(ca | cb)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(ca & cb)) / float(union)


def _join_score(rois: list[ROIRecord], field: str) -> float | None:
    """Combine one scalar score field across a group of merging parents.

    This is a JOIN, not an average. An anomaly-branch parent has
    `anomaly_score` set and `change_score is None` (and the reverse for a
    change-branch parent) -- most fields on a merging pair/group are
    present on only ONE member, and there the value is simply inherited
    unchanged. When a field genuinely is present on more than one member
    (e.g. two `object`-profile anomaly ROIs merging, both with
    `anomaly_score` set), the MAX is taken: Phase-4 fusion exists to
    collapse duplicate detections of one physical target, and the stronger
    of two redundant readings is the more informative one to carry
    forward -- averaging would dilute whichever branch detected it more
    confidently. A field absent from every member stays None.
    """
    present = [v for v in (getattr(r, field) for r in rois) if v is not None]
    return max(present) if present else None


def _union_mask_and_bbox(
        rois: list[ROIRecord]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Union bbox (smallest enclosing box) + pixel-wise OR of every member's
    mask on that shared canvas.

    Union, not intersection: this is Phase-4 fusion of independently
    detected candidates for the same target, not a refinement/consensus
    step -- intersecting would silently shrink the ROI to only the pixels
    every branch happened to agree on, discarding real target extent that
    only one detector recovered.
    """
    bbox = _union_bbox(rois)
    r0, c0, r1, c1 = bbox
    canvas = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for r in rois:
        canvas |= _place(r, bbox)
    return bbox, canvas.astype(np.uint8)


def fuse_rois(anomaly_rois: list[ROIRecord], change_rois: list[ROIRecord], *,
              iou_threshold: float = 0.3) -> list[ROIRecord]:
    """Per D5 / plan.md Section 4.3. See the module docstring for the
    merge-vs-link rule and its rationale.

    The plan states the rule but leaves several mechanics open; D5 itself
    sets the standard that silence on an open point is not acceptable, so
    each is decided and documented here rather than left implicit:

    1. **Grouping.** ROIs are combined (`anomaly_rois + change_rois`, in
       that order) and an edge is added between two ROIs when their mask
       IoU is >= `iou_threshold` AND they share `target_profile` -- a
       cross-profile pair can never become a graph edge, however high its
       IoU. Each connected component of size > 1 becomes ONE fused ROI;
       singletons pass through with bbox/mask/scores/roi_id untouched
       (only `linked_roi_ids` may later be populated, step 6).

    2. **Transitivity: merging IS transitive.** If A-B and B-C both clear
       `iou_threshold` (all three sharing one profile) but A-C does not,
       all three still merge into ONE fused ROI -- this matches the
       connected-components semantics `geospatial.polygonize.mask_to_rois`
       already uses to turn raw pixels into ROIs. Rationale: once B is
       folded into a fused result, "A and C are both still-separate
       parents of B" is not an expressible output state -- B has exactly
       one fused identity, so the natural closure over the threshold graph
       is its whole connected component, not just its strongest edge.
       (Tested explicitly: a 3-ROI chain where the direct A-C IoU is below
       threshold but the component is still one fused ROI.)

    3. **Merged geometry** -- see `_union_mask_and_bbox`: union bbox, OR'd
       mask.

    4. **Merged scores** -- see `_join_score`: fields set by exactly one
       parent are inherited as-is (a JOIN); fields set by more than one
       parent take the max.

    5. **Fused `roi_id`.** Format is D5's `"{scene_id}:fused:{index:04d}"`.
       `scene_id` must agree across every input ROI -- fusion is inherently
       per-scene (bbox pixel coordinates from two different scenes are not
       spatially comparable), so a mismatch raises `ValueError` rather than
       guessing which scene is meant. `index` is assigned in the
       deterministic order components are first encountered scanning
       `anomaly_rois + change_rois` left to right -- NOT sorted by roi_id
       or by IoU -- so identical input always produces identical fused ids
       (a nondeterministic id would make run manifests irreproducible).
       `parent_roi_ids` on each fused ROI preserves that same left-to-right
       order.

    6. **Cross-profile linking is evaluated on the POST-merge output set**,
       not on the original parents: if two same-profile ROIs first fuse,
       it is the FUSED shape's mask -- not either original parent's mask --
       that is tested for overlap against a different-profile ROI, using
       the same `iou_threshold`. This keeps the linking decision consistent
       with what the caller actually sees in the returned list, and gives
       the module one "meaningfully overlapping" definition throughout
       rather than a second, silently different, one for the cross-profile
       case (the plan does not pin this down explicitly; reusing
       `iou_threshold` rather than "any nonzero overlap" was the reading
       adopted here).

    Parameters
    ----------
    anomaly_rois, change_rois : ROI lists to fuse -- typically Stage 3A / 3C
        output for ONE scene (`geospatial.polygonize.mask_to_rois`).
        Neither list needs to be homogeneous in `target_profile` (D6:
        either branch may run under either profile).
    iou_threshold : mask IoU at or above which two ROIs are treated as the
        "same detection" (same profile -> merge) or as a "linked"
        cross-profile pair (different profile -> reciprocal
        `linked_roi_ids`). Comparison is inclusive (`>=`).

    Returns
    -------
    list[ROIRecord]
        Every returned ROI passes `core.contracts.validate_roi`.
    """
    combined = list(anomaly_rois) + list(change_rois)
    if not combined:
        return []

    scene_ids = {_scene_id(r) for r in combined}
    if len(scene_ids) > 1:
        raise ValueError(
            f"fuse_rois: input ROIs span multiple scenes {sorted(scene_ids)}; "
            "fusion is per-scene (pixel bboxes from different scenes are not "
            "spatially comparable)")
    scene_id = scene_ids.pop()

    n = len(combined)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        for j in range(i + 1, n):
            if combined[i].target_profile != combined[j].target_profile:
                continue
            if _mask_iou(combined[i], combined[j]) >= iou_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    output: list[ROIRecord] = []
    fused_index = 0
    for root in sorted(groups):   # root == smallest member index -> first-encountered order
        members = [combined[i] for i in groups[root]]
        if len(members) == 1:
            r = members[0]
            # Copy (not mutate) -- cross-profile linking below may append to
            # linked_roi_ids, and the caller's original ROI list must not be
            # touched by this function.
            output.append(replace(
                r, linked_roi_ids=list(r.linked_roi_ids), parent_roi_ids=list(r.parent_roi_ids)))
            continue

        bbox, mask = _union_mask_and_bbox(members)
        fused = ROIRecord(
            roi_id=f"{scene_id}:fused:{fused_index:04d}",
            source_branch="fused",
            target_profile=members[0].target_profile,   # shared by construction (edges same-profile only)
            bbox=bbox,
            mask=mask,
            **{field: _join_score(members, field) for field in _SCORE_FIELDS},
            linked_roi_ids=[],
            parent_roi_ids=[m.roi_id for m in members],
        )
        fused_index += 1
        output.append(fused)

    for i in range(len(output)):
        for j in range(i + 1, len(output)):
            a, b = output[i], output[j]
            if a.target_profile == b.target_profile:
                continue
            if _mask_iou(a, b) >= iou_threshold:
                if b.roi_id not in a.linked_roi_ids:
                    a.linked_roi_ids.append(b.roi_id)
                if a.roi_id not in b.linked_roi_ids:
                    b.linked_roi_ids.append(a.roi_id)

    for roi in output:
        validate_roi(roi)
    return output
