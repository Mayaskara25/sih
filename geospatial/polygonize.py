"""PLAN.md §2.6."""
from __future__ import annotations

import affine
import numpy as np
import rasterio.features
import shapely.geometry
import shapely.ops

from core.contracts import ROIRecord, SceneMeta
from segmentation.postfilter import connected_components


def mask_to_rois(mask: np.ndarray, meta: SceneMeta, *, source_branch: str,
                  target_profile: str) -> list[ROIRecord]:
    """Connected components -> C5 ROIRecords. source_branch and target_profile
    are set HERE, at ROI birth, per D5 -- never inferred downstream.
    """
    labels, n = connected_components(mask, connectivity=8)
    rois: list[ROIRecord] = []
    for i in range(1, n + 1):
        rows, cols = np.where(labels == i)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        local_mask = (labels[r0:r1, c0:c1] == i).astype(np.uint8)
        roi = ROIRecord(
            roi_id=f"{meta.scene_id}:{source_branch}:{len(rois):04d}",
            source_branch=source_branch,
            target_profile=target_profile,
            bbox=(r0, c0, r1, c1),
            mask=local_mask,
        )
        rois.append(roi)
    return rois


def rois_to_polygons(rois: list[ROIRecord], meta: SceneMeta) -> list[shapely.geometry.Polygon]:
    """rasterio.features.shapes on each ROI mask, pixel -> native-CRS via
    meta.transform. Holes preserved. Polygons are NOT reprojected here (C7).

    One polygon per ROI: connectivity=8 matches the 8-connectivity used to
    build the ROI in mask_to_rois, so shapes() should already yield a single
    piece; unary_union guards the (rasterization) edge case where it doesn't,
    keeping this 1:1 with `rois`.
    """
    polygons = []
    for roi in rois:
        r0, c0, _, _ = roi.bbox
        local_transform = meta.transform * affine.Affine.translation(c0, r0)
        shapes = rasterio.features.shapes(
            roi.mask, mask=roi.mask.astype(bool), transform=local_transform, connectivity=8)
        pieces = [shapely.geometry.shape(geom) for geom, _value in shapes]
        polygons.append(shapely.ops.unary_union(pieces) if len(pieces) > 1 else pieces[0])
    return polygons
