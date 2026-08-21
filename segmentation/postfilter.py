"""PLAN.md §2.5. Phase 2 uses morphology only -- shape_plausibility and
filter_rois are Phase 3B scope (profile-driven post-filtering) and are not
implemented here.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import label


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
