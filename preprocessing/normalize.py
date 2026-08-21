"""PLAN.md §2.2."""
from __future__ import annotations

import numpy as np

from core.contracts import SceneMeta

# Standard water-absorption bands for the pre-corrected 200-band Indian Pines
# product. NOTE: PLAN.md §2.2's comment lists indices [104-108,150-163,220],
# but those index into the ORIGINAL 220-band AVIRIS layout. D13.1 confirmed
# the shipped Indian_pines_corrected.mat cube is ALREADY the 200-band
# water-band-removed product (220 -> 200 upstream, one variable, no metadata
# recording which 20 bands were dropped). Applying the 220-band indices to a
# 200-band array would drop the wrong bands outright (and index 220 doesn't
# exist), so drop_bad_bands is a documented no-op for this source.
INDIAN_PINES_BAD_BANDS_ORIGINAL_220 = tuple(range(103, 108)) + tuple(range(149, 163)) + (219,)


def drop_bad_bands(cube: np.ndarray, meta: SceneMeta) -> tuple[np.ndarray, SceneMeta]:
    if meta.source == "indian_pines":
        # Already removed upstream (D13.1) -- nothing to drop from this cube.
        return cube, meta
    keep = ~meta.bad_bands
    from dataclasses import replace

    new_wavelengths = meta.wavelengths[keep] if meta.wavelengths is not None else None
    new_meta = replace(meta, wavelengths=new_wavelengths, bad_bands=meta.bad_bands[keep])
    # Boolean-mask indexing along a non-leading axis doesn't guarantee a
    # C-contiguous result even when the mask keeps everything (numpy quirk) --
    # cube must satisfy C1's C-contiguous requirement.
    return np.ascontiguousarray(cube[..., keep]), new_meta


def l2_normalize(cube: np.ndarray) -> np.ndarray:
    """Per-pixel spectral L2 normalization, brightness-invariant. NaN-safe."""
    norm = np.sqrt(np.nansum(cube ** 2, axis=-1, keepdims=True))
    norm = np.where(norm == 0, 1.0, norm)
    out = cube / norm
    return np.ascontiguousarray(out.astype(np.float32))


def standardize(cube: np.ndarray) -> np.ndarray:
    """Per-band zero-mean unit-variance standardization. NaN-safe."""
    mean = np.nanmean(cube, axis=(0, 1), keepdims=True)
    std = np.nanstd(cube, axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    out = (cube - mean) / std
    return np.ascontiguousarray(out.astype(np.float32))
