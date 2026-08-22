"""PLAN.md §3C.1 -- sub-pixel co-registration of bi-temporal cube pairs.

Misregistration is the single most-cited real-world failure source in the
change-detection literature: a silently misregistered pair produces change
maps that are pure artefact. This module therefore refuses to return garbage
-- `RegistrationFailure` is raised when the achieved alignment quality is
worse than 1 px RMSE.

Two stages:
  1. `skimage.registration.phase_cross_correlation` on band-averaged
     panchromatic proxies -> sub-pixel translation estimate.
  2. optional `cv2.findTransformECC` (MOTION_AFFINE) refinement for
     residual rotation/scale.

t2 is resampled onto the t1 grid (contract C4). NaN nodata is excluded from
both estimates (filled with per-band means for correlation, restored after
warping).
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from core.contracts import SceneMeta


class RegistrationFailure(Exception):
    """Raised when a pair cannot be aligned to better than 1 px RMSE."""


_RMSE_LIMIT_PX = 1.0


def _panchromatic_proxy(cube: np.ndarray) -> np.ndarray:
    """Band-averaged [H, W] float32 proxy with NaN filled by the band mean."""
    pan = np.nanmean(cube.astype(np.float64), axis=-1)
    if np.isnan(pan).any():
        fill = float(np.nanmean(pan))
        pan = np.where(np.isnan(pan), fill, pan)
    return pan.astype(np.float32)


def coregister_subpixel(cube_t1: np.ndarray, cube_t2: np.ndarray,
                        meta_t1: SceneMeta, meta_t2: SceneMeta, *,
                        upsample_factor: int = 20, refine: bool = True
                        ) -> tuple[np.ndarray, dict]:
    """Co-register cube_t2 onto cube_t1's grid (PLAN.md §3C.1).

    Parameters
    ----------
    cube_t1, cube_t2 : [H, W, B] float32, NaN = nodata. Same scene size.
    meta_t1, meta_t2 : SceneMeta of each epoch (report context only; the
        grids are assumed pixel-aligned up to the estimated transform).
    upsample_factor : phase-correlation upsampling; shift precision is
        1/upsample_factor px.
    refine : run the ECC affine refinement stage after PCC.

    Returns
    -------
    (cube_t2_aligned, report): the warped t2 cube ([H, W] float32, NaN
    restored positionally) and a report dict
    {shift_px, rmse_px, ecc_score, converged}.

    Raises
    ------
    RegistrationFailure if rmse_px > 1.0 -- a silently misregistered pair
    produces change maps that are pure artefact.
    """
    t1 = np.asarray(cube_t1)
    t2 = np.asarray(cube_t2)
    if t1.shape != t2.shape:
        raise ValueError(
            f"coregister_subpixel: shape mismatch {t1.shape} vs {t2.shape}")
    if t1.ndim != 3:
        raise ValueError(f"coregister_subpixel: expected [H, W, B], got {t1.shape}")

    pan1 = _panchromatic_proxy(t1)
    pan2 = _panchromatic_proxy(t2)

    shift_vec, pcc_error, _phasediff = phase_cross_correlation(
        pan1, pan2, upsample_factor=upsample_factor)
    shift_row, shift_col = float(shift_vec[0]), float(shift_vec[1])

    ecc_score = float("nan")
    converged = False

    def _warp(img: np.ndarray, mat: np.ndarray,
              interp: int = cv2.INTER_LINEAR, *,
              inverse_map: bool = False) -> np.ndarray:
        # NOTE on conventions: the PCC shift follows the scipy.ndimage.shift
        # semantic ("move the image BY this much"), which corresponds to
        # warpAffine WITHOUT WARP_INVERSE_MAP. The ECC matrix follows the
        # OpenCV findTransformECC recipe, which REQUIRES WARP_INVERSE_MAP
        # when applied. Mixing these up double-counts the translation.
        flags = interp | (cv2.WARP_INVERSE_MAP if inverse_map else 0)
        return cv2.warpAffine(
            img, mat.astype(np.float32), (img.shape[1], img.shape[0]),
            flags=flags, borderMode=cv2.BORDER_REFLECT)

    # Stage 1: undo the bulk sub-pixel translation found by phase
    # correlation. Applied to the image BEFORE stage 2 so ECC only has to
    # explain the *residual* rotation/scale.
    trans = _translation_matrix(shift_row, shift_col)
    pan2_t = _warp(pan2, trans)

    # Stage 2: optional ECC affine refinement of what remains.
    ecc_matrix: np.ndarray | None = None
    if refine:
        ecc_warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    200, 1e-7)
        try:
            cc, ecc_warp = cv2.findTransformECC(
                pan1, pan2_t, ecc_warp, cv2.MOTION_AFFINE, criteria,
                inputMask=None, gaussFiltSize=5)
            ecc_score = float(cc)
            converged = True
            ecc_matrix = np.asarray(ecc_warp, dtype=np.float64)
        except cv2.error:
            pass

    # MEASURED residual: re-estimate the shift between t1 and the fully
    # warped t2 proxy. This is the honest alignment quality -- not an
    # analytic property of the estimated transform.
    aligned_pan = (_warp(pan2_t, ecc_matrix, inverse_map=True)
                   if ecc_matrix is not None else pan2_t)
    resid_vec, _, _ = phase_cross_correlation(
        pan1, aligned_pan, upsample_factor=upsample_factor)
    rmse_px = float(np.hypot(resid_vec[0], resid_vec[1]))

    if not np.isfinite(rmse_px) or rmse_px > _RMSE_LIMIT_PX:
        raise RegistrationFailure(
            f"coregister_subpixel: residual rmse {rmse_px:.3f} px exceeds "
            f"{_RMSE_LIMIT_PX} px limit (pcc shift row={shift_row:.2f}, "
            f"col={shift_col:.2f}); an un-alignable pair must raise, not "
            f"return artefact change maps")

    # Warp every band through the same two-stage pipeline; restore the
    # input NaN mask positionally so nodata never becomes a fake "change".
    _, _, b = t1.shape
    valid2 = ~np.isnan(t2).any(axis=-1)
    out = np.empty_like(t1)
    for k in range(b):
        band = np.where(np.isnan(t2[..., k]),
                        np.nanmean(t2[..., k]), t2[..., k]).astype(np.float32)
        warped = _warp(band, trans)
        if ecc_matrix is not None:
            warped = _warp(warped, ecc_matrix, inverse_map=True)
        out[..., k] = warped

    valid_aligned = _warp(valid2.astype(np.float32), trans,
                          interp=cv2.INTER_NEAREST) > 0.5
    if ecc_matrix is not None:
        valid_aligned = (_warp(valid_aligned.astype(np.float32), ecc_matrix,
                               interp=cv2.INTER_NEAREST,
                               inverse_map=True) > 0.5)
    out[~valid_aligned] = np.nan

    report = {
        "shift_px": {"row": shift_row, "col": shift_col},
        "residual_px": {"row": float(resid_vec[0]), "col": float(resid_vec[1])},
        "rmse_px": rmse_px,
        "ecc_score": ecc_score,
        "converged": converged,
        "pcc_error": float(pcc_error),
        "meta_t1_georef": meta_t1.georef,
        "meta_t2_georef": meta_t2.georef,
    }
    return out, report

    report = {
        "shift_px": shift_report,
        "rmse_px": rmse_px,
        "ecc_score": ecc_score,
        "converged": converged,
        "pcc_error": float(pcc_error),
        "meta_t1_georef": meta_t1.georef,
        "meta_t2_georef": meta_t2.georef,
    }
    return out, report


def _translation_matrix(shift_row: float, shift_col: float) -> np.ndarray:
    return np.array([[1.0, 0.0, shift_col],
                     [0.0, 1.0, shift_row]], dtype=np.float64)
