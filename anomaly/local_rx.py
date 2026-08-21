"""PLAN.md §3A.2.

Dual concentric window RX: background mu/Sigma at each pixel come from the
ANNULUS between an outer window and an inner guard window, not from the
whole scene (contrast anomaly/rx.py::global_rx, whose background is every
valid pixel in the image). This is the detector that should beat global RX
on scenes where the "background" is not spatially stationary -- an anomaly
sitting in one corner of ABU-Airport-1 should not be judged against the
statistics of the opposite corner.

Band reduction runs FIRST and is mandatory (not an optimization): at the
native band count (e.g. B=205 for ABU) an annulus of outer=21, inner=5 has
only 21**2 - 5**2 = 416 samples, which cannot support a rank-205 covariance
estimate -- Sigma would be singular and the detector would silently
degenerate to noise. Reducing to n_components first (default 20) keeps the
per-pixel sample count comfortably above the parameter count.
"""
from __future__ import annotations

import numpy as np

from preprocessing.harmonize import reduce_bands


def _integral_image(a: np.ndarray) -> np.ndarray:
    """Summed-area table of `a` along its first two axes, trailing axes
    (band, or band x band) carried through untouched. Padded with a leading
    zero row/column so that box queries never need to special-case r0==0 or
    c0==0 -- see _box_sums.
    """
    h, w = a.shape[:2]
    ii = np.zeros((h + 1, w + 1) + a.shape[2:], dtype=np.float64)
    ii[1:, 1:, ...] = np.cumsum(np.cumsum(a, axis=0, dtype=np.float64), axis=1)
    return ii


def _box_sums(ii: np.ndarray, r0: np.ndarray, r1: np.ndarray,
              c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Inclusion-exclusion box query, evaluated for every (row, col) pair
    at once: r0/r1/c0/c1 are length-H / length-W index vectors (window
    bounds per row, per column), and the outer-product indexing below scores
    every pixel's window sum in one gather instead of a per-pixel loop --
    this is the "integral image" half of the O(H*W*C^2) complexity claim in
    the plan; looping per pixel here would make it O(H*W) python calls on
    top.
    """
    a = ii[r1[:, None], c1[None, :]]
    b = ii[r0[:, None], c1[None, :]]
    c = ii[r1[:, None], c0[None, :]]
    d = ii[r0[:, None], c0[None, :]]
    return a - b - c + d


def local_rx(cube: np.ndarray, *, inner: int = 5, outer: int = 21,
             reg: float = 1e-4, n_components: int = 20) -> np.ndarray:
    """Dual concentric window RX.

    cube: [H, W, B] float32, NaN = nodata.
    returns: [H, W] float32, NaN wherever the input pixel was NaN, and also
    wherever a pixel's annulus has fewer than n_components*2 valid
    neighbours (edge pixels under a truncated annulus, or pixels surrounded
    by nodata).

    outer / inner are window SIDE LENGTHS (not radii); both are treated as
    odd (half = size // 2, window = [center - half, center + half]) -- every
    value this module is spec'd against (21/5, 15/3, 7/5) is odd. An even
    value still runs, just with its window silently short by one pixel on
    the high side.

    Background mu/Sigma at pixel (r, c) are estimated ONLY from the annulus
    (outer window minus inner guard window) centered on (r, c), so the guard
    excludes the pixel itself and its immediate neighbours from its own
    background estimate -- the point of local RX over global RX is exactly
    that this keeps a real target's own bright pixels out of the statistics
    used to judge it.

    Implementation: PCA band reduction first (mandatory, see module
    docstring), then two integral images per component -- one for the first
    moment (sum of x), one for the second moment (sum of x @ x.T) -- turn
    the annulus mu/Sigma at every pixel into four O(1) table lookups instead
    of re-scanning the window per pixel. Since outer and inner windows are
    concentric, annulus_sum = outer_box_sum - inner_box_sum; no separate
    annulus-shaped integral image is needed.
    """
    assert outer > inner and (outer ** 2 - inner ** 2) > n_components * 2, (
        f"annulus too small for n_components: outer={outer}, inner={inner} "
        f"-> {outer**2 - inner**2} samples, need > {n_components * 2}"
    )

    h, w, b = cube.shape

    # reduce_bands (preprocessing/harmonize.py) already does exactly what
    # this step needs: fit_on=None fits a fresh PCA on the cube's own valid
    # (non-NaN) pixels and transforms the whole cube, carrying NaN through
    # at invalid pixels. Reused rather than re-implemented; the "fit a
    # pre-fitted transformer" branch of reduce_bands does not apply here --
    # local_rx is a per-scene, unsupervised detector like global_rx, with no
    # train/eval split to fit on beforehand.
    reduced, _ = reduce_bands(cube, n_components=n_components, method="pca", fit_on=None)
    c = reduced.shape[-1]

    valid = ~np.any(np.isnan(reduced), axis=-1)          # [H, W]
    # Selection, not arithmetic: NaN reduced-pixels get replaced by 0 via
    # np.where (a gather), never multiplied against -- `0.0 * NaN == NaN`
    # would silently poison every window sum that touches a nodata pixel.
    filled = np.where(valid[..., None], reduced, 0.0).astype(np.float64)  # [H, W, C]

    half_outer, half_inner = outer // 2, inner // 2
    rows, cols = np.arange(h), np.arange(w)
    outer_r0, outer_r1 = np.clip(rows - half_outer, 0, h), np.clip(rows + half_outer + 1, 0, h)
    outer_c0, outer_c1 = np.clip(cols - half_outer, 0, w), np.clip(cols + half_outer + 1, 0, w)
    inner_r0, inner_r1 = np.clip(rows - half_inner, 0, h), np.clip(rows + half_inner + 1, 0, h)
    inner_c0, inner_c1 = np.clip(cols - half_inner, 0, w), np.clip(cols + half_inner + 1, 0, w)

    count_ii = _integral_image(valid.astype(np.float64))
    sum1_ii = _integral_image(filled)
    sum2_ii = _integral_image(filled[..., :, None] * filled[..., None, :])

    outer_count = _box_sums(count_ii, outer_r0, outer_r1, outer_c0, outer_c1)
    inner_count = _box_sums(count_ii, inner_r0, inner_r1, inner_c0, inner_c1)
    annulus_count = outer_count - inner_count                              # [H, W]

    outer_sum1 = _box_sums(sum1_ii, outer_r0, outer_r1, outer_c0, outer_c1)
    inner_sum1 = _box_sums(sum1_ii, inner_r0, inner_r1, inner_c0, inner_c1)
    annulus_sum1 = outer_sum1 - inner_sum1                                 # [H, W, C]

    outer_sum2 = _box_sums(sum2_ii, outer_r0, outer_r1, outer_c0, outer_c1)
    inner_sum2 = _box_sums(sum2_ii, inner_r0, inner_r1, inner_c0, inner_c1)
    annulus_sum2 = outer_sum2 - inner_sum2                                 # [H, W, C, C]

    enough = annulus_count >= (n_components * 2)
    score_mask = valid & enough

    # Guard against 0/0 for pixels that will be masked out of the final
    # result anyway -- selection (np.where), not arithmetic, keeps the
    # divide-by-zero out of the values that actually get used.
    safe_count = np.where(annulus_count > 0, annulus_count, 1.0)
    mu = annulus_sum1 / safe_count[..., None]                              # [H, W, C]
    sigma = (annulus_sum2 / safe_count[..., None, None]
             - mu[..., :, None] * mu[..., None, :])                        # [H, W, C, C]
    sigma = sigma + reg * np.eye(c, dtype=np.float64)

    scores = np.full((h, w), np.nan, dtype=np.float64)
    if score_mask.any():
        dv = filled[score_mask] - mu[score_mask]                          # [N, C]
        sigma_sel = sigma[score_mask]                                     # [N, C, C]
        # Batched LU solve, not a per-pixel cho_factor/cho_solve loop and
        # never an explicit inverse: global_rx factors ONE covariance
        # matrix, but local_rx has a distinct covariance per query pixel,
        # so the "never form an explicit inverse" rule here is satisfied by
        # np.linalg.solve (decomposition-based, like cho_solve) rather than
        # np.linalg.inv, vectorized across all N pixels in a single call to
        # stay inside the runtime budget.
        solved = np.linalg.solve(sigma_sel, dv[..., None])[..., 0]        # [N, C]
        scores[score_mask] = np.einsum("ij,ij->i", dv, solved)

    return scores.astype(np.float32)
