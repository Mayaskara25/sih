"""PLAN.md §3A.4 -- Collaborative Representation Detector (CRD).

Each pixel y is represented as a linear combination of its annulus
neighbours X (outer window minus inner guard window, same dual-concentric-
window idea as local_rx §3A.2) under a Tikhonov-weighted least squares fit:

    w = (X^T X + lam * Gamma^T Gamma)^-1 X^T y
    Gamma = diag(||y - x_i||_2)      -- biases the fit against dissimilar
                                         neighbours (a neighbour far from y
                                         in spectral space is penalized more
                                         heavily for contributing weight)
    score = ||y - X w||_2            -- large residual = poorly represented
                                         by its own neighbourhood = anomalous

Implementation notes.

- The annulus is a SQUARE window (outer x outer minus a centered inner x
  inner box), the same convention local_rx uses, not a circular one.
- The image is padded by outer // 2 with NaN (never reflect/edge padding):
  an earlier reflect-padded version was found, by test, to fold a border
  pixel's own value back into its own annulus at some corners (two
  perpendicular reflections composing back to the original coordinate),
  giving that pixel a "neighbour" at distance 0 -- Gamma_i = 0 there kills
  the Tikhonov penalty for that slot AT ANY lam, so the pixel trivially and
  perfectly reconstructs itself and its score silently collapses to ~0
  regardless of lam or how genuinely anomalous it is. NaN-padding instead
  routes every border pixel through the same real-neighbours-only
  "slow path" already required for interior nodata (below) -- a border
  pixel's annulus is genuinely truncated to the real pixels that exist,
  never a fabricated or duplicated one.
- No explicit matrix inverse is formed: `np.linalg.solve` on the batched
  K x K systems (K = outer**2 - inner**2) for the fast path, or per-pixel
  for the slow path, analogous to global_rx's cho_factor/cho_solve on a
  single system.
- NaN handling per D15: a pixel whose OWN value is NaN scores NaN, no
  computation attempted. A pixel whose neighbourhood contains one or more
  NaN neighbours -- nodata mixed into the annulus, OR simply an
  image-border annulus that runs off the edge into the NaN padding above --
  is NEVER fed through matmul/einsum with the NaN present: it is masked out
  of the fast batched path and recomputed on a per-pixel slow path that
  gathers only the valid neighbours first. `0.0 * NaN == NaN` silently
  poisons a whole batch if this masking is skipped (D15) -- the two-path
  split exists specifically to avoid that.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Tiny numerical nugget added to every system's diagonal, independent of
# `lam`. Gram + lam*diag(dist**2) is PD (hence invertible) for ANY lam > 0
# as long as every neighbour distance is > 0 -- but reflect-padding a
# window that is large relative to a corner pixel's distance from the image
# edge can, geometrically, mirror a pixel's own value back into its own
# annulus (distance exactly 0.0), which would otherwise make that one
# diagonal entry -- and, in degenerate cases, the whole system -- singular.
# This nugget is many orders below any meaningful lam*dist**2 term, so it
# does not perturb the lam -> inf collapse test.
_NUGGET = 1e-9


def _annulus_mask(outer: int, inner: int) -> np.ndarray:
    """Boolean [outer, outer] mask: True in the annulus, False in the
    centered inner x inner guard box (and, trivially, nowhere outside the
    outer box since the array itself IS the outer box)."""
    mask = np.ones((outer, outer), dtype=bool)
    offset = (outer - inner) // 2
    mask[offset:offset + inner, offset:offset + inner] = False
    return mask


def _solve_one(neighbours: np.ndarray, y: np.ndarray, lam: float) -> float:
    """Slow path: neighbours [K_i, B] (already NaN-free, K_i may be < the
    full annulus size), y [B]. Returns the CRD residual norm for one pixel.
    K_i == 0 has no representation available; the well-defined limit of the
    model (lam -> inf, or equivalently "no usable neighbours") is w == 0,
    i.e. the score collapses to ||y||_2 -- see kw `lam` docs below.
    """
    if neighbours.shape[0] == 0:
        return float(np.linalg.norm(y))
    dist = np.linalg.norm(y[None, :] - neighbours, axis=-1)          # [K_i]
    gram = neighbours @ neighbours.T                                  # [K_i, K_i]
    a = gram + lam * np.diag(dist ** 2) + _NUGGET * np.eye(neighbours.shape[0])
    rhs = neighbours @ y                                               # [K_i]
    w = np.linalg.solve(a, rhs)
    residual = y - w @ neighbours
    return float(np.linalg.norm(residual))


def crd(cube: np.ndarray, *, inner: int = 5, outer: int = 15, lam: float = 1e-2) -> np.ndarray:
    """Collaborative Representation Detector.

    cube: [H, W, B] float32, NaN = nodata.
    returns: [H, W] float32, NaN wherever the input pixel was NaN.

    lam -> inf drives w -> 0 (the lam * Gamma^T Gamma term swamps X^T X), so
    the score collapses to ||y||_2 -- the sanity check that the regularizer
    is wired the right way round (large lam = trust neighbours LESS, not
    more).
    """
    assert outer > inner > 0, "outer window must be strictly larger than the inner guard"
    assert outer % 2 == 1 and inner % 2 == 1, (
        "outer/inner must be odd so the annulus is exactly centered on the "
        "pixel (reflect-padding below assumes outer == 2*(outer//2)+1)")

    h, w, b = cube.shape
    flat = cube.reshape(-1, b).astype(np.float64)
    valid = ~np.any(np.isnan(flat), axis=-1)
    valid_grid = valid.reshape(h, w)

    scores = np.full((h, w), np.nan, dtype=np.float64)
    if not valid_grid.any():
        return scores.astype(np.float32)

    ro = outer // 2
    cube64 = cube.astype(np.float64)
    # NaN/False padding, not reflect/edge -- see module docstring: reflecting
    # real data can fold a border pixel's own value back into its own
    # annulus. NaN padding instead falls straight into the same
    # real-neighbours-only slow path used for interior nodata.
    padded = np.pad(cube64, ((ro, ro), (ro, ro), (0, 0)), mode="constant", constant_values=np.nan)
    padded_valid = np.pad(valid_grid, ((ro, ro), (ro, ro)), mode="constant", constant_values=False)

    # `windows`/`valid_windows` are stride-tricks VIEWS -- O(1) extra memory,
    # not O(H*W*K*B). Materializing the annulus for every pixel at once
    # (windows[:, :, ann, :] over ALL H,W) was tried first and OOM-killed the
    # process on a 150x150x188 ABU scene at default outer=15 (K=200): that
    # single array alone is H*W*K*B*8 bytes ~= 6.8 GB. Everything below
    # gathers only a bounded CHUNK of pixels' neighbourhoods at a time.
    windows = sliding_window_view(padded, (outer, outer, b))[:, :, 0, :, :, :]     # [H, W, outer, outer, B] view
    valid_windows = sliding_window_view(padded_valid, (outer, outer))               # [H, W, outer, outer] view

    ann = _annulus_mask(outer, inner)
    k = int(ann.sum())
    neighbourhood_valid = valid_windows[:, :, ann]    # [H, W, K] bool -- cheap (1 byte/elt)

    all_neighbours_valid = neighbourhood_valid.all(axis=-1)    # [H, W]
    fast_mask = valid_grid & all_neighbours_valid
    slow_mask = valid_grid & ~all_neighbours_valid

    # --- fast, vectorized (but memory-bounded) path -----------------------
    # Chunk size keeps the dominant per-chunk array (the [m, K, K] Gram/system
    # matrix, or [m, K, B] neighbour tensor if B > K) under ~200 MB -- this
    # machine has no swap and ~8 GB available (CONTRIBUTING.md).
    _BUDGET_BYTES = 200_000_000
    per_pixel_bytes = 8 * k * max(k, b)
    chunk = max(1, min(int(_BUDGET_BYTES // per_pixel_bytes), 20_000))

    fast_rows, fast_cols = np.nonzero(fast_mask)
    eye_k = np.eye(k)
    for start in range(0, fast_rows.size, chunk):
        rc = fast_rows[start:start + chunk]
        cc = fast_cols[start:start + chunk]

        x = windows[rc, cc][:, ann, :]                             # [m, K, B]
        y = cube64[rc, cc]                                          # [m, B]

        gram = np.einsum("mkb,mlb->mkl", x, x)                     # [m, K, K] == X^T X
        dist = np.linalg.norm(y[:, None, :] - x, axis=-1)          # [m, K]
        a = gram + lam * (dist ** 2)[:, :, None] * eye_k[None, :, :] + _NUGGET * eye_k[None, :, :]
        rhs = np.einsum("mkb,mb->mk", x, y)                        # [m, K] == X^T y

        wgt = np.linalg.solve(a, rhs[..., None])[..., 0]            # [m, K]
        recon = np.einsum("mkb,mk->mb", x, wgt)                     # [m, B]
        scores[rc, cc] = np.linalg.norm(y - recon, axis=-1)

    # --- slow, per-pixel path: annulus has >= 1 NaN neighbour to mask out --
    slow_rows, slow_cols = np.nonzero(slow_mask)
    for r, c in zip(slow_rows, slow_cols):
        nb_full = windows[r, c]                        # [outer, outer, B] view
        keep = neighbourhood_valid[r, c]                # [K] bool
        nb = nb_full[ann][keep]                          # [K_i, B], NaN-free by construction
        scores[r, c] = _solve_one(nb, cube64[r, c], lam)

    return scores.astype(np.float32)
