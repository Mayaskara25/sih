"""D9 / §3A.1 -- band harmonization. The JOIN that makes the HAD100 background
pool (AVIRIS-NG 425 bands, AVIRIS-Classic 224 bands) a single tensor (D11.3).

Source MUST be the raw ENVI cube (via preprocessing.raster_loader on the
.hdr), never main.py's band-selected output -- band_select leaves interior
holes up to 276 nm wide and 43/184 canonical bands uncovered (D11.6,
re-verified against the files while building this module). This module never
reads files itself, so that choice lives entirely with the caller; what this
module guarantees is that whatever gap survives into it makes harmonize()
raise rather than interpolate across it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.contracts import SceneMeta, validate_scene

CANONICAL_WL = np.arange(400, 2501, 10, dtype=np.float32)     # 211  (verified)
WATER_WINDOWS = [(1350, 1450), (1800, 1950)]                  # endpoints INCLUSIVE
RETAINED_BANDS = 184                                          # 211 - 27  (verified)


def water_mask(wl: np.ndarray = CANONICAL_WL) -> np.ndarray:
    """True where the band is inside a water-absorption window. Endpoints are
    INCLUSIVE -- 1350 and 1450 nm are inside the feature, not clean shoulders.
    Exact equality is safe: every canonical wavelength is exactly
    representable in float32 (verified). See D9.
    """
    m = np.zeros(wl.shape, dtype=bool)
    for lo, hi in WATER_WINDOWS:
        m |= (wl >= lo) & (wl <= hi)
    return m


def sort_spectral_axis(cube: np.ndarray, wl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (cube, wl) with the band axis sorted by ASCENDING wavelength and
    duplicate wavelengths collapsed by mean.

    MANDATORY before any interpolation. AVIRIS's four spectrometers overlap,
    so band index order != wavelength order: 262/262 HAD100 aviris_normal
    headers descend at three seams (D11.4). np.interp requires ascending xp
    and DOES NOT CHECK -- it returns silent garbage across those seams. The
    cube keeps its shape and passes every assertion while being wrong in
    three spectral regions.

    Verified against a real AVIRIS-Classic header: a plain stable sort alone
    is already strictly increasing (min gap 0.21 nm, no exact ties) -- the
    duplicate-collapse path below is defensive for input this module has not
    seen, not something today's real headers exercise.
    """
    wl = np.asarray(wl, dtype=np.float64)
    order = np.argsort(wl, kind="stable")
    wl_sorted = wl[order]
    cube_sorted = np.ascontiguousarray(cube[..., order])

    unique_wl, inverse = np.unique(wl_sorted, return_inverse=True)
    if len(unique_wl) != len(wl_sorted):
        h, w, b = cube_sorted.shape
        flat = cube_sorted.reshape(-1, b).astype(np.float64)
        # One-hot averaging matrix: M[j, k] = 1/count[j] where band k collapses
        # into unique wavelength j. collapsed = flat @ M.T averages duplicates
        # in one vectorized step instead of a Python loop over bands.
        counts = np.bincount(inverse, minlength=len(unique_wl)).astype(np.float64)
        M = np.zeros((len(unique_wl), b), dtype=np.float64)
        M[inverse, np.arange(b)] = 1.0
        M /= counts[:, None]
        collapsed = flat @ M.T
        cube_sorted = collapsed.reshape(h, w, len(unique_wl))
        wl_sorted = unique_wl

    assert np.all(np.diff(wl_sorted) > 0), "spectral axis not strictly increasing"
    return np.ascontiguousarray(cube_sorted.astype(np.float32)), wl_sorted.astype(np.float32)


def coverage_ok(wl_src: np.ndarray, wl_target: np.ndarray, *, tol: float | None = None) -> bool:
    """Every target wavelength has a source band within one median source step.
    False means interpolation would fabricate data across a hole (D11.6).

    wl_src need not be pre-sorted -- sorted internally, so this is safe to
    call standalone (e.g. directly against a raw header's wavelength array)
    as well as from inside harmonize() on an already-sorted axis.
    """
    wl_src_sorted = np.sort(np.asarray(wl_src, dtype=np.float64))
    wl_target = np.asarray(wl_target, dtype=np.float64)
    if tol is None:
        diffs = np.diff(wl_src_sorted)
        tol = float(np.median(diffs)) if len(diffs) else np.inf

    idx = np.searchsorted(wl_src_sorted, wl_target)
    idx = np.clip(idx, 1, len(wl_src_sorted) - 1)
    left = wl_src_sorted[idx - 1]
    right = wl_src_sorted[idx]
    nearest = np.minimum(np.abs(wl_target - left), np.abs(wl_target - right))
    return bool(np.all(nearest <= tol))


def _linear_interp_matrix(target_wl: np.ndarray, wl_src: np.ndarray) -> np.ndarray:
    """[len(target_wl), len(wl_src)] linear-interpolation weight matrix, for
    TESTING / cross-checking against np.interp only -- see harmonize()'s
    docstring for why the production path never multiplies through this
    matrix via matmul. wl_src must be strictly ascending.

    Targets outside [wl_src[0], wl_src[-1]] clamp to the nearest edge band --
    the same constant-extrapolation behaviour np.interp uses by default. This
    is only reached for targets coverage_ok already accepted as within one
    source step of the edge; anything farther is refused before this runs,
    so clamping here is bounded slop, never fabrication across a real gap.
    """
    n_src = len(wl_src)
    W = np.zeros((len(target_wl), n_src), dtype=np.float64)
    idx = np.searchsorted(wl_src, target_wl)
    for j, (t, i) in enumerate(zip(target_wl, idx)):
        if i <= 0:
            W[j, 0] = 1.0
        elif i >= n_src:
            W[j, -1] = 1.0
        else:
            lo, hi = i - 1, i
            frac = (t - wl_src[lo]) / (wl_src[hi] - wl_src[lo])
            W[j, lo] = 1.0 - frac
            W[j, hi] = frac
    return W


def _interpolate_bands(cube_sorted: np.ndarray, wl_src: np.ndarray,
                        target_wl: np.ndarray) -> np.ndarray:
    """[H, W, len(target_wl)] linear interpolation. wl_src must be strictly
    ascending (sort_spectral_axis's output).

    Gathers each target band from AT MOST its two bracketing source bands,
    rather than a dense matmul against the full weight matrix in
    _linear_interp_matrix. This is not a style choice: `0.0 * NaN == NaN` in
    IEEE 754, so a matmul sums every source column including zero-weighted
    ones, and a single NaN band anywhere in a pixel's spectrum poisons every
    output band regardless of that band's actual weight -- silently
    collapsing per-band nodata into whole-pixel nodata. Confirmed by a failing
    test before this was fixed. Gathering only the 1-2 columns each target
    band actually depends on means a NaN band only poisons the (at most two)
    target bands that genuinely reference it.

    Targets outside [wl_src[0], wl_src[-1]] clamp to the nearest edge band
    (matches np.interp's default and _linear_interp_matrix's convention); only
    reached for targets coverage_ok already accepted as within one source
    step of the edge.
    """
    n_src = len(wl_src)
    idx = np.searchsorted(wl_src, target_wl)
    lo_idx = np.clip(idx - 1, 0, n_src - 1)
    hi_idx = np.clip(idx, 0, n_src - 1)

    denom = wl_src[hi_idx] - wl_src[lo_idx]
    safe_denom = np.where(denom > 0, denom, 1.0)   # avoid 0/0 where lo_idx == hi_idx
    frac = np.clip((target_wl - wl_src[lo_idx]) / safe_denom, 0.0, 1.0)

    lo_vals = cube_sorted[..., lo_idx]
    hi_vals = cube_sorted[..., hi_idx]
    blend = lo_vals * (1.0 - frac) + hi_vals * frac
    # Select lo/hi directly (not through the blend) when frac is exactly 0/1,
    # so a NaN in the OTHER (zero-weighted) bracket band can't leak in via
    # `value * 0.0 == NaN` -- the same hazard the matmul path had, at the
    # single-band-pair scale instead of the whole-row scale.
    out = np.where(frac == 1.0, hi_vals, np.where(frac == 0.0, lo_vals, blend))
    return out


def harmonize(cube: np.ndarray, meta: SceneMeta, *, target_wl: np.ndarray = CANONICAL_WL
              ) -> tuple[np.ndarray, SceneMeta]:
    """sort_spectral_axis -> restrict to the RETAINED_BANDS canonical
    wavelengths (water_mask, endpoints inclusive) -> coverage_ok gate ->
    linear interpolation onto that grid.

    This is also the JOIN between sensors. AVIRIS-NG reduces to 276 bands and
    AVIRIS-Classic to 162 under main.py's band_select (D11.3), so pooling the
    two background halves is only legal AFTER this runs -- never before, and
    the two harmonize() outputs stack (D11.3's join, verified below).

    SOURCE IS THE RAW ENVI CUBE, never main.py's band-selected .npy (D11.6):
    band_select leaves interior holes up to 276 nm wide, and np.interp
    bridges them with a straight line instead of raising. This module does
    not enforce that at the file level (it never reads files) -- it enforces
    the CONSEQUENCE: coverage_ok is checked BEFORE interpolating, and raises
    rather than interpolating across a hole, so band-selected input is
    self-defending regardless of how it got here.
    """
    if meta.wavelengths is None:
        raise ValueError(
            f"{meta.scene_id}: harmonize requires meta.wavelengths, source={meta.source!r} "
            "ships none (O8 excludes ABU/HYDICE/Indian Pines from this path)")

    cube_sorted, wl_sorted = sort_spectral_axis(cube, meta.wavelengths)

    retained_mask = ~water_mask(target_wl)
    retained_wl = target_wl[retained_mask]

    if not coverage_ok(wl_sorted, retained_wl):
        raise ValueError(
            f"{meta.scene_id}: canonical band with no source wavelength within one "
            "sensor step -- refusing to interpolate across a hole (D11.6). If this "
            "cube came from main.py's band_select output rather than raw ENVI, that "
            "is the cause.")

    out = _interpolate_bands(cube_sorted.astype(np.float64), wl_sorted.astype(np.float64), retained_wl)
    out = np.ascontiguousarray(out.astype(np.float32))

    new_meta = replace(
        meta,
        wavelengths=retained_wl.astype(np.float32),
        bad_bands=np.zeros(len(retained_wl), dtype=bool),
    )
    validate_scene(out, new_meta)
    return out, new_meta


def reduce_bands(cube: np.ndarray, *, n_components: int = 30, method: str = "pca",
                  fit_on: np.ndarray | object | None = None) -> tuple[np.ndarray, object]:
    """PCA/kPCA band reduction, RETAINED_BANDS (184) -> n_components. Never
    called from harmonize() itself -- fitting requires knowing which pixels
    are the TRAIN split, and harmonize() runs per-scene, before any split
    exists. This is called from segmentation/datasets.py, behind the split
    (D15's amendment: fitting on the whole pool leaks scoring-scene spectral
    statistics into the representation every model is built on).

    fit_on:
      - None: fit fresh on `cube`'s own valid (non-NaN) pixels. Only for a
        one-off transform where sharing a basis across scenes doesn't matter.
      - ndarray [N, B]: fit fresh on this external pixel pool -- e.g. a
        bounded random subsample of the TRAIN split's background patches
        (§3B.3) -- then transform `cube`. This is the call whose returned
        transformer gets pickled to data/processed/.
      - an object with .transform (an already-fitted transformer, e.g.
        unpickled): applied to `cube` WITHOUT refitting. This is what every
        other split/scene (eval, val, real scoring) must use. "Refitting at
        inference is a bug" (§3A.1) is enforced structurally here, not just
        documented: passing the loaded transformer takes this branch and
        never calls .fit again.

    Returns ([H, W, n_components] float32, the transformer). NaN input
    pixels stay NaN in the output; the transformer never sees a NaN row.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    valid = ~np.any(np.isnan(flat), axis=-1)

    if fit_on is not None and hasattr(fit_on, "transform"):
        transformer = fit_on
    else:
        if method == "pca":
            from sklearn.decomposition import PCA
            transformer = PCA(n_components=n_components)
        elif method == "kpca":
            from sklearn.decomposition import KernelPCA
            transformer = KernelPCA(n_components=n_components, kernel="rbf")
        else:
            raise ValueError(f"unknown method {method!r}, expected 'pca' or 'kpca'")
        fit_data = fit_on if fit_on is not None else flat[valid]
        transformer.fit(fit_data)

    out_flat = np.full((flat.shape[0], n_components), np.nan, dtype=np.float32)
    if valid.any():
        out_flat[valid] = transformer.transform(flat[valid]).astype(np.float32)
    return out_flat.reshape(h, w, n_components), transformer
