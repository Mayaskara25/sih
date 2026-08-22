"""Plan §3C.7 -- cloud / shadow / cirrus mask (C3 convention, feeds `c_clear`
in the D4 confidence).

Two paths:

* **Sentinel-2** reads the scene classification layer (SCL), which this
  codebase's loader will deliver as a separate [H,W] uint8 array (keyword-only
  `scl`). Classes 3 (cloud shadow), 8 (cloud medium probability), 9 (cloud
  high probability) and 10 (thin cirrus) map to 1; everything else to 0.
  A Sentinel-2 scene without an SCL array raises -- falling back to spectral
  thresholds on a 13-band MSI grid would silently reclassify the whole
  branch's S2 scenes with heuristics tuned for hyperspectral sensors.
* **Hyperspectral fallback** (`method="spectral"`, or `"auto"` when no SCL is
  supplied) thresholds three physical cues: bright blue (~450 nm), dark SWIR
  (~1600 nm) and low NDVI (red ~665 nm / NIR ~842 nm). Bands are located by
  nearest wavelength, never by index (D9/D13.4); a scene that ships no
  wavelengths at all (ABU / HYDICE, D13) raises rather than guessing.

NaN policy (deliberate): a NaN pixel can never satisfy all three threshold
tests, so it lands on mask value 0 ("clear"). This is *not* a claim that NaN
is clear -- nodata is not cloud -- it simply keeps the C3 mask binary and
lets the score raster's own NaN exclude the pixel downstream. Consumers must
combine this mask with a finite-score check, never use it alone as validity.
"""
from __future__ import annotations

import numpy as np

from core.contracts import SceneMeta, validate_mask

# Sentinel-2 SCL classes counted as cloud/shadow/cirrus (plan §3C.7).
_SCL_CLOUDY = frozenset({3, 8, 9, 10})

_VALID_METHODS = frozenset({"auto", "spectral"})


def _nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    """Index of the band closest to `target_nm` in a raw wavelength array.

    Mirrors preprocessing.bands.select_band but takes the array directly --
    callers here have already established that wavelengths exist.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    return int(np.argmin(np.abs(wl - target_nm)))


def cloud_shadow_mask(
    cube: np.ndarray,
    meta: SceneMeta,
    *,
    method: str = "auto",
    scl: np.ndarray | None = None,
    blue_nm: float = 450.0,
    swir_nm: float = 1600.0,
    red_nm: float = 665.0,
    nir_nm: float = 842.0,
    blue_thr: float = 0.25,
    swir_thr: float = 0.20,
    ndvi_thr: float = 0.20,
) -> np.ndarray:
    """Return a C3 uint8 [H,W] mask: 1 = cloud/shadow/cirrus, 0 = clear.

    Parameters
    ----------
    cube : np.ndarray
        [H,W,B] float32 reflectance cube (NaN = nodata).
    meta : SceneMeta
        Scene metadata (C1); `meta.wavelengths` is required by the spectral
        path and must be None-free for sources that ship none (D13).
    method : str
        "auto" uses the SCL path when `scl` is provided, otherwise the
        spectral path; "spectral" forces the spectral path.
    scl : np.ndarray | None
        Sentinel-2 scene classification layer, uint8 [H,W]. Required for
        source=="sentinel2".
    blue_nm, swir_nm, red_nm, nir_nm : float
        Target wavelengths (nm) for the spectral path, resolved by nearest
        band against `meta.wavelengths`.
    blue_thr, swir_thr, ndvi_thr : float
        Spectral-path thresholds: cloudy iff
        blue > blue_thr AND swir < swir_thr AND ndvi < ndvi_thr.

    Returns
    -------
    np.ndarray
        uint8 [H,W], values {0,1} (validated with core.contracts.validate_mask).

    Raises
    ------
    ValueError
        If source=="sentinel2" and `scl` is None; if the spectral path runs
        on a scene with `meta.wavelengths is None` (ABU/HYDICE, D13); on
        impossible combinations (`scl` given while method=="spectral",
        unknown method, or shape mismatches).

    Notes
    -----
    NaN pixels are returned as mask 0 by construction of the threshold
    comparisons (a NaN comparison is False). See module docstring for why
    this is safe and what consumers must do about it.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"{meta.scene_id}: method {method!r} not in {sorted(_VALID_METHODS)}")
    if method == "spectral" and scl is not None:
        raise ValueError(
            f"{meta.scene_id}: contradictory request -- method='spectral' "
            "cannot be combined with an SCL array")

    h, w, b = cube.shape

    if scl is not None:
        if meta.source != "sentinel2":
            raise ValueError(
                f"{meta.scene_id}: an SCL array was provided but source="
                f"{meta.source!r} is not 'sentinel2' -- SCL only exists on "
                "Sentinel-2 scenes")
        scl = np.asarray(scl)
        if scl.shape != (h, w):
            raise ValueError(
                f"{meta.scene_id}: scl.shape {scl.shape} != cube spatial "
                f"shape {(h, w)}")
        return np.isin(scl, list(_SCL_CLOUDY)).astype(np.uint8)

    if meta.source == "sentinel2":
        raise ValueError(
            f"{meta.scene_id}: source=='sentinel2' requires the SCL band via "
            "the `scl` argument; refusing to fall back to spectral thresholds "
            "(plan §3C.7)")

    # --- spectral path -------------------------------------------------------
    if meta.wavelengths is None:
        raise ValueError(
            f"{meta.scene_id}: spectral cloud masking requires meta.wavelengths "
            f"to locate the ~{blue_nm:.0f}/{red_nm:.0f}/{nir_nm:.0f}/"
            f"{swir_nm:.0f} nm bands, but source={meta.source!r} ships none "
            "(D13 -- ABU/HYDICE have no wavelength array; masking by band index "
            "would be exactly the guess D13.4 forbids)")
    wl = np.asarray(meta.wavelengths)
    if wl.shape != (b,):
        raise ValueError(
            f"{meta.scene_id}: meta.wavelengths shape {wl.shape} does not "
            f"match cube band count ({b},)")

    blue = cube[:, :, _nearest_band(wl, blue_nm)]
    swir = cube[:, :, _nearest_band(wl, swir_nm)]
    red = cube[:, :, _nearest_band(wl, red_nm)]
    nir = cube[:, :, _nearest_band(wl, nir_nm)]

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
        cloudy = (blue > blue_thr) & (swir < swir_thr) & (ndvi < ndvi_thr)

    # NaN comparisons evaluate False, so NaN pixels never satisfy all three
    # tests and land on mask 0 -- the documented nodata-is-not-cloud choice.
    mask = cloudy.astype(np.uint8)
    validate_mask(mask)
    return mask
