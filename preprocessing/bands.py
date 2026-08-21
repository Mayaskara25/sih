"""D9/D10/D13.4 -- nearest-wavelength band selection.

No caller may select a spectral band by its integer index: band count and
band order differ per sensor (D9 -- AVIRIS-NG 425 bands vs. AVIRIS-Classic
224, ABU alone spans seven distinct counts: 205/204/193/191/188/102), so
"band 30" names a different physical wavelength on every scene. This module
is the one place a requested wavelength (nm) becomes a band index, and it
raises rather than guesses whenever the answer cannot be verified against the
scene's own wavelength array:

  * `meta.wavelengths is None` -- ABU, HYDICE and Indian Pines ship no
    wavelength array at all (D13.4/D20). There is nothing to search, so any
    index this module returned would be a silent guess wearing a real
    number's clothes -- exactly the "resample by band index" failure mode
    D13.4 rejected as the most dangerous of its three options.
  * the nearest available band is farther than `tol_nm` from the request --
    same failure mode, milder: a sensor that simply does not cover the
    requested wavelength (a truncated VNIR-only cube, a coarse-step scene)
    would otherwise silently hand back its edge band as if it were the
    requested one.

Both raises are the correct behaviour, not an inconvenience to work around:
callers (`anomaly.scoring.spectral_index_score`, D20's `fuse_scores`) are
expected to catch the absence and drop the component, never to widen the
tolerance until the exception goes away.
"""
from __future__ import annotations

import numpy as np

from core.contracts import SceneMeta

# ~1.5x AVIRIS/HAD100's native ~10 nm step (D9's canonical grid is also
# 10 nm). Wide enough that ordinary sensor sampling always finds a band;
# narrow enough that a genuinely uncovered wavelength (edge-of-range,
# missing window) still raises instead of silently returning the nearest
# edge band.
DEFAULT_TOL_NM = 15.0


def select_band(meta: SceneMeta, wavelength_nm: float, *, tol_nm: float = DEFAULT_TOL_NM) -> int:
    """Nearest-wavelength band index lookup against `meta.wavelengths`.

    Parameters
    ----------
    meta : SceneMeta
        Must carry a real `wavelengths` array (C1); see module docstring for
        why `None` is refused rather than papered over.
    wavelength_nm : float
        Requested wavelength, nanometres.
    tol_nm : float
        Maximum allowed distance between `wavelength_nm` and the nearest
        band actually present, nanometres.

    Returns
    -------
    int
        Index into the band axis of a cube described by `meta`.

    Raises
    ------
    ValueError
        If `meta.wavelengths` is None, or the nearest band exceeds `tol_nm`.
    """
    if meta.wavelengths is None:
        raise ValueError(
            f"{meta.scene_id}: select_band requires meta.wavelengths, but source="
            f"{meta.source!r} ships none (D13.4 -- ABU, HYDICE and Indian Pines have "
            "no wavelength array; selecting a band by index instead is exactly what "
            "this module exists to prevent, see D20)")

    wl = np.asarray(meta.wavelengths, dtype=np.float64)
    if wl.size == 0:
        raise ValueError(f"{meta.scene_id}: meta.wavelengths is empty")

    idx = int(np.argmin(np.abs(wl - wavelength_nm)))
    delta = abs(float(wl[idx]) - wavelength_nm)
    if delta > tol_nm:
        raise ValueError(
            f"{meta.scene_id}: nearest band to {wavelength_nm:.1f} nm is "
            f"{float(wl[idx]):.1f} nm ({delta:.1f} nm away, tol_nm={tol_nm}) -- "
            "refusing to return a band this far from the request; this scene does "
            "not verifiably cover the requested wavelength")
    return idx


def select_bands(meta: SceneMeta, wavelengths_nm: list[float], *,
                  tol_nm: float = DEFAULT_TOL_NM) -> list[int]:
    """Vectorized convenience wrapper: one `select_band` call per wavelength.

    Raises on the first unresolvable wavelength, same as `select_band` --
    there is no partial-success mode, because a caller building a multi-band
    index (D6's `ndbi`, `bsi`, ...) needs every band or none.
    """
    return [select_band(meta, wl, tol_nm=tol_nm) for wl in wavelengths_nm]
