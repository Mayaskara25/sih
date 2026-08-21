"""D9/D13.4/D20 -- nearest-wavelength band selection never guesses a band."""
import affine
import numpy as np
import pytest
import rasterio.crs

from core.contracts import SceneMeta
from preprocessing.bands import select_band, select_bands


def _meta(wavelengths, *, source="had100", scene_id="s1"):
    b = 0 if wavelengths is None else len(wavelengths)
    return SceneMeta(
        scene_id=scene_id,
        crs=rasterio.crs.CRS.from_epsg(32615),
        transform=affine.Affine(10.0, 0, 0, 0, -10.0, 0),
        wavelengths=wavelengths,
        bad_bands=np.zeros(b, dtype=bool),
        gsd_m=10.0,
        source=source,
        georef="real",
    )


def test_select_band_returns_nearest_index():
    wl = np.array([450.0, 550.0, 660.0, 860.0, 1650.0, 2200.0], dtype=np.float32)
    meta = _meta(wl)
    assert select_band(meta, 660.0) == 2          # exact match
    assert select_band(meta, 665.0) == 2           # 5nm off, nearest is red
    assert select_band(meta, 858.0) == 3           # nearest is nir
    assert select_band(meta, 445.0) == 0           # nearest is the shortest band


def test_select_band_raises_when_no_wavelength_array():
    """ABU/HYDICE/Indian Pines case (D13.4) -- must raise, never return a
    bare index for an unverifiable scene."""
    meta = _meta(None, source="abu")
    with pytest.raises(ValueError, match="wavelengths"):
        select_band(meta, 660.0)


def test_select_band_raises_beyond_tolerance():
    wl = np.array([450.0, 550.0, 660.0], dtype=np.float32)
    meta = _meta(wl)
    # 660 is the nearest available band to 900nm, but 240nm away -- far
    # outside any sensible tolerance, must raise rather than silently
    # returning the edge band.
    with pytest.raises(ValueError, match="tol_nm"):
        select_band(meta, 900.0, tol_nm=15.0)


def test_select_band_accepts_within_explicit_tolerance():
    wl = np.array([450.0, 550.0, 660.0], dtype=np.float32)
    meta = _meta(wl)
    # Same 240nm gap, but explicitly widened tolerance is honoured -- the
    # raise is a default-tolerance decision, not an unconditional refusal.
    assert select_band(meta, 900.0, tol_nm=300.0) == 2


def test_select_bands_raises_on_any_unresolvable_wavelength():
    wl = np.array([450.0, 550.0, 660.0], dtype=np.float32)
    meta = _meta(wl)
    with pytest.raises(ValueError):
        select_bands(meta, [450.0, 5000.0])


def test_select_bands_returns_indices_in_request_order():
    wl = np.array([450.0, 550.0, 660.0, 860.0], dtype=np.float32)
    meta = _meta(wl)
    assert select_bands(meta, [860.0, 450.0, 660.0]) == [3, 0, 2]
