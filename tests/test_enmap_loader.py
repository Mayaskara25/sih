"""EnMAP L2A loader extension -- companion to test_loader.py / test_loader_dtype.py.

Covers the sidecar-parsing path `preprocessing.raster_loader` gained for
`source="enmap"` + a `*-SPECTRAL_IMAGE_COG.TIF` filename (PLAN.md O11/D32).
Dispatch is keyed on the filename SUFFIX, not just `source == "enmap"` --
verified here by `test_generic_enmap_source_tif_without_the_filename_suffix_
is_unaffected`, which is also the regression guard for the two pre-existing
tests (`test_loader.py::test_tif_dispatch_reads_real_crs_and_transform`,
`test_pipeline_e2e.py`) that already call `load_scene(..., source="enmap")`
on synthetic fixtures with no sidecar at all.

Fixtures are built in-process (rasterio + a hand-written METADATA.XML), named
exactly like a real product so the dispatch condition is genuinely exercised
-- not `.mat`/generic-`.tif` fixtures relabelled.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.crs

from core.contracts import validate_scene
from preprocessing import raster_loader

ROOT = Path(__file__).resolve().parents[1]
ENMAP_DIR = ROOT / "data" / "raw" / "enmap"
_real_scenes = sorted(ENMAP_DIR.glob("*-SPECTRAL_IMAGE_COG.TIF")) if ENMAP_DIR.exists() else []
_have_real_enmap = bool(_real_scenes)

SCENE_STEM = "ENMAP01-____L2A-DT0000000000_20260101T000000Z_001_V010000_20260102T000000Z"


def _write_metadata_xml(path: Path, *, n_bands: int, wl_start: float = 418.416,
                         wl_step: float = 9.0, nonmonotonic: bool = False,
                         nan_band: int | None = None, gain: float = 0.0001) -> None:
    """A minimal but structurally real METADATA.XML: <level_X><metadata>
    <bandCharacterisation><bandID number="k"><wavelengthCenterOfBand>...
    Same element names/nesting as the real DLR product (verified against the
    files in data/raw/enmap/), just far fewer other fields.
    """
    bands = []
    for i in range(1, n_bands + 1):
        wl = wl_start + (i - 1) * wl_step
        if nonmonotonic and i == n_bands:
            wl = wl_start  # last band duplicates the first wavelength -> non-ascending
        if nan_band is not None and i == nan_band:
            wl_text = "nan"
        else:
            wl_text = repr(wl)
        bands.append(
            f'      <bandID number="{i}">\n'
            f"        <wavelengthCenterOfBand>{wl_text}</wavelengthCenterOfBand>\n"
            f"        <FWHMOfBand>6.5</FWHMOfBand>\n"
            f"        <GainOfBand>{gain}</GainOfBand>\n"
            f"        <OffsetOfBand>0</OffsetOfBand>\n"
            f"      </bandID>\n"
        )
    xml = (
        '<level_X xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        "  <metadata>\n"
        "    <base>\n"
        "      <startTime>2026-01-01T00:00:00.000000Z</startTime>\n"
        "      <stopTime>2026-01-01T00:00:05.000000Z</stopTime>\n"
        "    </base>\n"
        "    <bandCharacterisation>\n" + "".join(bands) + "    </bandCharacterisation>\n"
        "  </metadata>\n"
        "</level_X>\n"
    )
    path.write_text(xml)
    ET.fromstring(xml)  # fail fast in the test itself if the fixture is malformed


def _write_spectral_tif(path: Path, *, n_bands: int, h: int = 6, w: int = 5,
                         nodata_border: bool = True) -> None:
    rng = np.random.default_rng(0)
    cube = rng.integers(-500, 3000, size=(n_bands, h, w)).astype(np.int16)
    if nodata_border:
        cube[:, 0, :] = -32768
        cube[:, :, 0] = -32768
    transform = rasterio.Affine(30.0, 0, 500_000.0, 0, -30.0, 3_500_000.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=n_bands, dtype="int16",
        crs="EPSG:32643", transform=transform, nodata=-32768,
    ) as ds:
        ds.write(cube)


# --- happy path --------------------------------------------------------------

def test_enmap_synthetic_product_loads_with_real_wavelengths_and_nodata_as_nan(tmp_path):
    n_bands = 10
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML"
    _write_spectral_tif(spectral, n_bands=n_bands)
    _write_metadata_xml(metadata, n_bands=n_bands)

    cube, meta = raster_loader.load_scene(spectral, source="enmap")

    assert cube.shape == (6, 5, n_bands)
    assert cube.dtype == np.float32
    assert meta.georef == "real"
    assert meta.source == "enmap"
    assert meta.wavelengths is not None
    assert meta.wavelengths.shape == (n_bands,)
    assert np.all(np.diff(meta.wavelengths) > 0), "wavelengths must be strictly ascending"
    np.testing.assert_allclose(meta.wavelengths[0], 418.416, atol=1e-3)
    assert meta.acquired == "2026-01-01T00:00:00.000000Z"

    # -32768 nodata border must become NaN, never survive as -32768.0 and
    # never wrap to a large positive value (the D13.2 hazard, re-checked here
    # on the EnMAP dispatch path specifically).
    assert np.isnan(cube[0, 0, 0])
    assert np.isnan(cube[0, 0, :]).all()
    assert not np.any(cube == np.float32(-32768.0))
    assert not np.any(cube[~np.isnan(cube)] > 32767)

    validate_scene(cube, meta)


def test_enmap_metadata_xml_xml_double_suffix_is_found_too(tmp_path):
    """Real products on disk use BOTH -METADATA.XML and -METADATA.XML.XML
    (verified: 7 of 8 downloaded products use the double suffix). Both must
    resolve."""
    n_bands = 4
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML.XML"
    _write_spectral_tif(spectral, n_bands=n_bands)
    _write_metadata_xml(metadata, n_bands=n_bands)

    cube, meta = raster_loader.load_scene(spectral, source="enmap")
    assert meta.wavelengths.shape == (n_bands,)


# --- must-fail paths (task 2's explicit requirements) ------------------------

def test_missing_metadata_sidecar_raises(tmp_path):
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    _write_spectral_tif(spectral, n_bands=4)
    # no METADATA.XML written at all
    with pytest.raises(FileNotFoundError):
        raster_loader.load_scene(spectral, source="enmap")


def test_unparseable_metadata_sidecar_raises(tmp_path):
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML"
    _write_spectral_tif(spectral, n_bands=4)
    metadata.write_text("<level_X><metadata>not closed properly")
    with pytest.raises(ValueError):
        raster_loader.load_scene(spectral, source="enmap")


def test_nonmonotonic_wavelength_axis_raises(tmp_path):
    n_bands = 8
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML"
    _write_spectral_tif(spectral, n_bands=n_bands)
    _write_metadata_xml(metadata, n_bands=n_bands, nonmonotonic=True)
    with pytest.raises(ValueError, match="ascending"):
        raster_loader.load_scene(spectral, source="enmap")


def test_nan_poisoned_wavelength_axis_raises(tmp_path):
    n_bands = 8
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML"
    _write_spectral_tif(spectral, n_bands=n_bands)
    _write_metadata_xml(metadata, n_bands=n_bands, nan_band=4)
    with pytest.raises(ValueError, match="non-finite"):
        raster_loader.load_scene(spectral, source="enmap")


def test_band_count_mismatch_between_sidecar_and_cube_raises(tmp_path):
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML"
    _write_spectral_tif(spectral, n_bands=6)
    _write_metadata_xml(metadata, n_bands=9)   # sidecar disagrees with the cube
    with pytest.raises(ValueError, match="bands"):
        raster_loader.load_scene(spectral, source="enmap")


def test_nodata_is_never_silently_valid(tmp_path):
    """Every pixel in the nodata border must be NaN, and NaN pixels must not
    be counted as valid data by a naive downstream mean/std -- this is the
    concrete failure mode task 2 names ('nodata being silently treated as
    valid data')."""
    n_bands = 5
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML"
    _write_spectral_tif(spectral, n_bands=n_bands, h=8, w=8, nodata_border=True)
    _write_metadata_xml(metadata, n_bands=n_bands)

    cube, meta = raster_loader.load_scene(spectral, source="enmap")
    border_mask = np.zeros((8, 8), dtype=bool)
    border_mask[0, :] = True
    border_mask[:, 0] = True

    assert np.isnan(cube[border_mask]).all(), "nodata border pixels must all be NaN"
    # A naive np.mean (not nanmean) over a nodata-containing cube is NaN --
    # proof nodata cannot be silently averaged in as if it were valid.
    assert np.isnan(np.mean(cube))
    # A nodata-aware reducer must be unaffected by how many border pixels exist.
    assert np.isfinite(np.nanmean(cube))


# --- fully-nodata band detection ---------------------------------------------

def test_band_that_is_nodata_scene_wide_is_marked_bad(tmp_path):
    """A band that is nodata for EVERY pixel poisons standardize()'s per-band
    nanmean/nanvar (NaN for that band, all pixels) and then global_rx's
    any-band-NaN validity mask (every pixel excluded, not just that band) --
    reproduced on a real EnMAP product during the Phase 5 Level 2 run
    (PLAN.md D32: bands 131-135 are 100% nodata in all 8 scenes on disk).
    The loader must mark such a band bad_bands=True so drop_bad_bands removes
    it before either of those runs, rather than leaving every caller to
    rediscover the same failure.
    """
    n_bands = 6
    spectral = tmp_path / f"{SCENE_STEM}-SPECTRAL_IMAGE_COG.TIF"
    metadata = tmp_path / f"{SCENE_STEM}-METADATA.XML"
    h, w = 8, 8
    rng = np.random.default_rng(0)
    cube_raw = rng.integers(-500, 3000, size=(n_bands, h, w)).astype(np.int16)
    cube_raw[2, :, :] = -32768   # band index 2 (0-based) entirely nodata
    transform = rasterio.Affine(30.0, 0, 500_000.0, 0, -30.0, 3_500_000.0)
    with rasterio.open(
        spectral, "w", driver="GTiff", height=h, width=w, count=n_bands, dtype="int16",
        crs="EPSG:32643", transform=transform, nodata=-32768,
    ) as ds:
        ds.write(cube_raw)
    _write_metadata_xml(metadata, n_bands=n_bands)

    cube, meta = raster_loader.load_scene(spectral, source="enmap")
    assert meta.bad_bands.tolist() == [False, False, True, False, False, False]
    assert np.isnan(cube[..., 2]).all()

    # The actual downstream failure this guards against: drop_bad_bands must
    # remove the poisoned band before standardize/global_rx ever see it.
    from preprocessing.normalize import drop_bad_bands, standardize
    clean_cube, clean_meta = drop_bad_bands(cube, meta)
    assert clean_cube.shape[-1] == n_bands - 1
    norm = standardize(clean_cube)
    assert not np.isnan(norm).all(axis=(0, 1)).any(), (
        "no band in the post-drop cube should be entirely NaN")


# --- regression guard: existing generic .tif dispatch must be untouched -----

def test_generic_enmap_source_tif_without_the_filename_suffix_is_unaffected(tmp_path):
    """source='enmap' alone must NOT trigger the sidecar path -- only the
    real product filename suffix does. This is the exact scenario
    tests/test_loader.py::test_tif_dispatch_reads_real_crs_and_transform and
    tests/test_pipeline_e2e.py already exercise; reproduced here as a named
    regression guard so a future change to the dispatch condition fails
    loudly instead of silently breaking those two files.
    """
    path = tmp_path / "not_a_real_enmap_product.tif"
    cube = np.random.default_rng(0).normal(size=(3, 5, 6)).astype(np.float32)
    transform = rasterio.Affine(10.0, 0, 100.0, 0, -10.0, 200.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=5, width=6, count=3, dtype="float32",
        crs="EPSG:32616", transform=transform,
    ) as ds:
        ds.write(cube)

    loaded_cube, meta = raster_loader.load_scene(path, source="enmap")
    assert meta.wavelengths is None
    assert meta.acquired is None
    validate_scene(loaded_cube, meta)


# --- real-file smoke test, guarded ------------------------------------------

@pytest.mark.skipif(not _have_real_enmap, reason="no EnMAP products in data/raw/enmap/")
def test_real_enmap_product_loads_cleanly():
    path = sorted(_real_scenes)[0]
    cube, meta = raster_loader.load_scene(path, source="enmap")
    assert cube.ndim == 3
    assert cube.shape[-1] == 224
    assert meta.georef == "real"
    assert meta.wavelengths is not None
    assert meta.wavelengths.shape == (224,)
    assert np.all(np.diff(meta.wavelengths) > 0)
    assert meta.acquired is not None
    assert meta.crs is not None
    assert np.isnan(cube).any(), "a real EnMAP scene has a nodata border"
    validate_scene(cube, meta)
