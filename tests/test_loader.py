"""§2.1 / §12 test_loader.py -- .mat/.tif/.hdr dispatch, synthetic-affine flag,
tag round-trip, HAD100 .hdr -> georef == "real". D13.2 dtype safety lives in
the dedicated tests/test_loader_dtype.py.
"""
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.crs

from core.contracts import ContractViolation, validate_scene
from preprocessing import raster_loader

ROOT = Path(__file__).resolve().parents[1]
INDIAN_PINES = ROOT / "data" / "benchmark" / "indian_pines" / "Indian_pines_corrected.mat"
HAD100_NG_HDR = (ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data"
                 / "aviris_ng_normal" / "ang20191004t185054_13.hdr")
HAD100_CLASSIC_HDR = (ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data"
                       / "aviris_normal" / "f170507t01p00r10_1.hdr")
ABU_URBAN_2 = ROOT / "data" / "benchmark" / "abu" / "abu-urban-2.mat"

_have_had100 = HAD100_NG_HDR.exists() and HAD100_CLASSIC_HDR.exists()
_have_indian_pines = INDIAN_PINES.exists()
_have_abu = ABU_URBAN_2.exists()


# --- .mat dispatch, synthetic affine (D2) -----------------------------------

@pytest.mark.skipif(not _have_indian_pines, reason="Indian Pines not fetched")
def test_mat_dispatch_indian_pines_is_synthetic_georef():
    cube, meta = raster_loader.load_scene(INDIAN_PINES, source="indian_pines")
    assert cube.shape == (145, 145, 200)
    assert cube.dtype == np.float32
    assert meta.georef == "synthetic"
    assert meta.wavelengths is None          # D13.1: the file ships none
    assert meta.crs == rasterio.crs.CRS.from_string(raster_loader.SYNTHETIC_CRS)
    validate_scene(cube, meta)


@pytest.mark.skipif(not _have_abu, reason="ABU not fetched")
def test_mat_dispatch_abu_int16_negatives_survive_through_load_scene():
    """D13.2 end to end: not just cast_to_float32 in isolation (see
    test_loader_dtype.py) but the actual load_scene() path on a real ABU
    scene with genuine negatives (45626 in abu-urban-2)."""
    cube, meta = raster_loader.load_scene(ABU_URBAN_2, source="abu")
    assert cube.shape == (100, 100, 207)
    assert cube.dtype == np.float32
    n_negative = int(np.sum(cube < 0))
    assert n_negative == 45626
    assert not np.any(cube > 32767), "int16 negatives must not have wrapped to uint16"
    validate_scene(cube, meta)


# --- .hdr dispatch, real HAD100 georeference (D11.5) ------------------------

@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_hdr_dispatch_had100_ng_is_real_georef_with_425_wavelengths():
    cube, meta = raster_loader.load_scene(HAD100_NG_HDR, source="had100")
    assert cube.shape[-1] == 425
    assert meta.georef == "real"
    assert meta.wavelengths is not None
    assert meta.wavelengths.shape == (425,)
    validate_scene(cube, meta)   # NG wavelengths are ascending (D11.4: 0/260 non-monotonic)

    # This scene's 'map info' carries rotation=33 -- a hand-rolled axis-aligned
    # parse of tie-point + pixel-size would silently produce the wrong affine
    # (confirmed while building this loader). Pin against GDAL's own ENVI
    # driver, opened directly on the sibling .dat, as the independent check.
    with rasterio.open(HAD100_NG_HDR.with_suffix(".dat")) as ds:
        assert meta.transform == ds.transform
        assert meta.crs == ds.crs


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_hdr_dispatch_had100_classic_has_224_wavelengths_but_non_monotonic():
    """D11.4: AVIRIS-Classic wavelength arrays are non-monotonic in the raw
    header. The loader parses them as-is (sorting is harmonize.py's job, D9)
    and validate_scene correctly refuses the result until it's been sorted.
    """
    cube, meta = raster_loader.load_scene(HAD100_CLASSIC_HDR, source="had100")
    assert cube.shape[-1] == 224
    assert meta.georef == "real"
    assert meta.wavelengths.shape == (224,)
    assert np.any(np.diff(meta.wavelengths) <= 0), "expected a non-monotonic axis (D11.4)"
    with pytest.raises(ContractViolation):
        validate_scene(cube, meta)

    # This scene's 'map info' carries rotation=90 -- a NORTH-UP axis-aligned
    # affine would have transform.a == pixel_size (nonzero) and transform.b
    # == 0. The true, GDAL-verified transform is the opposite: a 90-degree
    # rotation swaps scale into the off-diagonal entirely.
    assert meta.transform.a == pytest.approx(0.0, abs=1e-9)
    assert meta.transform.b == pytest.approx(16.9, abs=1e-6)
    with rasterio.open(HAD100_CLASSIC_HDR.with_suffix(".dat")) as ds:
        assert meta.transform == ds.transform


# --- .tif dispatch -----------------------------------------------------------

def test_tif_dispatch_reads_real_crs_and_transform(tmp_path):
    path = tmp_path / "scene.tif"
    cube = np.random.default_rng(0).normal(size=(3, 5, 6)).astype(np.float32)
    transform = rasterio.Affine(10.0, 0, 100.0, 0, -10.0, 200.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=5, width=6, count=3, dtype="float32",
        crs="EPSG:32616", transform=transform,
    ) as ds:
        ds.write(cube)

    loaded_cube, meta = raster_loader.load_scene(path, source="enmap")
    assert loaded_cube.shape == (5, 6, 3)
    assert meta.georef == "real"
    assert meta.crs == rasterio.crs.CRS.from_epsg(32616)
    assert meta.transform == transform
    validate_scene(loaded_cube, meta)


# --- save_score_raster round-trip -------------------------------------------

@pytest.mark.skipif(not _have_indian_pines, reason="Indian Pines not fetched")
def test_save_score_raster_tag_roundtrip(tmp_path):
    cube, meta = raster_loader.load_scene(INDIAN_PINES, source="indian_pines")
    score = np.random.default_rng(0).normal(size=cube.shape[:2]).astype(np.float32)
    base = tmp_path / f"{meta.scene_id}_anom"

    raw_path, norm_path = raster_loader.save_score_raster(score, meta, base, method="global_rx")

    assert raw_path.exists() and norm_path.exists()
    with rasterio.open(norm_path) as ds:
        tags = ds.tags()
        assert float(tags["NORM_V_LO"]) == pytest.approx(
            np.percentile(score, 1.0), rel=0, abs=1e-6)
        assert float(tags["NORM_V_HI"]) == pytest.approx(
            np.percentile(score, 99.9), rel=0, abs=1e-6)
        assert tags["SCENE_ID"] == meta.scene_id
        assert tags["GEOREF"] == "synthetic"
