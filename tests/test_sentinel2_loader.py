"""Sentinel-2 loader extension -- companion to test_loader.py / test_enmap_loader.py.

Covers the tag-parsing path `preprocessing.raster_loader` gained for
`source="sentinel2"` + a `*_stack.tif` filename (PLAN.md O5/§8 Level 3), and
`load_sentinel2_scl`. Dispatch is keyed on the filename SUFFIX, not just
`source == "sentinel2"` -- verified here by
`test_generic_sentinel2_source_tif_without_stack_suffix_is_unaffected`, the
regression guard for every pre-existing `.tif` test.

Fixtures are built in-process (rasterio + real GDAL tags), named exactly like
a real `scripts/fetch_sentinel2.py` output so the dispatch condition is
genuinely exercised -- not a generic-`.tif` fixture relabelled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from core.contracts import validate_mask, validate_scene
from preprocessing import raster_loader
from preprocessing.cloud_mask import cloud_shadow_mask

PRODUCT_NAME = "S2B_MSIL2A_20260101T052649_N0512_R105_T43RGM_20260101T091136.SAFE"


def _write_stack_tif(path: Path, *, h: int = 6, w: int = 5,
                      wavelengths=(492.3, 559.0, 665.0, 864.0, 1610.4, 2185.7),
                      sensing_time="2026-01-01T05:41:00.000000Z",
                      fill_border: bool = True, extra_tags: dict | None = None,
                      omit_tags: tuple[str, ...] = ()) -> None:
    n_bands = len(wavelengths)
    rng = np.random.default_rng(0)
    cube = rng.integers(1, 4000, size=(n_bands, h, w)).astype(np.uint16)
    if fill_border:
        cube[:, 0, :] = 0
        cube[:, :, 0] = 0
    transform = rasterio.Affine(20.0, 0, 753260.0, 0, -20.0, 3121800.0)

    tags = {
        "SOURCE": "sentinel2", "PRODUCT_ID": "fake-id", "PRODUCT_NAME": PRODUCT_NAME,
        "SENSING_TIME": sensing_time,
        "PRODUCT_START_TIME": "2026-01-01T05:26:49.024Z",
        "PROCESSING_BASELINE": "05.12",
        "BOA_QUANTIFICATION_VALUE": "10000.0",
        "BOA_ADD_OFFSET_DISTINCT": "[-1000.0]",
        "BOA_ADD_OFFSET_UNIFORM": "True",
        "BAND_ORDER": "B02,B03,B04,B8A,B11,B12",
        "WAVELENGTHS_NM": ",".join(repr(w_) for w_ in wavelengths),
        "RESOLUTION_M": "20.0",
    }
    for k in omit_tags:
        tags.pop(k, None)
    if extra_tags:
        tags.update(extra_tags)

    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=n_bands, dtype="uint16",
        crs="EPSG:32643", transform=transform, nodata=0,
    ) as ds:
        ds.write(cube)
        ds.update_tags(**tags)


def _write_scl_tif(path: Path, *, h: int = 6, w: int = 5, value: int = 4) -> None:
    transform = rasterio.Affine(20.0, 0, 753260.0, 0, -20.0, 3121800.0)
    scl = np.full((h, w), value, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8",
        crs="EPSG:32643", transform=transform,
    ) as ds:
        ds.write(scl, 1)


# --- happy path --------------------------------------------------------------

def test_sentinel2_stack_loads_with_wavelengths_and_acquired(tmp_path):
    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    _write_stack_tif(stack)

    cube, meta = raster_loader.load_scene(stack, source="sentinel2")

    assert cube.shape == (6, 5, 6)
    assert cube.dtype == np.float32
    assert meta.source == "sentinel2"
    assert meta.georef == "real"
    assert meta.gsd_m == 20.0
    assert meta.wavelengths is not None
    assert meta.wavelengths.shape == (6,)
    assert np.all(np.diff(meta.wavelengths) > 0)
    np.testing.assert_allclose(meta.wavelengths[0], 492.3, atol=1e-3)

    # D33: acquired must be the real per-tile sensing time, and it must not
    # equal (nor silently fall back to) a run-time / start-time stand-in.
    assert meta.acquired == "2026-01-01T05:41:00.000000Z"
    assert meta.acquired != "2026-01-01T05:26:49.024Z"  # PRODUCT_START_TIME, wrong source

    # fill border (DN 0) -> NaN, never survives as 0 or wraps to a sentinel
    assert np.isnan(cube[0, 0, 0])
    assert np.isnan(cube[0, 0, :]).all()

    validate_scene(cube, meta)


def test_sentinel2_scl_loads_matching_grid_and_feeds_cloud_mask(tmp_path):
    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    scl_path = tmp_path / f"{PRODUCT_NAME}_scl.tif"
    _write_stack_tif(stack, fill_border=False)
    _write_scl_tif(scl_path, value=8)  # cloud medium probability

    cube, meta = raster_loader.load_scene(stack, source="sentinel2")
    scl = raster_loader.load_sentinel2_scl(stack)

    assert scl.dtype == np.uint8
    assert scl.shape == cube.shape[:2]

    mask = cloud_shadow_mask(cube, meta, scl=scl)
    validate_mask(mask)
    assert mask.all(), "every pixel is SCL class 8 (cloud) -- mask must be all 1s"


def test_generic_sentinel2_source_tif_without_stack_suffix_is_unaffected(tmp_path):
    """Dispatch is keyed on the filename suffix, not just source=='sentinel2'
    -- a plain .tif (e.g. an intermediate product, or a pre-existing fixture)
    must load exactly as it did before this extension existed: no tag lookup,
    wavelengths/acquired stay None."""
    plain = tmp_path / "not_a_stack_file.tif"
    transform = rasterio.Affine(20.0, 0, 0.0, 0, -20.0, 0.0)
    with rasterio.open(plain, "w", driver="GTiff", height=4, width=4, count=3,
                       dtype="uint16", crs="EPSG:32643", transform=transform) as ds:
        ds.write(np.ones((3, 4, 4), dtype=np.uint16))

    cube, meta = raster_loader.load_scene(plain, source="sentinel2")
    assert meta.wavelengths is None
    assert meta.acquired is None
    validate_scene(cube, meta)


# --- must-fail paths (task 3's explicit requirement: fail on a missing offset/
#     quantification-equivalent field -- here, the tags that stand in for it) --

def test_missing_wavelengths_tag_raises(tmp_path):
    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    _write_stack_tif(stack, omit_tags=("WAVELENGTHS_NM",))
    with pytest.raises(ValueError, match="WAVELENGTHS_NM"):
        raster_loader.load_scene(stack, source="sentinel2")


def test_missing_sensing_time_tag_raises_not_silently_none(tmp_path):
    """D33: this must FAIL, not return acquired=None and let a well-formed-
    string check pass for the wrong reason (the exact way D33 survived
    undetected for every other source until EnMAP)."""
    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    _write_stack_tif(stack, omit_tags=("SENSING_TIME",))
    with pytest.raises(ValueError, match="SENSING_TIME"):
        raster_loader.load_scene(stack, source="sentinel2")


def test_wavelength_band_count_mismatch_raises(tmp_path):
    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    _write_stack_tif(stack, extra_tags={"WAVELENGTHS_NM": "492.3,559.0,665.0"})  # 3, not 6
    with pytest.raises(ValueError, match="6 bands"):
        raster_loader.load_scene(stack, source="sentinel2")


def test_non_ascending_wavelengths_raises(tmp_path):
    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    _write_stack_tif(stack, extra_tags={
        "WAVELENGTHS_NM": "492.3,559.0,665.0,864.0,1610.4,900.0"})  # last one out of order
    with pytest.raises(ValueError, match="ascending"):
        raster_loader.load_scene(stack, source="sentinel2")


def test_load_sentinel2_scl_missing_sibling_raises(tmp_path):
    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    _write_stack_tif(stack)
    # no *_scl.tif written
    with pytest.raises(FileNotFoundError):
        raster_loader.load_sentinel2_scl(stack)


# --- D33 end-to-end: meta.acquired must actually resolve correctly through
#     geospatial.geojson._resolve_timestamp, not just be a well-formed string
#     (a well-formed-but-wrong string is exactly how D33 went unnoticed). ---

def test_acquired_resolves_through_geojson_timestamp_logic(tmp_path):
    import dataclasses
    from datetime import datetime, timezone

    from geospatial.geojson import _resolve_timestamp

    stack = tmp_path / f"{PRODUCT_NAME}_stack.tif"
    _write_stack_tif(stack, sensing_time="2026-06-17T05:41:01.854886Z")
    _, meta = raster_loader.load_scene(stack, source="sentinel2")

    resolved = _resolve_timestamp(None, meta)
    assert resolved == "2026-06-17T05:41:01Z"

    # Must be the real tile SENSING_TIME, not a run-time stand-in: a scene
    # with acquired=None falls back to now() (a different code path in
    # _resolve_timestamp -- D33's bug), which cannot equal the fixed 2026
    # sensing time under any real clock.
    no_acquired = dataclasses.replace(meta, acquired=None)
    fallback = _resolve_timestamp(None, no_acquired)
    assert fallback != resolved
    assert datetime.now(timezone.utc).year >= 2026  # sanity: "now" is not 2026-06-17
