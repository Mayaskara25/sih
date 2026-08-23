"""Windowed GeoTIFF reads (docs/ui_plan.md section 0.1, plan.md D35).

The UI's OOM guard depends on load_scene(window=...) reading a bounded
sub-window with a REAL window_transform. These tests pin the shape,
georeferencing and failure behaviour of that path.
"""
from pathlib import Path

import numpy as np
import pytest
import rasterio

from preprocessing import raster_loader
from preprocessing.raster_loader import load_scene


@pytest.fixture()
def small_tif(tmp_path: Path) -> Path:
    path = tmp_path / "scene.tif"
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(4, 40, 50)).astype(np.float32)
    transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_480_000.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=40, width=50, count=4, dtype="float32",
        crs="EPSG:32616", transform=transform,
    ) as ds:
        ds.write(cube)
    return path


def test_window_read_shape_and_values(small_tif: Path):
    cube_full, _ = load_scene(small_tif, source="enmap")
    cube_win, meta = load_scene(small_tif, source="enmap", window=(5, 10, 20, 25))
    assert cube_win.shape == (20, 25, 4)
    np.testing.assert_allclose(cube_win, cube_full[5:25, 10:35, :])
    assert meta.scene_id == "scene_win_5_10_20x25"


def test_window_transform_is_real_georeferencing(small_tif: Path):
    """The windowed meta.transform must equal GDAL's own window_transform --
    i.e. world coordinates of window pixels stay correct (D35's 'real
    window_transform' requirement), not an affine re-derived from origin."""
    _, meta = load_scene(small_tif, source="enmap", window=(5, 10, 20, 25))
    with rasterio.open(small_tif) as ds:
        expected = ds.window_transform(rasterio.windows.Window(
            col_off=10, row_off=5, width=25, height=20))
    assert meta.transform == expected
    assert meta.gsd_m == pytest.approx(10.0)


def test_out_of_bounds_window_raises(small_tif: Path):
    with pytest.raises(ValueError, match="does not fit"):
        load_scene(small_tif, source="enmap", window=(30, 0, 20, 10))
    with pytest.raises(ValueError, match="does not fit"):
        load_scene(small_tif, source="enmap", window=(-1, 0, 10, 10))


def test_mat_scenes_reject_window():
    pytest.importorskip("scipy.io")
    with pytest.raises(ValueError, match="only supported for GeoTIFF"):
        load_scene("whatever.mat", source="abu", window=(0, 0, 10, 10))
