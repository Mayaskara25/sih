"""ui/preview.py -- false-color RGB + ROI overlay derived from the run's
manifest bboxes (D35: preview and export cannot disagree with the record).
"""
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from ui.preview import _pick_bands, render_preview


def _make_scene(tmp_path: Path, h=40, w=50, b=224) -> Path:
    path = tmp_path / "scene.tif"
    rng = np.random.default_rng(3)
    cube = rng.normal(scale=0.1, size=(b, h, w)).astype(np.float32)
    transform = rasterio.Affine(10.0, 0, 500_000.0, 0, -10.0, 4_480_000.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=b, dtype="float32",
        crs="EPSG:32616", transform=transform,
    ) as ds:
        ds.write(cube)
    return path


def test_pick_bands_within_bounds_and_distinct():
    for b in (3, 7, 224):
        i, j, k = _pick_bands(b)
        assert 0 <= i < b and 0 <= j < b and 0 <= k < b
        assert len({i, j, k}) == 3


def test_render_full_scene_with_boxes(tmp_path):
    scene = _make_scene(tmp_path)
    rois = [dict(roi_id="scene:anomaly:0000", bbox=[5, 6, 14, 20]),
            dict(roi_id="scene:anomaly:0001", bbox=[30, 40, 38, 48])]
    out = render_preview(scene, None, rois, tmp_path / "preview.png")
    assert out.exists()
    img = Image.open(out)
    assert img.size == (50, 40)


def test_render_window_shifts_boxes(tmp_path):
    """A ROI outside the rendered window must be skipped, not crash; an inside
    one is drawn at window coordinates (bbox shifted by the window offset)."""
    scene = _make_scene(tmp_path)
    rois = [dict(roi_id="s:1", bbox=[2, 3, 8, 9]),      # inside window
            dict(roi_id="s:2", bbox=[100, 100, 110, 110])]  # outside
    out = render_preview(scene, (10, 10, 20, 25), rois, tmp_path / "preview_win.png")
    assert out.exists()
    assert Image.open(out).size == (25, 20)


def test_render_handles_empty_rois(tmp_path):
    scene = _make_scene(tmp_path, h=16, w=16, b=4)
    out = render_preview(scene, None, [], tmp_path / "preview_empty.png")
    assert Image.open(out).size == (16, 16)
