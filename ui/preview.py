"""False-color preview with ROI boxes -- rendered from what the run wrote.

D35: the preview derives from the run's own outputs so the picture cannot
disagree with the record. The background RGB is read straight from the scene
GeoTIFF (3 bands only -- a few MB, never the whole cube) through the same
window the pipeline used; the boxes come from run_manifest.json's per-ROI
pixel bboxes, whose roi_ids key into the exported GeoJSON.

No tkinter import at module level (headless-testable, docs/ui_plan.md section 5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

MAX_BOXES = 40
BOX_PAD = 3


def _pick_bands(band_count: int) -> tuple[int, int, int]:
    """Three spread bands for a false-color preview (evenly spaced across the
    cube). Deliberately index-based (the reference sih2.py.py approach):
    EnMAP COGs carry no per-band wavelength tags (D32), so nothing better
    exists without parsing the sidecar, and this is visualization only."""
    if band_count < 3:
        raise ValueError(f"need >= 3 bands for an RGB preview, got {band_count}")
    i, j, k = np.linspace(0, band_count - 1, 3).astype(int)
    return int(i), int(j), int(k)


def _stretch(channel: np.ndarray) -> np.ndarray:
    valid = channel[np.isfinite(channel)]
    if valid.size == 0:
        return np.zeros_like(channel)
    lo, hi = np.percentile(valid, 2), np.percentile(valid, 98)
    out = np.clip((channel - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    out[~np.isfinite(channel)] = 0.0
    return out


def render_preview(scene_path: str | Path,
                   window: tuple[int, int, int, int] | None,
                   manifest_rois: list[dict], out_png: str | Path,
                   max_boxes: int = MAX_BOXES) -> Path:
    """Render `<scene>[window]` as false-color RGB with ROI boxes drawn.

    `window` = (row_off, col_off, height, width) -- the SAME tuple passed to
    run_pipeline; None means the whole scene. ROI bboxes in the manifest are
    whole-scene pixel coordinates and are shifted into window coordinates.
    """
    scene_path = Path(scene_path)
    with rasterio.open(scene_path) as ds:
        band_idx = _pick_bands(ds.count)
        if window is None:
            raw = ds.read(list(i + 1 for i in band_idx)).astype(np.float32)
        else:
            r0, c0, h, w = (int(v) for v in window)
            win = rasterio.windows.Window(col_off=c0, row_off=r0, width=w, height=h)
            raw = ds.read([i + 1 for i in band_idx], window=win).astype(np.float32)

    rgb = np.moveaxis(raw, 0, -1)
    rgb = np.dstack([_stretch(rgb[:, :, k]) for k in range(3)])
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    r_off = 0 if window is None else int(window[0])
    c_off = 0 if window is None else int(window[1])
    for roi in manifest_rois[:max_boxes]:
        r_0, c_0, r_1, c_1 = roi["bbox"]
        x1 = max(0, c_0 - c_off - BOX_PAD)
        y1 = max(0, r_0 - r_off - BOX_PAD)
        x2 = min(img.width - 1, c_1 - c_off + BOX_PAD)
        y2 = min(img.height - 1, r_1 - r_off + BOX_PAD)
        if x2 <= x1 or y2 <= y1:
            continue  # ROI lies outside the rendered window
        label = str(roi.get("roi_id", "?")).rsplit(":", 1)[-1]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(255, 255, 255))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png
