# UI plan — SIH Anomaly Detection desktop app

**Status: planned, not started.** Written 2026-08-23. Reference implementation:
`temp , downloads/sih2.py.py` (a standalone Tkinter EnMAP anomaly GUI that
carries its own RX/PCA/Excel code). This plan adapts its *UI* to this repo; it
deliberately does NOT reuse its science code, because this repo already has a
validated pipeline with contracts and structured outputs.

## 0. Core principle

The reference script re-implements load → standardize → PCA → local RX →
threshold → regions → coordinates.xlsx inside one file. Duplicating that here
would fork the science. Instead the UI is a **thin front-end over
`pipeline/run_pipeline.py`**, which already provides:

- detector registry: `global_rx`, `local_rx`, `kernel_rx`, `crd`, `fused`
  (`pipeline/run_pipeline.py::DETECTORS`)
- stage chain: load → drop_bad_bands → standardize → detector →
  percentile_normalize → threshold → morphology → mask_to_rois → geojson
- contract validation per stage (`core/contracts.py`)
- structured outputs: score rasters, `*_mask.tif`, `*_rois.geojson`,
  `run_manifest.json` (timings, git SHA, package versions)

## 1. Scope (v1 — agreed)

Single-window **anomaly detection only**: pick a scene, choose detector +
threshold, run, inspect preview + log, export `coordinates.xlsx`.
Segmentation (U-Net), change detection and edge benchmarks are v2+ tabs.

## 2. New files

All UI code lives in a new `ui/` package — no overlap with files other agents
are editing.

### `ui/app.py` — main window
Layout mirrors `sih2.py.py::EnMAPAnomalyGUI` (proven pattern):
- header label + one-line pipeline description
- scene picker row: Entry + "Choose scene" dialog, defaulting to
  `data/raw/enmap`; source auto-detected from filename prefix
  (`ENMAP01-*` → enmap, HAD100 hdr → had100, S2 naming → sentinel-2)
- params row: detector dropdown (from `DETECTORS.keys()`), threshold-% slider,
  normalize method (`standardize`/`l2`), profile (`object`/`landcover`),
  free-form detector-params JSON box (same contract as CLI `--detector-params`)
- Run button → disables itself, spawns daemon `threading.Thread`
- `ttk.Progressbar` + status StringVar
- bottom `ttk.Panedwindow`: Processing Log (`tk.Text`) left, annotated preview
  right
- all worker→UI updates marshalled via `root.after()` exactly as in the
  reference (`ui_log`, `ui_progress`, `finish_success`, `finish_error`)

### `ui/excel_export.py` — coordinates.xlsx writer
Port of the reference's `save_coordinates_excel` (3 sheets: Anomaly Regions /
Anomaly Pixels / Metadata; bold blue headers, freeze panes, auto-filter),
with these input changes:
- regions come from `run_pipeline`'s ROI list (bbox, mask, anomaly_score) —
  not from a private contour extractor
- pixel→latlon stays rasterio `transform.xy` + `warp_transform` to EPSG:4326,
  fed by `meta.crs` / `meta.transform` from `preprocessing.raster_loader`
- Metadata sheet records manifest facts: detector, threshold_pct, timings_s,
  n_rois, git_sha instead of the reference's hardcoded method strings

### `ui/preview.py` — false-color RGB + ROI overlay
Port of `make_preview_rgb` / `draw_boxes`: three spread bands, 2–98 percentile
stretch, boxes for up to N ROIs with ID + score labels. ROIs come from the
GeoJSON output so preview and exported geometry cannot disagree.

## 3. One edit to an existing file (coordination required)

Add optional callbacks to the stage runner in `pipeline/run_pipeline.py`
(`stage()` at line ~120): `run_pipeline(..., progress_fn=None, log_fn=None)`
calling `progress_fn(stage_index, n_stages, name)` after each stage. ~5 lines.
The UI maps this to real progress instead of fake percentages.

Fallback while the sentinel-2 agent owns that file: indeterminate progress bar
(`mode="indeterminate"`), no edit needed. Decide at build time.

## 4. Per-run outputs

Everything `run_pipeline` already writes into `--out`:
`{scene_id}_anom_{detector}_(raw|norm).tif`, `{scene_id}_mask.tif`,
`{scene_id}_rois.geojson`, `run_manifest.json`
— plus, written by the UI layer:
`coordinates.xlsx` (main deliverable, same role as the reference's) and an
in-app preview. Status line reports n_ROIs, total wall time and git SHA from
the manifest.

## 5. Tests

- `tests/test_ui_excel_export.py` — synthetic tiny cube/meta; assert sheets,
  lat/lon columns finite, metadata rows match manifest.
- `tests/test_ui_preview.py` — RGB shape/dtype, boxes drawn within bounds.
- Headless-safe: neither module imports tkinter at import time; only
  `ui/app.py` does, and it is excluded from unit tests (Tk needs a display).
- Optional smoke: run the UI worker path on a small fixture scene end-to-end.

## 6. Out of scope for v1 (recorded for v2)

- Segmentation tab: run `experiments/seg_arch/unet_pretext.pt` via
  `segmentation/infer.py` on detected ROIs.
- Change-detection tab: temporal arms in `change_detection/`.
- Edge tab: launchers for `edge/benchmark.py` outputs.
- Sentinel-2 source support lands with the S2 pipeline agent's work; the UI
  picks it up automatically once `source="sentinel2"` is accepted by
  `load_scene`.

## 7. Build order

1. `ui/excel_export.py` + tests (pure functions, no Tk).
2. `ui/preview.py` + tests.
3. `pipeline/run_pipeline.py` callback hook (coordinate with S2 agent).
4. `ui/app.py` wiring everything.
5. Manual run against one `data/raw/enmap` scene; verify xlsx opens in
   LibreOffice/Excel and GeoJSON matches drawn boxes.
