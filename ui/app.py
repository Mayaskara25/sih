"""SIH anomaly-detection desktop UI -- a thin front-end over
pipeline/run_pipeline.py (D35, docs/ui_plan.md).

What this deliberately does NOT do:
  - no detection science here; run_pipeline owns every stage
  - coordinates.xlsx and the preview are derived from the run's own outputs
    (GeoJSON / mask TIFF / manifest), never recomputed (D35 rules 1-2)
  - no simulated figures are displayed anywhere (O2)

Memory guard (docs/ui_plan.md section 0.1): a full EnMAP scene is ~1.3 GB as
one float32 copy and this machine has been kernel-OOM-killed at 8.7 GB RSS.
The default is therefore a WINDOWED read whose size is shown and bounded
before Run; a full-scene request whose estimated working set exceeds
MEM_LIMIT_MB is refused with an explanation rather than killed mid-run.
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

import numpy as np
import rasterio
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from pipeline.run_pipeline import DETECTORS, run_pipeline
from ui.excel_export import export_coordinates_xlsx
from ui.preview import render_preview

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_DIR = REPO / "data" / "raw" / "enmap"

# Estimated peak working set of the in-memory pipeline, as a multiple of one
# float32 cube copy (cube + standardized copy + detector temporaries).
#
# DERIVED FROM MEASUREMENT, NOT ASSUMED. The two data points this project
# actually has, both from the Phase 5 Level 2 runs recorded in plan.md D32:
#
#   1178x1229x224 full scene   base 1,297 MB   peak 8,700 MB   factor 6.71  (OOM-KILLED)
#    600x 402x224 window       base   216 MB   peak   829 MB   factor 3.84  (completed)
#
# The factor GROWS with scene size, so calibrating on the small case
# under-predicts the large one -- which is the dangerous direction. An earlier
# value of 4.0 estimated the full scene at 5,189 MB, i.e. UNDER the limit
# below, and would therefore have ALLOWED the exact scene the kernel killed.
# Take the larger measured factor so the estimate is conservative where it
# matters; on the window case it over-predicts (1,449 vs 829 MB), which costs
# nothing because that case is allowed either way.
COPY_FACTOR = 6.71
MEM_LIMIT_MB = 6_000.0        # D35: refuse rather than be OOM-killed
                              # ~13 GB machine with NO swap, so the kernel
                              # SIGKILLs rather than raising MemoryError.

DEFAULT_WIN_H, DEFAULT_WIN_W = 600, 402   # Level 2's proven window: 829 MB peak

_SOURCES = ["auto", "enmap", "had100", "sentinel2", "aviris",
            "abu", "hydice_urban_anomaly", "indian_pines"]


def guess_source(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("enmap01"):
        return "enmap"
    if name.endswith("_stack.tif"):
        return "sentinel2"
    if name.endswith(".hdr"):
        return "had100"
    return None


def probe_shape(path: Path) -> tuple[int, int, int]:
    """(height, width, band_count) without reading pixels."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as ds:
            return ds.height, ds.width, ds.count
    raise ValueError(f"cannot probe {ext!r} scenes -- pick a GeoTIFF for the UI")


def estimate_mb(h: int, w: int, b: int) -> float:
    return h * w * b * 4 * COPY_FACTOR / 1e6


class AnomalyUI:
    STAGE_COUNT = 10   # staged calls inside run_pipeline; progress bar max

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("SIH Hyperspectral Anomaly Detection")
        root.geometry("1050x760")
        root.minsize(900, 640)

        self.scene_path = tk.StringVar()
        self.source = tk.StringVar(value="auto")
        self.detector = tk.StringVar(value="global_rx")
        self.threshold_pct = tk.DoubleVar(value=99.0)
        self.normalize = tk.StringVar(value="standardize")
        self.profile = tk.StringVar(value="object")
        self.detector_params = tk.StringVar()
        self.use_window = tk.BooleanVar(value=True)
        self.win_h = tk.IntVar(value=DEFAULT_WIN_H)
        self.win_w = tk.IntVar(value=DEFAULT_WIN_W)
        self.win_r0 = tk.IntVar(value=0)
        self.win_c0 = tk.IntVar(value=0)

        self.status = tk.StringVar(value="Choose a scene GeoTIFF to begin.")
        self.mem_estimate = tk.StringVar(value="Working-set estimate: -")
        self.result_line = tk.StringVar(value="")
        self.progress = tk.DoubleVar(value=0.0)
        self.preview_photo = None

        self._build()

    # ------------------------------------------------------------------ UI

    def _build(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="SIH Hyperspectral Anomaly Pipeline",
                  font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(outer, text=(
            "scene -> drop bad bands -> standardize -> detector -> threshold "
            "-> ROIs -> GeoJSON + mask + coordinates.xlsx"
        )).pack(anchor="w", pady=(2, 10))

        row = ttk.Frame(outer); row.pack(fill="x")
        ttk.Entry(row, textvariable=self.scene_path).pack(side="left", fill="x",
                                                          expand=True)
        ttk.Button(row, text="Choose scene...", command=self.choose_scene)\
            .pack(side="left", padx=(6, 0))

        prow = ttk.Frame(outer); prow.pack(fill="x", pady=(8, 0))
        ttk.Label(prow, text="Source").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Combobox(prow, textvariable=self.source, values=_SOURCES,
                     width=14, state="readonly").grid(row=0, column=1, padx=(0, 12))
        ttk.Label(prow, text="Detector").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Combobox(prow, textvariable=self.detector, values=sorted(DETECTORS),
                     width=11, state="readonly").grid(row=0, column=3, padx=(0, 12))
        ttk.Label(prow, text="Threshold %").grid(row=0, column=4, sticky="w", padx=(0, 4))
        ttk.Spinbox(prow, textvariable=self.threshold_pct, from_=90, to=99.99,
                    increment=0.25, width=7).grid(row=0, column=5, padx=(0, 12))
        ttk.Label(prow, text="Normalize").grid(row=0, column=6, sticky="w", padx=(0, 4))
        ttk.Combobox(prow, textvariable=self.normalize,
                     values=["standardize", "l2"], width=12,
                     state="readonly").grid(row=0, column=7, padx=(0, 12))
        ttk.Label(prow, text="Profile").grid(row=0, column=8, sticky="w", padx=(0, 4))
        ttk.Combobox(prow, textvariable=self.profile,
                     values=["object", "landcover"], width=10,
                     state="readonly").grid(row=0, column=9)

        jrow = ttk.Frame(outer); jrow.pack(fill="x", pady=(6, 0))
        ttk.Label(jrow, text="Detector params JSON (optional)").pack(side="left")
        ttk.Entry(jrow, textvariable=self.detector_params).pack(
            side="left", fill="x", expand=True, padx=(8, 0))

        # --- window / memory frame -------------------------------------
        wframe = ttk.Labelframe(outer, text="Windowed read (recommended -- "
                                            "full scenes can exceed RAM)", padding=8)
        wframe.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(wframe, text="Use window",
                        variable=self.use_window,
                        command=self._update_mem).pack(side="left")
        for label, var in (("rows", self.win_h), ("cols", self.win_w),
                           ("row start", self.win_r0), ("col start", self.win_c0)):
            ttk.Label(wframe, text=label).pack(side="left", padx=(10, 2))
            sb = ttk.Spinbox(wframe, textvariable=var, from_=0, to=100000,
                             width=7, command=self._update_mem)
            sb.pack(side="left")
        ttk.Button(wframe, text="Center window", command=self.center_window)\
            .pack(side="left", padx=(10, 0))
        ttk.Label(wframe, textvariable=self.mem_estimate).pack(side="left", padx=(16, 0))

        arow = ttk.Frame(outer); arow.pack(fill="x", pady=(10, 4))
        self.run_button = ttk.Button(arow, text="Run detection",
                                     command=self.start)
        self.run_button.pack(side="left")
        ttk.Label(arow, textvariable=self.result_line).pack(side="left", padx=(14, 0))

        ttk.Progressbar(outer, variable=self.progress,
                        maximum=self.STAGE_COUNT).pack(fill="x")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(4, 8))

        content = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        content.pack(fill="both", expand=True)

        log_frame = ttk.Labelframe(content, text="Processing log", padding=6)
        content.add(log_frame, weight=1)
        self.log_box = tk.Text(log_frame, wrap="word", width=48)
        self.log_box.pack(fill="both", expand=True)

        prev_frame = ttk.Labelframe(content, text="Preview (ROIs from the run's GeoJSON)",
                                    padding=6)
        content.add(prev_frame, weight=2)
        self.preview_label = ttk.Label(prev_frame, anchor="center",
                                       text="Preview appears here after a run.")
        self.preview_label.pack(fill="both", expand=True)

    # ------------------------------------------------------------ actions

    def choose_scene(self):
        path = filedialog.askopenfilename(
            title="Choose scene", initialdir=str(DEFAULT_SCENE_DIR),
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("ENVI header", "*.hdr"),
                       ("All files", "*.*")])
        if not path:
            return
        self.scene_path.set(path)
        auto = guess_source(Path(path))
        if auto:
            self.source.set(auto)
        try:
            h, w, b = probe_shape(Path(path))
            self.win_h.set(min(self.win_h.get(), h))
            self.win_w.set(min(self.win_w.get(), w))
            self.status.set(f"Scene: {h} x {w} x {b}. "
                            "Keep the window bounded unless you know RAM allows more.")
        except ValueError as exc:
            self.status.set(str(exc))
        self._update_mem()

    def center_window(self):
        path = self._require_scene(silent=True)
        if path is None:
            return
        h, w, _ = probe_shape(path)
        vh, vw = min(self.win_h.get(), h), min(self.win_w.get(), w)
        self.win_r0.set(max(0, (h - vh) // 2))
        self.win_c0.set(max(0, (w - vw) // 2))
        self._update_mem()

    def current_window(self) -> tuple[int, int, int, int] | None:
        """Clamped (r0, c0, h, w), or None when windowing is off or the scene
        format cannot be probed/windowed (.mat/.hdr are small enough not to
        need it -- the D35 OOM hazard is full-size EnMAP/S2 GeoTIFFs)."""
        path = self.scene_path.get().strip()
        if not self.use_window.get() or not path or not Path(path).exists():
            return None
        try:
            h_full, w_full, _b = probe_shape(Path(path))
        except ValueError:
            return None
        r0 = max(0, min(int(self.win_r0.get()), h_full - 1))
        c0 = max(0, min(int(self.win_c0.get()), w_full - 1))
        h = max(1, min(int(self.win_h.get()), h_full - r0))
        w = max(1, min(int(self.win_w.get()), w_full - c0))
        self.win_h.set(h); self.win_w.set(w)
        self.win_r0.set(r0); self.win_c0.set(c0)
        return r0, c0, h, w

    def _update_mem(self):
        path = self.scene_path.get().strip()
        if not path or not Path(path).exists():
            self.mem_estimate.set("Working-set estimate: -")
            return
        try:
            _, _, b = probe_shape(Path(path))
        except ValueError:
            self.mem_estimate.set("Working-set estimate: n/a for this format")
            return
        if self.use_window.get():
            mb = estimate_mb(self.win_h.get(), self.win_w.get(), b)
            self.mem_estimate.set(f"Estimated working set ~{mb:,.0f} MB "
                                  f"(window {self.win_h.get()}x{self.win_w.get()})")
        else:
            h, w, _ = probe_shape(Path(path))
            mb = estimate_mb(h, w, b)
            flag = "  OVER LIMIT -- will refuse" if mb > MEM_LIMIT_MB else ""
            self.mem_estimate.set(f"Estimated working set ~{mb:,.0f} MB (FULL SCENE){flag}")

    def _require_scene(self, silent: bool = False) -> Path | None:
        raw = self.scene_path.get().strip()
        if not raw:
            if not silent:
                messagebox.showerror("No scene", "Choose a scene first.")
            return None
        path = Path(raw)
        if not path.exists():
            if not silent:
                messagebox.showerror("File not found", f"{path} does not exist.")
            return None
        return path

    def start(self):
        path = self._require_scene()
        if path is None:
            return

        source = self.source.get()
        if source == "auto":
            guessed = guess_source(path)
            if not guessed:
                messagebox.showerror(
                    "Unknown source",
                    "Could not infer the source from the filename. Pick one "
                    "explicitly in the Source dropdown.")
                return
            source = guessed

        window = self.current_window()
        try:
            h_full, w_full, bands = probe_shape(path)
            if window is None:
                est = estimate_mb(h_full, w_full, bands)
                if est > MEM_LIMIT_MB:
                    messagebox.showerror(
                        "Scene too large for memory",
                        f"A full-scene run is estimated at ~{est:,.0f} MB and this "
                        f"machine has been OOM-killed at 8.7 GB (no swap).\n\n"
                        f"Enable 'Use window' (default {DEFAULT_WIN_H}x"
                        f"{DEFAULT_WIN_W}) and re-run.")
                    return
        except ValueError:
            pass  # .mat/.hdr: no probe, small scenes -- nothing to guard

        params_text = self.detector_params.get().strip()
        try:
            detector_params = json.loads(params_text) if params_text else None
        except json.JSONDecodeError as exc:
            messagebox.showerror("Bad detector params JSON", str(exc))
            return

        out_dir = path.parent / f"{path.stem}_ui_run"
        self.log_box.delete("1.0", "end")
        self.progress.set(0.0)
        self.result_line.set("")
        self.run_button.config(state="disabled")
        self.status.set("Running...")

        threading.Thread(target=self._worker, daemon=True, args=(
            path, source, window, detector_params, out_dir)).start()

    # ------------------------------------------------------------- worker

    def _worker(self, path: Path, source: str, window, detector_params, out_dir: Path):
        try:
            manifest = run_pipeline(
                scene=path, source=source, detector=self.detector.get(),
                threshold_pct=float(self.threshold_pct.get()),
                profile=self.profile.get(), out_dir=out_dir,
                normalize_method=self.normalize.get(),
                detector_params=detector_params, window=window,
                progress_fn=lambda i, name: self.ui_progress(i),
                log_fn=self.ui_log)

            outputs = manifest["outputs"]
            xlsx_path = export_coordinates_xlsx(
                out_dir / "coordinates.xlsx", outputs["geojson"],
                out_dir / "run_manifest.json")
            preview_path = render_preview(
                path, window, manifest.get("rois", []),
                out_dir / "preview_rois.png")

            wall = sum(manifest["timings_s"].values())
            summary = (f"{manifest['n_rois']} ROI(s) in {wall:.1f}s "
                       f"[{manifest['detector']} @ p{manifest['threshold_pct']}]")

            def finish():
                self._show_preview(preview_path)
                self.run_button.config(state="normal")
                self.status.set("Finished successfully.")
                self.result_line.set(summary)
                messagebox.showinfo(
                    "Run complete",
                    f"{summary}\n\n"
                    f"coordinates.xlsx: {xlsx_path}\n"
                    f"Outputs folder:   {out_dir}")
            self.root.after(0, finish)

        except Exception as exc:  # noqa: BLE001
            self.ui_log("\nERROR:")
            self.ui_log(traceback.format_exc())
            msg = str(exc)
            self.root.after(0, lambda: (
                self.run_button.config(state="normal"),
                self.status.set("Processing failed. Check the log."),
                messagebox.showerror("Processing failed", msg)))

    # --------------------------------------------------------- UI marshals

    def ui_log(self, text):
        def update():
            self.log_box.insert("end", str(text) + "\n")
            self.log_box.see("end")
        self.root.after(0, update)

    def ui_progress(self, value):
        self.root.after(0, lambda: self.progress.set(value))

    def _show_preview(self, path: Path):
        try:
            image = Image.open(path)
            image.thumbnail((620, 520))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception:
            self.preview_label.configure(text=f"Preview saved at:\n{path}")


def main():
    root = tk.Tk()
    AnomalyUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
