"""§3B.6 -- ROI-only segmentation inference (Roadmap §6 stage 5).

`segment_rois` is the ONLY operational inference path. Full-scene inference
(running the U-Net over every pixel of a scene) exists solely as the
comparison baseline that will live under `edge/` (§3D) -- it is never used
here and this module does not implement it.

Design decisions (plan.md leaves these to the implementer):

* **bbox smaller than `patch`: expand, don't pad.** A window is cropped
  centered on the ROI and clipped so it stays entirely inside the scene
  whenever the scene is at least `patch` px on that axis. The model was
  trained on real 64x64 background context (segmentation/datasets.py); a
  zero/NaN-padded patch around a tiny ROI would hand it a distribution it
  never saw in training, whereas real neighbouring pixels are exactly the
  kind of context it was trained on. Padding (NaN) is used only in the
  unavoidable case where the scene itself is smaller than `patch` on some
  axis -- there is no real context left to expand into.

* **bbox larger than `patch`: tile, don't resize.** The model's receptive
  field and learned spatial scale are fixed to native-resolution 64x64
  patches (segmentation/train_unet.py::LightUNet). Resizing a large ROI
  down to 64x64 would require choosing an interpolation scheme for an
  already PCA-reduced, standardized 30-band cube and would distort exactly
  the spatial scale the model learned; tiling with stride `patch` keeps
  every tile at native resolution and pixel-for-pixel correspondence with
  the input cube, at the cost of possibly several forward passes per ROI
  (batched together via the `batch` parameter).

* **`seg_prob` = mean sigmoid probability over pixels inside `roi.mask`**,
  pooled across every window that covers the ROI's bbox (a first-window-wins
  composite of overlapping/adjacent tiles avoids double counting where tiles
  overlap after edge-clipping). This is the reading D4 already assumes:
  "Mean segmentation probability inside polygon" feeds `c_seg` at weight
  0.25 (plan.md D4) -- `filter_rois`/`shape_plausibility` (segmentation/
  postfilter.py, §3B.7) are a *separate*, geometry-only stage 4 gate that
  runs before this stage-5 inference and does not consume `seg_prob`.

**D15 leakage constraint.** The model consumes C=30 PCA-reduced bands, and
that PCA transformer is fit ONCE on the TRAIN split only
(segmentation/datasets.py::fit_reduce_bands_transformer, persisted at
experiments/seg_arch/reduce_bands_transformer.pkl). Re-fitting a transformer
on the scene being scored -- even accidentally, e.g. by passing a raw
ndarray where `preprocessing.harmonize.reduce_bands`'s `fit_on` duck-types
on `.transform` -- leaks scoring-scene spectral statistics into the
representation (exactly the leak D15 moved `reduce_bands` out of §3A.1 to
prevent). `segment_rois` therefore requires an object that already exposes
`.transform` and refuses (`TypeError`) anything else, rather than silently
falling through to `reduce_bands`'s fit-fresh branch.

**Shared preprocessing, not duplicated.** `reduce_bands` (preprocessing/
harmonize.py) and `standardize` (preprocessing/normalize.py) are imported
and called exactly as segmentation/datasets.py's SyntheticSegDataset and
RealSegDataset call them (transform-only reduce_bands, then per-patch
standardize -- D19: raw PCA output is radiance-scale, thousands in
magnitude, and the model was trained on standardized input only). Nothing
here reimplements that logic.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from core.contracts import ROIRecord, SceneMeta
from preprocessing.harmonize import reduce_bands
from preprocessing.normalize import standardize

ROOT = Path(__file__).resolve().parents[1]

# Kept as a local constant rather than importing segmentation.datasets.N_COMPONENTS:
# datasets.py transitively pulls in pandas/sklearn/scipy.io/preprocessing.background_pool
# for a single int. The two must still agree -- segmentation/datasets.py:46 defines the
# same value, and fit_reduce_bands_transformer (datasets.py) is what actually produces
# the pickled transformer this module loads by default.
N_COMPONENTS = 30

DEFAULT_TRANSFORMER_PATH = ROOT / "experiments" / "seg_arch" / "reduce_bands_transformer.pkl"


def _load_default_transformer(path: Path = DEFAULT_TRANSFORMER_PATH):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _clip_origin(start: int, scene_size: int, patch: int) -> int:
    """Slide a window origin so the window stays inside [0, scene_size) when
    the scene is big enough to allow it; otherwise pin to 0 and let the
    caller pad. Real pixels are always preferred over padded ones."""
    if scene_size <= patch:
        return 0
    return int(max(0, min(start, scene_size - patch)))


def _windows_for_bbox(bbox: tuple[int, int, int, int], patch: int,
                       scene_shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Patch-aligned window origins (row0, col0) covering `bbox`. One window
    if the bbox fits inside a single patch (centered, expand case); several
    tiled at stride=`patch` if it doesn't (tile case). Deduplicated after
    edge-clipping so a tile that got pushed inward doesn't get scheduled
    twice."""
    r0, c0, r1, c1 = bbox
    scene_h, scene_w = scene_shape
    bbox_h, bbox_w = r1 - r0, c1 - c0

    if bbox_h <= patch and bbox_w <= patch:
        center_r, center_c = (r0 + r1) // 2, (c0 + c1) // 2
        raw = [(center_r - patch // 2, center_c - patch // 2)]
    else:
        row_starts = list(range(r0, r1, patch)) if bbox_h > patch else [r0]
        col_starts = list(range(c0, c1, patch)) if bbox_w > patch else [c0]
        raw = [(rr, cc) for rr in row_starts for cc in col_starts]

    seen: set[tuple[int, int]] = set()
    windows: list[tuple[int, int]] = []
    for rr, cc in raw:
        wr = _clip_origin(rr, scene_h, patch)
        wc = _clip_origin(cc, scene_w, patch)
        if (wr, wc) not in seen:
            seen.add((wr, wc))
            windows.append((wr, wc))
    return windows


def _extract_window(cube: np.ndarray, r0w: int, c0w: int, patch: int) -> np.ndarray:
    """[patch, patch, B] crop at (r0w, c0w), NaN-padded at the bottom/right
    if the scene is smaller than `patch` on some axis. Never shifts the
    window to "fake" alignment -- the canvas origin always equals (r0w, c0w)
    in scene coordinates, which is what the caller needs to map ROI-mask
    pixels back into this window."""
    scene_h, scene_w, b = cube.shape
    r1w = min(r0w + patch, scene_h)
    c1w = min(c0w + patch, scene_w)
    canvas = np.full((patch, patch, b), np.nan, dtype=np.float32)
    canvas[: r1w - r0w, : c1w - c0w] = cube[r0w:r1w, c0w:c1w]
    return canvas


def _prepare_model_input(window_cube: np.ndarray, transformer, n_components: int) -> np.ndarray:
    """window_cube [patch, patch, B] (raw bands, may contain NaN, incl. any
    edge padding) -> [n_components, patch, patch] float32 model input.

    reduce_bands with fit_on=transformer (an already-fitted object) takes
    the transform-only branch -- see preprocessing/harmonize.py -- so this
    never calls .fit(). standardize matches segmentation/datasets.py's
    dataset classes exactly (D19: PCA output is unnormalized radiance
    scale). nan_to_num runs LAST, after standardize's nanmean/nanstd, so a
    padded region can never poison the per-band statistics with fake zeros
    -- it only stops NaN from propagating into real pixels' outputs through
    the conv layers' receptive field. Padded pixels never coincide with any
    roi.mask==1 pixel (padding only occurs beyond the true scene edge, and
    ROI bboxes are built from real scene pixels -- geospatial/polygonize.py),
    so the fill value can never influence seg_prob.
    """
    reduced, _ = reduce_bands(window_cube, n_components=n_components, fit_on=transformer)
    normed = standardize(reduced)
    normed = np.nan_to_num(normed, nan=0.0)
    return np.moveaxis(normed, -1, 0).astype(np.float32)   # [C, H, W]


def segment_rois(cube: np.ndarray, meta: SceneMeta, rois: list[ROIRecord], model, *,
                  patch: int = 64, batch: int = 16, transformer=None,
                  n_components: int = N_COMPONENTS,
                  transformer_path: Path = DEFAULT_TRANSFORMER_PATH,
                  device: str | torch.device | None = None) -> list[ROIRecord]:
    """ROI-only inference (Roadmap §6 stage 5) -- crops each ROI bbox to a
    patch-aligned window, runs the model, writes seg_prob back onto the
    ROIRecord. Full-scene inference exists ONLY as the edge/ comparison
    baseline; it is never the operational path.

    `meta` is accepted (unused beyond documentation/typing) for interface
    consistency with the rest of the Stage 4-6 pipeline, which threads
    (cube, meta, ...) through every stage; all the geometry this function
    needs is already in `cube.shape` and each ROI's `bbox`/`mask`.

    Parameters
    ----------
    transformer : object or None
        The FITTED PCA transformer (`.transform` present). If None, loaded
        from `transformer_path` (default: experiments/seg_arch/
        reduce_bands_transformer.pkl, the checkpoint scripts/
        train_seg_pretext.py writes). Must already be fitted -- see D15
        note above the module docstring. Passing anything without a
        `.transform` attribute raises TypeError rather than silently
        reaching `reduce_bands`'s fit-fresh branch.
    """
    if not rois:
        return rois

    if transformer is None:
        transformer = _load_default_transformer(transformer_path)
    if not hasattr(transformer, "transform"):
        raise TypeError(
            "segment_rois requires an already-FITTED transformer (an object exposing "
            ".transform, e.g. the unpickled experiments/seg_arch/reduce_bands_transformer.pkl). "
            "Got an object with no .transform attribute -- passing this through to "
            "preprocessing.harmonize.reduce_bands would silently fit a NEW transformer on "
            "this scoring scene, which is exactly the leak D15 exists to prevent "
            "(plan.md D15; see segmentation/datasets.py's module docstring).")

    fitted_n = getattr(transformer, "n_components_", None)
    if fitted_n is not None and fitted_n != n_components:
        raise ValueError(
            f"transformer was fit with n_components_={fitted_n} but n_components={n_components} "
            "was requested; reduce_bands allocates its output array from n_components and "
            "assigns transformer.transform(...) into it, so a mismatch is a shape error or "
            "silent misalignment downstream, not a clean failure at this boundary.")

    scene_shape = cube.shape[:2]

    was_training = model.training if hasattr(model, "training") else False
    if hasattr(model, "eval"):
        model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Build every (roi_index, window_origin, model_input) job across all ROIs
    # up front, then run them through the model in chunks of `batch` -- this
    # is what makes `batch` do real work rather than being a per-ROI no-op.
    jobs: list[tuple[int, int, int, np.ndarray]] = []
    for roi_idx, roi in enumerate(rois):
        for (r0w, c0w) in _windows_for_bbox(roi.bbox, patch, scene_shape):
            window_cube = _extract_window(cube, r0w, c0w, patch)
            model_input = _prepare_model_input(window_cube, transformer, n_components)
            jobs.append((roi_idx, r0w, c0w, model_input))

    prob_maps = [np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)
                 for (r0, c0, r1, c1) in (roi.bbox for roi in rois)]

    with torch.no_grad():
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            batch_arr = np.stack([job[3] for job in chunk], axis=0)
            batch_t = torch.from_numpy(batch_arr).to(device)
            logits = model(batch_t)
            probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()   # [n, patch, patch]

            for (roi_idx, r0w, c0w, _inp), prob_win in zip(chunk, probs):
                roi = rois[roi_idx]
                r0, c0, r1, c1 = roi.bbox
                ov_r0, ov_r1 = max(r0, r0w), min(r1, r0w + patch)
                ov_c0, ov_c1 = max(c0, c0w), min(c1, c0w + patch)
                if ov_r0 >= ov_r1 or ov_c0 >= ov_c1:
                    continue
                pm = prob_maps[roi_idx]
                dest = pm[ov_r0 - r0: ov_r1 - r0, ov_c0 - c0: ov_c1 - c0]
                src = prob_win[ov_r0 - r0w: ov_r1 - r0w, ov_c0 - c0w: ov_c1 - c0w]
                unfilled = np.isnan(dest)
                dest[unfilled] = src[unfilled]   # first-window-wins; dest is a view into pm

    for roi, pm in zip(rois, prob_maps):
        vals = pm[roi.mask.astype(bool)]
        vals = vals[~np.isnan(vals)]
        roi.seg_prob = float(vals.mean()) if vals.size else 0.0

    if was_training and hasattr(model, "train"):
        model.train()
    return rois
