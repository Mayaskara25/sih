"""§3B.6 segmentation/infer.py -- ROI-only inference tests.

Fast, CPU-only. A GPU training run may be active elsewhere in the repo at
test time, so every model here is either a freshly-constructed (untrained)
LightUNet forced onto "cpu", or a tiny recording stand-in with no real
weights at all -- neither touches the GPU or the real checkpoint.

Two constraints get the most attention, per plan.md:

* D15 -- the PCA transformer that reduces 184 raw bands to C=30 must be
  APPLIED, never re-fit, at inference. `test_transformer_is_applied_not_refit`
  proves this behaviourally: it shows a transformer refit on the scoring
  scene would produce a detectably different model input, then asserts the
  model actually saw the pre-fitted transformer's output and NOT that
  alternative.
* D19 -- raw PCA output is radiance-scale (thousands); the model was
  trained on standardized input only. `test_standardization_applied_before_model`
  proves the model input is compressed to ~zero-mean/unit-variance even
  when the transformer emits large-magnitude output.
"""
from __future__ import annotations

import affine
import numpy as np
import pytest
import rasterio
import rasterio.crs
import torch
from sklearn.decomposition import PCA

from core.contracts import ROIRecord, SceneMeta, validate_roi
from preprocessing.harmonize import reduce_bands
from preprocessing.normalize import standardize
from segmentation.infer import (
    DEFAULT_TRANSFORMER_PATH,
    N_COMPONENTS,
    _windows_for_bbox,
    segment_rois,
)
from segmentation.train_unet import LightUNet


# --- fixtures ----------------------------------------------------------------

def _meta(**overrides) -> SceneMeta:
    defaults = dict(
        scene_id="test_scene",
        crs=rasterio.crs.CRS.from_epsg(32616),
        transform=affine.Affine(10.0, 0, 400_000.0, 0, -10.0, 4_000_000.0),
        wavelengths=None,
        bad_bands=np.zeros(4, dtype=bool),
        gsd_m=10.0,
        source="had100",
        georef="synthetic",
    )
    defaults.update(overrides)
    return SceneMeta(**defaults)


def _roi(index: int, bbox: tuple[int, int, int, int], *, scene_id: str = "test_scene",
         mask: np.ndarray | None = None) -> ROIRecord:
    r0, c0, r1, c1 = bbox
    if mask is None:
        mask = np.ones((r1 - r0, c1 - c0), dtype=np.uint8)
    return ROIRecord(
        roi_id=f"{scene_id}:anomaly:{index:04d}",
        source_branch="anomaly",
        target_profile="object",
        bbox=bbox,
        mask=mask,
    )


def _fit_transformer(*, n_bands: int, n_components: int = N_COMPONENTS,
                      n_pool: int = 2000, loc: float = 0.0, scale: float = 1.0,
                      seed: int = 0) -> PCA:
    """A transformer fit on its OWN pool, standing in for the real
    reduce_bands_transformer.pkl (fit on the TRAIN split, D15) -- distinct
    statistics from whatever scene it is later applied to."""
    rng = np.random.default_rng(seed)
    pool = rng.normal(loc=loc, scale=scale, size=(n_pool, n_bands)).astype(np.float64)
    return PCA(n_components=n_components).fit(pool)


def _cpu_model(n_bands: int = N_COMPONENTS) -> LightUNet:
    model = LightUNet(in_channels=n_bands)
    model.to("cpu")
    return model


class _RecordingModel:
    """Stands in for a real model: records every input tensor it is called
    with (for asserting exactly what preprocessing produced) and returns
    deterministic logits of the right shape. Deliberately exposes no
    `.parameters()` so segment_rois's device auto-detection falls back to
    "cpu" (see infer.py's `except (StopIteration, AttributeError)` branch)."""

    def __init__(self):
        self.calls: list[torch.Tensor] = []

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(x.detach().clone())
        b, _c, h, w = x.shape
        return torch.zeros((b, 1, h, w))


# --- seg_prob written for every ROI, no silent drops --------------------------

def test_seg_prob_written_for_every_roi_in_unit_range_no_drop():
    n_bands = 40
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(150, 150, n_bands)).astype(np.float32)
    transformer = _fit_transformer(n_bands=n_bands)
    model = _cpu_model()

    rois = [
        _roi(0, (10, 10, 20, 20)),
        _roi(1, (60, 60, 90, 100)),
        _roi(2, (120, 5, 140, 30)),
    ]

    out = segment_rois(cube, _meta(), rois, model, transformer=transformer, device="cpu")

    assert len(out) == len(rois)
    for roi in out:
        assert roi.seg_prob is not None
        assert 0.0 <= roi.seg_prob <= 1.0


# --- expand case (bbox < patch) and tile case (bbox > patch) -----------------

def test_bbox_smaller_than_patch_expand_case_valid_seg_prob():
    n_bands = 40
    rng = np.random.default_rng(2)
    cube = rng.normal(size=(150, 150, n_bands)).astype(np.float32)
    transformer = _fit_transformer(n_bands=n_bands, seed=1)
    model = _cpu_model()

    roi = _roi(0, (70, 70, 78, 78))   # 8x8, well inside a 64px patch
    out = segment_rois(cube, _meta(), [roi], model, transformer=transformer,
                        patch=64, device="cpu")

    assert out[0].seg_prob is not None
    assert np.isfinite(out[0].seg_prob)
    assert 0.0 <= out[0].seg_prob <= 1.0


def test_bbox_larger_than_patch_tile_case_valid_seg_prob():
    n_bands = 40
    rng = np.random.default_rng(3)
    cube = rng.normal(size=(150, 150, n_bands)).astype(np.float32)
    transformer = _fit_transformer(n_bands=n_bands, seed=2)
    model = _cpu_model()

    roi = _roi(0, (0, 0, 130, 100))   # bigger than 64x64 on both axes -> tiled
    out = segment_rois(cube, _meta(), [roi], model, transformer=transformer,
                        patch=64, device="cpu")

    assert out[0].seg_prob is not None
    assert np.isfinite(out[0].seg_prob)
    assert 0.0 <= out[0].seg_prob <= 1.0


# --- D15: the fitted transformer is applied, NEVER re-fit ---------------------

def test_transformer_is_applied_not_refit_D15():
    """Behavioural D15 guard: build a transformer fit on a pool with very
    different statistics from the scoring scene, so that re-fitting on the
    scene (the leak) would produce a DETECTABLY DIFFERENT model input than
    applying the pre-fitted transformer (the correct path). Then assert the
    model actually received the pre-fitted-transformer output, not the
    would-be-refit output.
    """
    n_bands, n_components = 40, 10
    rng = np.random.default_rng(4)
    # Scene statistics deliberately far from the pool the "good" transformer
    # was fit on, so a re-fit basis would differ substantially.
    scene = rng.normal(loc=50.0, scale=20.0, size=(64, 64, n_bands)).astype(np.float32)
    good_transformer = _fit_transformer(
        n_bands=n_bands, n_components=n_components, loc=0.0, scale=1.0, seed=5)

    # What a LEAKY re-fit-on-the-scoring-scene implementation would produce.
    leaky_reduced, _leaky_transformer = reduce_bands(
        scene, n_components=n_components, fit_on=None)
    leaky_input = np.nan_to_num(standardize(leaky_reduced), nan=0.0)
    leaky_input = np.moveaxis(leaky_input, -1, 0).astype(np.float32)

    # What the CORRECT transform-only path produces.
    correct_reduced, _ = reduce_bands(scene, n_components=n_components, fit_on=good_transformer)
    correct_input = np.nan_to_num(standardize(correct_reduced), nan=0.0)
    correct_input = np.moveaxis(correct_input, -1, 0).astype(np.float32)

    # Sanity: the two preprocessing paths really do diverge for this fixture
    # -- otherwise "never re-fit" would be untestable here.
    assert not np.allclose(correct_input, leaky_input, atol=1e-2)
    assert np.max(np.abs(correct_input - leaky_input)) > 0.5

    recorder = _RecordingModel()
    roi = _roi(0, (0, 0, 64, 64))   # bbox == patch == scene: exactly one window, no padding
    segment_rois(scene, _meta(), [roi], recorder, transformer=good_transformer,
                  patch=64, n_components=n_components, device="cpu")

    assert len(recorder.calls) == 1
    seen = recorder.calls[0][0].numpy()   # [C, 64, 64], batch of 1

    np.testing.assert_allclose(seen, correct_input, atol=1e-4)
    assert not np.allclose(seen, leaky_input, atol=1e-2), (
        "segment_rois produced model input matching a FRESH FIT on the scoring "
        "scene -- this is the D15 leak reduce_bands was moved out of §3A.1 to prevent")


def test_transformer_without_transform_attribute_raises_typeerror():
    n_bands = 20
    cube = np.zeros((64, 64, n_bands), dtype=np.float32)
    model = _cpu_model(n_bands=5)
    roi = _roi(0, (0, 0, 64, 64))

    not_a_transformer = np.zeros((5, 5))   # ndarray has no .transform
    with pytest.raises(TypeError):
        segment_rois(cube, _meta(), [roi], model, transformer=not_a_transformer,
                     n_components=5, device="cpu")


def test_transformer_n_components_mismatch_raises_valueerror():
    n_bands = 20
    cube = np.zeros((64, 64, n_bands), dtype=np.float32)
    model = _cpu_model(n_bands=5)
    roi = _roi(0, (0, 0, 64, 64))

    transformer = _fit_transformer(n_bands=n_bands, n_components=5, seed=6)
    assert transformer.n_components_ == 5

    with pytest.raises(ValueError):
        segment_rois(cube, _meta(), [roi], model, transformer=transformer,
                     n_components=7, device="cpu")   # deliberately mismatched


# --- D19: standardization applied before the model reaches it ----------------

def test_standardization_applied_before_model_D19():
    """A transformer whose raw output is radiance-scale (thousands, per D19)
    must reach the model compressed to ~zero-mean/unit-variance -- proof
    standardize() actually runs on the reduce_bands output before the model
    sees it, not the raw PCA-scale numbers."""
    n_bands, n_components = 12, 6

    class RadianceScaleTransformer:
        n_components_ = n_components

        def __init__(self):
            self._rng = np.random.default_rng(7)
            self._w = self._rng.normal(size=(n_bands, n_components))

        def transform(self, x: np.ndarray) -> np.ndarray:
            base = x @ self._w
            return (base * 1000.0 - 6000.0)   # thousands-scale, D19

    transformer = RadianceScaleTransformer()
    rng = np.random.default_rng(8)
    scene = rng.normal(size=(64, 64, n_bands)).astype(np.float32)   # full patch, no NaN padding

    # Sanity: confirm the raw (pre-standardize) reduce_bands output really is
    # large-magnitude for this fixture, else this test proves nothing.
    raw_reduced, _ = reduce_bands(scene, n_components=n_components, fit_on=transformer)
    assert np.nanmean(np.abs(raw_reduced)) > 500.0

    recorder = _RecordingModel()
    roi = _roi(0, (0, 0, 64, 64))
    segment_rois(scene, _meta(), [roi], recorder, transformer=transformer,
                 patch=64, n_components=n_components, device="cpu")

    seen = recorder.calls[0][0].numpy()   # [C, 64, 64]
    assert np.max(np.abs(seen)) < 50.0, "model input still radiance-scale -- standardize missing?"
    # bbox == patch == scene exactly, so no NaN padding: standardize's
    # per-band mean/std over this window should land within numerical noise
    # of 0/1 for every channel.
    per_channel_mean = seen.reshape(n_components, -1).mean(axis=1)
    per_channel_std = seen.reshape(n_components, -1).std(axis=1)
    np.testing.assert_allclose(per_channel_mean, 0.0, atol=1e-3)
    np.testing.assert_allclose(per_channel_std, 1.0, atol=1e-3)


def test_standardization_ordering_with_nan_padding_D19():
    """§3B.6's `_prepare_model_input` docstring claims nan_to_num runs LAST,
    AFTER standardize's nanmean/nanstd -- so a NaN-padded region (scene
    smaller than `patch`) can never poison the real pixels' per-band
    statistics with fake zeros. The other D19 test above never exercises
    padding at all (every scene there is >= patch on both axes), so it
    cannot distinguish correct ordering from the bug this docstring warns
    against (nan_to_num before standardize, which WOULD drag the real
    region's mean/std off 0/1). This test forces the padding branch: a
    40x40 scene against patch=64 pads 40:64 on both axes with NaN before
    _extract_window hands the window to preprocessing.
    """
    n_bands, n_components = 12, 6
    rng = np.random.default_rng(11)
    scene = rng.normal(size=(40, 40, n_bands)).astype(np.float32)   # scene < patch=64
    transformer = _fit_transformer(n_bands=n_bands, n_components=n_components, seed=11)

    recorder = _RecordingModel()
    roi = _roi(0, (0, 0, 40, 40))   # bbox == whole (real) scene, well inside patch
    segment_rois(scene, _meta(), [roi], recorder, transformer=transformer,
                 patch=64, n_components=n_components, device="cpu")

    assert len(recorder.calls) == 1
    seen = recorder.calls[0][0].numpy()   # [C, 64, 64]

    assert not np.isnan(seen).any(), "NaN padding must not reach the model"
    # Padding (originally NaN) is filled with 0.0 by nan_to_num -- if it were
    # filled BEFORE standardize instead, it would count as real zero-valued
    # data in the nanmean/nanstd, biasing the real region's statistics away
    # from 0/1 below.
    assert np.all(seen[:, 40:, :] == 0.0)
    assert np.all(seen[:, :, 40:] == 0.0)

    real = seen[:, :40, :40].reshape(n_components, -1)
    per_channel_mean = real.mean(axis=1)
    per_channel_std = real.std(axis=1)
    np.testing.assert_allclose(per_channel_mean, 0.0, atol=1e-3)
    np.testing.assert_allclose(per_channel_std, 1.0, atol=1e-3)


def test_windows_for_bbox_tiling_covers_the_whole_bbox():
    """Direct test of the expand-vs-tile window schedule (infer.py module
    docstring): for a bbox larger than `patch`, the union of returned
    patch-aligned windows must cover every pixel of the bbox, and origins
    must be deduplicated after edge-clipping (no duplicate scheduled job)."""
    patch = 64
    bbox = (0, 0, 130, 100)   # larger than patch on both axes -> tile case
    scene_shape = (150, 150)

    windows = _windows_for_bbox(bbox, patch, scene_shape)

    assert len(windows) == len(set(windows)), "duplicate window origins after edge-clipping"

    covered = np.zeros((150, 150), dtype=bool)
    for (r0, c0) in windows:
        covered[r0:r0 + patch, c0:c0 + patch] = True

    r0, c0, r1, c1 = bbox
    assert covered[r0:r1, c0:c1].all(), "tiled windows do not cover the full bbox"


def test_windows_for_bbox_expand_case_single_window_inside_scene():
    """bbox smaller than patch -> exactly one centered, in-bounds window."""
    patch = 64
    bbox = (70, 70, 78, 78)   # 8x8, well inside a large scene
    scene_shape = (150, 150)

    windows = _windows_for_bbox(bbox, patch, scene_shape)

    assert len(windows) == 1
    r0, c0 = windows[0]
    assert 0 <= r0 <= 150 - patch and 0 <= c0 <= 150 - patch
    r0b, c0b, r1b, c1b = bbox
    assert r0 <= r0b and r0 + patch >= r1b
    assert c0 <= c0b and c0 + patch >= c1b


# --- every returned ROI still passes validate_roi -----------------------------

def test_returned_rois_pass_validate_roi():
    n_bands = 40
    rng = np.random.default_rng(9)
    cube = rng.normal(size=(150, 150, n_bands)).astype(np.float32)
    transformer = _fit_transformer(n_bands=n_bands, seed=9)
    model = _cpu_model()

    rois = [
        _roi(0, (5, 5, 15, 12)),
        _roi(1, (40, 40, 110, 90)),
    ]
    out = segment_rois(cube, _meta(), rois, model, transformer=transformer, device="cpu")

    for roi in out:
        validate_roi(roi)   # raises ContractViolation on failure


# --- misc ----------------------------------------------------------------

def test_empty_roi_list_returns_immediately():
    cube = np.zeros((64, 64, 10), dtype=np.float32)
    model = _cpu_model(n_bands=5)
    out = segment_rois(cube, _meta(), [], model, device="cpu")
    assert out == []


def test_default_transformer_path_loads_and_scores_real_pkl():
    """Sanity check of the transformer=None default-loading branch, against
    the real experiments/seg_arch/reduce_bands_transformer.pkl (24KB; safe
    to load, unlike the 6.29GB background pool). Fitted on 184 raw bands ->
    C=30, matching N_COMPONENTS."""
    assert DEFAULT_TRANSFORMER_PATH.exists(), (
        "experiments/seg_arch/reduce_bands_transformer.pkl missing -- "
        "cannot exercise the default transformer-loading path")

    rng = np.random.default_rng(10)
    cube = rng.normal(size=(80, 80, 184)).astype(np.float32)
    model = _cpu_model(n_bands=N_COMPONENTS)
    roi = _roi(0, (10, 10, 30, 30))

    out = segment_rois(cube, _meta(), [roi], model, device="cpu")   # transformer=None -> default load

    assert out[0].seg_prob is not None
    assert 0.0 <= out[0].seg_prob <= 1.0
