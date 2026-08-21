"""§3B.3 segmentation/datasets.py."""
from pathlib import Path

import numpy as np
import pytest
import torch

from preprocessing.harmonize import RETAINED_BANDS
from segmentation.datasets import (
    HAD100_TEST_CROP_DICT,
    MANIFEST_PATH,
    POOL_PATH,
    RealSegDataset,
    SyntheticSegDataset,
    fit_reduce_bands_transformer,
    load_manifest_records,
    train_val_scene_split,
)

ROOT = Path(__file__).resolve().parents[1]
HAD100_ROOT = ROOT / "data" / "benchmark" / "had100" / "HAD100"
_have_pool = POOL_PATH.exists() and MANIFEST_PATH.exists()
_have_had100_test = (HAD100_ROOT / "data" / "aviris_ng_target").exists()


# --- RealSegDataset: EVAL ONLY, enforced structurally -----------------------

def test_real_seg_dataset_cannot_be_constructed_with_split_train():
    with pytest.raises(AssertionError, match="scoring-only"):
        RealSegDataset(source="had100", split="train", transformer=object())


@pytest.mark.parametrize("source", ["abu", "hydice_urban_anomaly"])
def test_real_seg_dataset_abu_hydice_raise_named_error(source):
    with pytest.raises(NotImplementedError, match="wavelength array"):
        RealSegDataset(source=source, split="eval", transformer=object())


def test_real_seg_dataset_rejects_unknown_source():
    with pytest.raises(ValueError):
        RealSegDataset(source="bogus", split="eval", transformer=object())


# --- no scene overlaps between the synthetic (train) source and the real
#     (eval) source -- the disjointness the "no id in both train and eval
#     manifest" accept criterion is checking for, applied to scene_id ------

@pytest.mark.skipif(not (_have_pool and _have_had100_test), reason="pool/HAD100 not built")
def test_background_pool_and_had100_test_scenes_are_disjoint():
    records = load_manifest_records()
    background_scene_ids = {r.scene_id for r in records}
    test_scene_ids = {p.stem for p in (HAD100_ROOT / "data" / "aviris_ng_target").glob("*.hdr")}
    assert background_scene_ids.isdisjoint(test_scene_ids)


@pytest.mark.skipif(not _have_had100_test, reason="HAD100 test scenes not fetched")
def test_had100_test_crop_dict_arithmetic_matches_d11_2():
    """18 keys, 6 with 2 crops => 94 raw scenes + 6 extra = 100 test patches."""
    n_raw = len(list((HAD100_ROOT / "data" / "aviris_ng_target").glob("*.hdr")))
    assert n_raw == 94
    assert len(HAD100_TEST_CROP_DICT) == 18
    extra = sum(len(v) - 1 for v in HAD100_TEST_CROP_DICT.values())
    assert extra == 6
    assert n_raw + extra == 100


# --- train_val_scene_split ---------------------------------------------------

@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_train_val_scene_split_is_disjoint_and_covers_all_patches():
    split = train_val_scene_split(val_fraction=0.2, seed=0)
    assert split["train_scene_ids"].isdisjoint(split["val_scene_ids"])
    total = len(split["train_array_indices"]) + len(split["val_array_indices"])
    assert total == 2088
    assert len(split["val_array_indices"]) / total == pytest.approx(0.2, abs=0.05)


# --- fit_reduce_bands_transformer: streamed, bounded, train-only -----------

@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_fit_reduce_bands_transformer_on_small_subsample():
    split = train_val_scene_split(val_fraction=0.2, seed=0)
    small_train = split["train_array_indices"][:20]
    transformer, n_sampled = fit_reduce_bands_transformer(
        train_array_indices=small_train, n_components=5, n_sample_pixels=2000, seed=0)
    assert hasattr(transformer, "transform")
    assert 0 < n_sampled <= 2000


# --- SyntheticSegDataset ------------------------------------------------------

@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_synthetic_seg_dataset_pretext_shapes_and_dtypes():
    split = train_val_scene_split(val_fraction=0.2, seed=0)
    transformer, _n = fit_reduce_bands_transformer(
        train_array_indices=split["train_array_indices"][:20], n_components=5,
        n_sample_pixels=2000, seed=0)

    ds = SyntheticSegDataset(
        mode="pretext", array_indices=split["train_array_indices"][:5],
        transformer=transformer, n_components=5, seed=0)
    assert len(ds) == 5
    patch, mask = ds[0]
    assert isinstance(patch, torch.Tensor) and isinstance(mask, torch.Tensor)
    assert patch.shape == (5, 64, 64)
    assert patch.dtype == torch.float32
    assert mask.shape == (1, 64, 64)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_synthetic_seg_dataset_implanted_requires_target_spectra():
    split = train_val_scene_split(val_fraction=0.2, seed=0)
    with pytest.raises(ValueError, match="target_spectra"):
        SyntheticSegDataset(
            mode="implanted", array_indices=split["train_array_indices"][:5],
            transformer=object())


@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_synthetic_seg_dataset_implanted_runs_with_synthetic_spectra():
    """Exercises the implanted path end to end with placeholder target
    spectra (not gated on load_target_spectra's suspended pools) --
    confirms the wiring, independent of the spectra-provenance question."""
    split = train_val_scene_split(val_fraction=0.2, seed=0)
    transformer, _n = fit_reduce_bands_transformer(
        train_array_indices=split["train_array_indices"][:20], n_components=5,
        n_sample_pixels=2000, seed=0)
    rng = np.random.default_rng(0)
    target_spectra = rng.normal(loc=50.0, size=(3, RETAINED_BANDS)).astype(np.float32)

    ds = SyntheticSegDataset(
        mode="implanted", array_indices=split["train_array_indices"][:3],
        transformer=transformer, n_components=5, target_spectra=target_spectra,
        n_targets=2, seed=0)
    patch, mask = ds[0]
    assert patch.shape == (5, 64, 64)
    assert mask.shape == (1, 64, 64)


# --- RealSegDataset(had100, eval) -- real scoring path ----------------------

@pytest.mark.skipif(not (_have_pool and _have_had100_test), reason="pool/HAD100 not built")
def test_real_seg_dataset_had100_eval_shapes():
    split = train_val_scene_split(val_fraction=0.2, seed=0)
    transformer, _n = fit_reduce_bands_transformer(
        train_array_indices=split["train_array_indices"][:20], n_components=5,
        n_sample_pixels=2000, seed=0)

    ds = RealSegDataset(source="had100", split="eval", transformer=transformer, n_components=5)
    assert len(ds) == 100
    patch, mask = ds[0]
    assert patch.shape == (5, 64, 64)
    assert patch.dtype == torch.float32
    assert mask.shape == (1, 64, 64)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
