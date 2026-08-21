"""§12 test_data_hygiene.py -- crop-level leakage (D11.5's third kind, the one
that inflates validation while scene-level and spectrum-level checks stay
green). Demonstrates the failure scene_groups() exists to prevent, not just
that scene_groups() returns labels -- a green suite should not survive a
future `pool[:1700]` / `pool[1700:]` split, and this is what catches it.

Runs against the REAL built background pool manifest (522 scenes, 2088
patches), not a synthetic stand-in, when it exists.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from preprocessing.background_pool import BackgroundPatch, scene_groups

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "had100_background_manifest.csv"
_have_pool = MANIFEST.exists()


def _load_records() -> list[BackgroundPatch]:
    df = pd.read_csv(MANIFEST)
    return [BackgroundPatch(**row) for row in df.to_dict("records")]


@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_naive_patch_index_split_leaks_scenes_across_train_val():
    """The failure scene_groups() exists to prevent: a naive split on patch
    (array) index puts a scene's 4 overlapping crops (D11.5) on both sides.
    """
    df = pd.read_csv(MANIFEST)
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=0)

    train_scenes = set(df.iloc[train_idx]["scene_id"])
    val_scenes = set(df.iloc[val_idx]["scene_id"])
    straddling = train_scenes & val_scenes
    assert len(straddling) > 0, (
        "expected the naive index split to leak at least one scene across "
        "train/val on the real 522-scene pool -- if this ever fails, it means "
        "the leak this test exists to demonstrate stopped reproducing, which "
        "would itself be worth investigating, not celebrating")


@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_group_shuffle_split_on_scene_groups_never_leaks():
    """The fix: GroupShuffleSplit keyed on scene_groups() puts all 4 crops of
    every scene on the same side, verified empirically against the real
    manifest rather than assumed from sklearn's docs."""
    records = _load_records()
    groups = scene_groups(records)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(splitter.split(np.arange(len(groups)), groups=groups))

    assert len(train_idx) > 0 and len(val_idx) > 0
    train_scenes = set(groups[train_idx])
    val_scenes = set(groups[val_idx])
    straddling = train_scenes & val_scenes
    assert straddling == set(), f"scene-grouped split leaked scenes: {straddling}"

    train_idx_set = set(train_idx.tolist())
    for scene in np.unique(groups):
        scene_positions = np.where(groups == scene)[0]
        sides = {pos in train_idx_set for pos in scene_positions}
        assert len(sides) == 1, f"scene {scene} split across train/val"


@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_every_scene_has_exactly_four_crops():
    records = _load_records()
    groups = scene_groups(records)
    _, counts = np.unique(groups, return_counts=True)
    assert (counts == 4).all()


@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_pool_tensor_has_no_nan_anywhere():
    """A streaming scan over the memmap, not a load of the full 6.3 GB
    tensor. Only the first 50 of 2088 patches (all NG) were checked when the
    pool was first built -- this covers all 2088, including the Classic half."""
    pool = np.load(ROOT / "data" / "processed" / "had100_background_pool.npy", mmap_mode="r")
    chunk = 64
    for i in range(0, pool.shape[0], chunk):
        assert not np.isnan(pool[i:i + chunk]).any(), f"NaN found in patches [{i}:{i + chunk}]"
