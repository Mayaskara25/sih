"""§3B background pool -- logic-level tests on a handful of real scenes, not
the full 522 (that's scripts/build_background_pool.py, run once against disk).
"""
from pathlib import Path

import numpy as np
import pytest

from preprocessing.background_pool import (
    BackgroundPatch,
    RETAINED_BANDS,
    build_background_pool,
    build_background_pool_to_disk,
    four_corner_offsets,
    harmonize_and_crop_scene,
    save_pool,
    save_pool_manifest,
    scene_groups,
)
from preprocessing.harmonize import harmonize
from preprocessing.raster_loader import load_scene

ROOT = Path(__file__).resolve().parents[1]
HAD100_ROOT = ROOT / "data" / "benchmark" / "had100" / "HAD100"
NG_HDR = HAD100_ROOT / "data" / "aviris_ng_normal" / "ang20191004t185054_13.hdr"
CLASSIC_81_HDR = HAD100_ROOT / "data" / "aviris_normal" / "f170507t01p00r10_1.hdr"
CLASSIC_66_HDR = HAD100_ROOT / "data" / "aviris_normal" / "f090710t01p00r10_10.hdr"

_have_had100 = NG_HDR.exists() and CLASSIC_81_HDR.exists() and CLASSIC_66_HDR.exists()


# --- crop geometry matches HAD100/main.py exactly ---------------------------

def test_four_corner_offsets_matches_main_py_slicing():
    # 81x81 (real NG/Classic background shape): main.py's img[-64:] on 81
    # rows means rows 17:81 -> offset 17. D11.5's ~47px overlap = 64-17.
    offsets = four_corner_offsets(81, 81)
    assert offsets == {0: (0, 0), 1: (17, 0), 2: (0, 17), 3: (17, 17)}


def test_four_corner_offsets_matches_main_py_slicing_66x66():
    # smaller background scene: overlap is even larger than the 81x81 case.
    offsets = four_corner_offsets(66, 66)
    assert offsets == {0: (0, 0), 1: (2, 0), 2: (0, 2), 3: (2, 2)}


def test_four_corner_offsets_non_square():
    offsets = four_corner_offsets(71, 81)
    assert offsets == {0: (0, 0), 1: (7, 0), 2: (0, 17), 3: (7, 17)}


def test_four_corner_offsets_rejects_too_small():
    with pytest.raises(ValueError):
        four_corner_offsets(50, 81)


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_four_corner_offsets_matches_main_py_on_real_cube_by_direct_slice():
    """Cross-check against main.py's literal slicing (img[:64]/img[-64:]),
    not just the offset arithmetic re-deriving the same numbers."""
    cube, _meta = load_scene(NG_HDR, source="had100")
    h, w = cube.shape[:2]
    offsets = four_corner_offsets(h, w)

    expected = {
        0: cube[:64, :, :][:, :64, :],
        1: cube[-64:, :, :][:, :64, :],
        2: cube[:64, :, :][:, -64:, :],
        3: cube[-64:, :, :][:, -64:, :],
    }
    for idx, (r0, c0) in offsets.items():
        got = cube[r0:r0 + 64, c0:c0 + 64, :]
        np.testing.assert_array_equal(got, expected[idx])


# --- harmonize-then-crop == crop-then-harmonize (order independence) -------

@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
@pytest.mark.parametrize("hdr_path", [NG_HDR, CLASSIC_81_HDR, CLASSIC_66_HDR],
                          ids=["ng_81", "classic_81", "classic_66"])
def test_harmonize_full_scene_then_crop_equals_crop_then_harmonize(hdr_path):
    """harmonize_and_crop_scene harmonizes the FULL scene once, then crops --
    4x cheaper than harmonizing each crop separately IF the two orders give
    identical output. Verified here, not assumed: harmonize() is per-pixel
    and has no cross-pixel dependency, so cropping and harmonizing commute.
    """
    sensor = "aviris_ng" if "ng" in hdr_path.parts[-2] else "aviris"
    patches, records = harmonize_and_crop_scene(hdr_path, sensor=sensor)

    cube, meta = load_scene(hdr_path, source="had100")
    for record, patch in zip(records, patches):
        r0, c0 = record.row_offset, record.col_offset
        crop = cube[r0:r0 + 64, c0:c0 + 64, :]
        crop_harmonized, _ = harmonize(crop, meta)
        np.testing.assert_allclose(patch, crop_harmonized, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_harmonize_and_crop_scene_shape_and_provenance():
    patches, records = harmonize_and_crop_scene(NG_HDR, sensor="aviris_ng")
    assert patches.shape == (4, 64, 64, RETAINED_BANDS)
    assert patches.dtype == np.float32
    assert len(records) == 4
    scene_ids = {r.scene_id for r in records}
    assert len(scene_ids) == 1   # all four crops share one source scene
    assert {r.crop_index for r in records} == {0, 1, 2, 3}
    assert all(r.sensor == "aviris_ng" for r in records)


# --- scene_groups: the one sanctioned split-safety mechanism ----------------

def test_scene_groups_collapses_four_crops_to_one_label():
    records = [
        BackgroundPatch(f"a_{i}", "scene_a", "aviris_ng", i, 0, 0, "p", i) for i in range(4)
    ] + [
        BackgroundPatch(f"b_{i}", "scene_b", "aviris", i, 0, 0, "p", 4 + i) for i in range(4)
    ]
    groups = scene_groups(records)
    assert len(np.unique(groups)) == 2
    assert (groups[:4] == "scene_a").all()
    assert (groups[4:] == "scene_b").all()


# --- assembly on a handful of real scenes (not all 522) ---------------------

@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_build_background_pool_stacks_ng_and_classic(tmp_path, monkeypatch):
    """Exercises build_background_pool's directory-scanning + concatenate
    (the D11.3 join) against a small real subset, by pointing it at a tmp
    HAD100 root containing only 2 NG + 2 Classic scenes."""
    fake_root = tmp_path / "HAD100"
    ng_dir = fake_root / "data" / "aviris_ng_normal"
    classic_dir = fake_root / "data" / "aviris_normal"
    ng_dir.mkdir(parents=True)
    classic_dir.mkdir(parents=True)

    real_ng_dir = HAD100_ROOT / "data" / "aviris_ng_normal"
    real_classic_dir = HAD100_ROOT / "data" / "aviris_normal"
    ng_hdrs = sorted(real_ng_dir.glob("*.hdr"))[:2]
    classic_hdrs = sorted(real_classic_dir.glob("*.hdr"))[:2]

    for hdr in ng_hdrs:
        (ng_dir / hdr.name).symlink_to(hdr)
        (ng_dir / hdr.with_suffix(".dat").name).symlink_to(hdr.with_suffix(".dat"))
    for hdr in classic_hdrs:
        (classic_dir / hdr.name).symlink_to(hdr)
        (classic_dir / hdr.with_suffix(".dat").name).symlink_to(hdr.with_suffix(".dat"))

    pool, records = build_background_pool(fake_root)

    assert pool.shape == (16, 64, 64, RETAINED_BANDS)   # 4 scenes x 4 crops
    assert pool.dtype == np.float32
    assert sum(1 for r in records if r.sensor == "aviris_ng") == 8
    assert sum(1 for r in records if r.sensor == "aviris") == 8
    assert [r.array_index for r in records] == list(range(16))
    assert len(np.unique(scene_groups(records))) == 4


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_build_background_pool_to_disk_matches_in_memory(tmp_path):
    """The memmap-backed path (used for the real 522-scene build, since the
    in-memory path OOM-killed on this machine's 8.3 GB-available/no-swap
    setup) must produce byte-identical output to the in-memory path."""
    fake_root = tmp_path / "HAD100"
    ng_dir = fake_root / "data" / "aviris_ng_normal"
    classic_dir = fake_root / "data" / "aviris_normal"
    ng_dir.mkdir(parents=True)
    classic_dir.mkdir(parents=True)

    real_ng_dir = HAD100_ROOT / "data" / "aviris_ng_normal"
    real_classic_dir = HAD100_ROOT / "data" / "aviris_normal"
    ng_hdrs = sorted(real_ng_dir.glob("*.hdr"))[:2]
    classic_hdrs = sorted(real_classic_dir.glob("*.hdr"))[:2]
    for hdr in ng_hdrs:
        (ng_dir / hdr.name).symlink_to(hdr)
        (ng_dir / hdr.with_suffix(".dat").name).symlink_to(hdr.with_suffix(".dat"))
    for hdr in classic_hdrs:
        (classic_dir / hdr.name).symlink_to(hdr)
        (classic_dir / hdr.with_suffix(".dat").name).symlink_to(hdr.with_suffix(".dat"))

    in_memory_pool, in_memory_records = build_background_pool(fake_root)

    pool_path = tmp_path / "out" / "pool.npy"
    disk_records = build_background_pool_to_disk(fake_root, pool_path)
    disk_pool = np.load(pool_path)

    np.testing.assert_array_equal(disk_pool, in_memory_pool)
    assert [r.__dict__ for r in disk_records] == [r.__dict__ for r in in_memory_records]

    summary = save_pool_manifest(disk_records, pool_path, tmp_path / "out")
    assert summary["shape"] == [16, 64, 64, RETAINED_BANDS]
    assert summary["n_ng_scenes"] == 2
    assert summary["n_classic_scenes"] == 2
    with open(tmp_path / "out" / "had100_background_manifest.csv") as f:
        assert len(f.readlines()) == 17   # header + 16 patches


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_save_pool_roundtrip(tmp_path):
    patches_ng, records_ng = harmonize_and_crop_scene(NG_HDR, sensor="aviris_ng")
    patches_c, records_c = harmonize_and_crop_scene(CLASSIC_81_HDR, sensor="aviris")
    pool = np.concatenate([patches_ng, patches_c], axis=0)
    records = [
        BackgroundPatch(**{**r.__dict__, "array_index": i})
        for i, r in enumerate(records_ng + records_c)
    ]

    summary = save_pool(pool, records, tmp_path)

    reloaded = np.load(tmp_path / "had100_background_pool.npy")
    np.testing.assert_array_equal(reloaded, pool)
    assert summary["shape"] == [8, 64, 64, RETAINED_BANDS]
    assert summary["n_ng_scenes"] == 1
    assert summary["n_classic_scenes"] == 1

    import csv
    with open(tmp_path / "had100_background_manifest.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert rows[0]["scene_id"] == records[0].scene_id
