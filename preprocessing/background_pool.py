"""HAD100 background pool assembly (§3B "Background pool sourcing").

522 raw ENVI background scenes (260 AVIRIS-NG + 262 AVIRIS-Classic) -> 2088
harmonized 64x64x184 patches, four corner crops per scene.

Two hard constraints from D11, both structural here rather than left to a
later step to get right:

1. NG and Classic are pooled ONLY after harmonize() runs -- D11.3's join.
   build_background_pool's np.concatenate is that join; nothing upstream of
   it mixes sensors.
2. Split by SOURCE SCENE, never by patch (D11.5) -- the four corner crops of
   one scene overlap by up to ~62 px per axis (worse than the 81x81 case's
   ~47 px for HAD100's smaller background scenes, since overlap = 64 -
   (h - 64)). scene_groups() is the one sanctioned way to derive split
   indices from this pool; anything that indexes the pool array directly by
   patch has to actively bypass it to leak.

Crop geometry matches HAD100/main.py's own four-corner rule (lines 103-111)
exactly -- reproduced as offset arithmetic here, not reimplemented as a
judgement call (D11.2: the pipeline does not re-derive crop geometry main.py
already decided). Source is the RAW ENVI cube (D11.6): feeding
HAD100Dataset/train's band_select output through harmonize() correctly
raises via coverage_ok -- that is the self-defence working, not a bug.

Harmonizing the FULL scene once, then cropping four times from the
harmonized result, is mathematically identical to harmonizing each crop
separately (harmonize() is per-pixel, independent of neighbours) but avoids
redundant interpolation over the ~47-62px of overlap between crops --
verified equal, not just assumed, in tests/test_background_pool.py.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from preprocessing.harmonize import RETAINED_BANDS, harmonize
from preprocessing.raster_loader import load_scene

PATCH_SIZE = 64

# Sensor -> raw-ENVI background subdirectory under HAD100/data/ (D11.1).
SENSOR_DIRS = {
    "aviris_ng": "aviris_ng_normal",
    "aviris": "aviris_normal",
}


def four_corner_offsets(h: int, w: int) -> dict[int, tuple[int, int]]:
    """{crop_index: (row_offset, col_offset)}, matching HAD100/main.py's
    id 0-3 exactly:
        id0 = img[:64,:,:][:,:64,:]      (top-left)
        id1 = img[-64:,:,:][:,:64,:]     (bottom-left)
        id2 = img[:64,:,:][:,-64:,:]     (top-right)
        id3 = img[-64:,:,:][:,-64:,:]    (bottom-right)
    Requires h >= 64 and w >= 64 -- true of every HAD100 background scene
    (minimum observed 66x66, D11.1); asserted here, not assumed.
    """
    if h < PATCH_SIZE or w < PATCH_SIZE:
        raise ValueError(f"scene too small for a {PATCH_SIZE}px crop: {h}x{w}")
    row0, row_n = 0, h - PATCH_SIZE
    col0, col_n = 0, w - PATCH_SIZE
    return {0: (row0, col0), 1: (row_n, col0), 2: (row0, col_n), 3: (row_n, col_n)}


@dataclass(frozen=True)
class BackgroundPatch:
    patch_id: str
    scene_id: str
    sensor: str            # "aviris_ng" | "aviris"
    crop_index: int        # 0-3
    row_offset: int
    col_offset: int
    source_path: str
    array_index: int       # row into the pool tensor this patch occupies


def harmonize_and_crop_scene(hdr_path: Path, *, sensor: str
                              ) -> tuple[np.ndarray, list[BackgroundPatch]]:
    """Load raw ENVI -> harmonize the full scene once -> four corner 64x64
    crops of the harmonized (184-band) result.
    Returns ([4, 64, 64, 184] float32, four BackgroundPatch records with
    array_index left at 0 -- the caller assigns real indices once patches
    are placed into the pool tensor).
    """
    cube, meta = load_scene(hdr_path, source="had100")
    harmonized, new_meta = harmonize(cube, meta)

    h, w = harmonized.shape[:2]
    offsets = four_corner_offsets(h, w)

    patches = np.empty((4, PATCH_SIZE, PATCH_SIZE, RETAINED_BANDS), dtype=np.float32)
    records = []
    for idx, (r0, c0) in offsets.items():
        patches[idx] = harmonized[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
        records.append(BackgroundPatch(
            patch_id=f"{new_meta.scene_id}_{idx}", scene_id=new_meta.scene_id, sensor=sensor,
            crop_index=idx, row_offset=r0, col_offset=c0, source_path=str(hdr_path),
            array_index=0,
        ))
    return patches, records


def _iter_scene_hdrs(had100_root: Path) -> list[tuple[str, Path]]:
    out = []
    for sensor, subdir in SENSOR_DIRS.items():
        for hdr_path in sorted((had100_root / "data" / subdir).glob("*.hdr")):
            out.append((sensor, hdr_path))
    return out


def build_background_pool(had100_root: Path, *, progress=None
                           ) -> tuple[np.ndarray, list[BackgroundPatch]]:
    """Assemble the full pool IN MEMORY: every raw ENVI scene in both
    background subdirectories, harmonized and four-corner-cropped,
    concatenated NG then Classic. The concatenate is D11.3's join -- only
    legal because every patch has already been through harmonize() onto the
    shared 184-band grid.

    For a handful of scenes (tests) only. At full scale (522 scenes, 2088
    patches, 6.29 GB) this holds the whole pool twice at peak -- once as a
    list of per-scene arrays, once as np.concatenate's copy -- which is what
    OOM-killed the first real run on an 8.3 GB-available / no-swap machine.
    build_background_pool_to_disk is the one that scales.
    """
    all_patches: list[np.ndarray] = []
    all_records: list[BackgroundPatch] = []

    for sensor, hdr_path in _iter_scene_hdrs(had100_root):
        patches, records = harmonize_and_crop_scene(hdr_path, sensor=sensor)
        all_patches.append(patches)
        all_records.extend(records)
        if progress is not None:
            progress(hdr_path, sensor)

    pool = np.concatenate(all_patches, axis=0)
    records_indexed = [
        BackgroundPatch(**{**asdict(r), "array_index": i}) for i, r in enumerate(all_records)
    ]
    return pool, records_indexed


def build_background_pool_to_disk(had100_root: Path, pool_path: Path, *, progress=None
                                   ) -> list[BackgroundPatch]:
    """Same assembly as build_background_pool, but the tensor never exists
    fully in RAM: a memmap is preallocated on disk at pool_path and each
    scene's 4 patches are written directly into their slice as computed.
    Peak RAM is O(one scene's working set), not O(pool size). This is the
    one used for the real 522-scene / 2088-patch / 6.29 GB build.
    """
    scenes = _iter_scene_hdrs(had100_root)
    n_patches = len(scenes) * 4

    pool_path = Path(pool_path)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool = np.lib.format.open_memmap(
        pool_path, mode="w+", dtype=np.float32,
        shape=(n_patches, PATCH_SIZE, PATCH_SIZE, RETAINED_BANDS),
    )

    records: list[BackgroundPatch] = []
    for i, (sensor, hdr_path) in enumerate(scenes):
        patches, scene_records = harmonize_and_crop_scene(hdr_path, sensor=sensor)
        pool[i * 4:(i + 1) * 4] = patches
        for j, r in enumerate(scene_records):
            records.append(BackgroundPatch(**{**asdict(r), "array_index": i * 4 + j}))
        if progress is not None:
            progress(hdr_path, sensor)

    pool.flush()
    del pool   # release the memmap; save_pool_manifest reopens read-only for hashing
    return records


def scene_groups(records: list[BackgroundPatch]) -> np.ndarray:
    """Group labels for GroupKFold / GroupShuffleSplit, aligned with the pool
    array's patch order (records[i].array_index == i for a pool built by
    build_background_pool). The ONE sanctioned way to derive train/val
    splits from this pool: splitting on patch index directly would put a
    scene's four heavily-overlapping crops (D11.5) on both sides."""
    return np.array([r.scene_id for r in records])


def save_pool(pool: np.ndarray, records: list[BackgroundPatch], out_dir: Path) -> dict:
    """Writes the tensor, a per-patch manifest CSV, and a summary JSON
    (counts, shape, sha256, build date) for provenance -- the same pattern
    scripts/fetch_*.py use for downloaded data, applied to a derived one."""
    import hashlib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / "had100_background_pool.npy"
    manifest_path = out_dir / "had100_background_manifest.csv"
    summary_path = out_dir / "had100_background_pool_summary.json"

    np.save(tensor_path, pool)

    fieldnames = list(BackgroundPatch.__dataclass_fields__)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    sha256 = hashlib.sha256()
    with open(tensor_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)

    n_ng = sum(1 for r in records if r.sensor == "aviris_ng")
    n_classic = sum(1 for r in records if r.sensor == "aviris")
    summary = dict(
        shape=list(pool.shape), dtype=str(pool.dtype),
        n_patches=len(records), n_ng_patches=n_ng, n_classic_patches=n_classic,
        n_ng_scenes=n_ng // 4, n_classic_scenes=n_classic // 4,
        size_bytes=int(pool.nbytes), sha256=sha256.hexdigest(),
        built=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
