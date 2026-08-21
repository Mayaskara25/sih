#!/usr/bin/env python3
"""Builds and caches the HAD100 background pool (§3B): 522 raw ENVI scenes
(260 AVIRIS-NG + 262 AVIRIS-Classic) -> 2088 harmonized 64x64x184 patches,
four corner crops per scene, stacked into one [2088, 64, 64, 184] float32
tensor (~6.29 GB) plus a per-patch manifest.

Not run under pytest -- like scripts/fetch_had100.py, this is a one-time
(or re-run-on-demand) data-prep step against files already on disk.
tests/test_background_pool.py covers the underlying logic in
preprocessing/background_pool.py on a handful of real scenes, not all 522.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAD100_ROOT = ROOT / "data" / "benchmark" / "had100" / "HAD100"
OUT_DIR = ROOT / "data" / "processed"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from preprocessing.background_pool import build_background_pool_to_disk, save_pool_manifest

    if not HAD100_ROOT.exists():
        print(f"MISSING: {HAD100_ROOT} -- run the fetcher first", file=sys.stderr)
        return 2

    pool_path = OUT_DIR / "had100_background_pool.npy"

    t0 = time.time()
    n_done = 0

    def progress(hdr_path: Path, sensor: str) -> None:
        nonlocal n_done
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/522 scenes ({sensor}: {hdr_path.name})", file=sys.stderr)

    # Written incrementally via a preallocated memmap -- the pool (6.29 GB)
    # never exists fully in RAM. An earlier in-memory version (list of
    # per-scene arrays + np.concatenate's copy, ~2x peak) was OOM-killed on
    # this machine's 8.3 GB available / no swap.
    records = build_background_pool_to_disk(HAD100_ROOT, pool_path, progress=progress)
    elapsed = time.time() - t0
    print(f"harmonized + cropped {len(records) // 4} scenes -> {len(records)} patches "
          f"in {elapsed:.1f}s")

    summary = save_pool_manifest(records, pool_path, OUT_DIR)
    print(f"wrote {pool_path} "
          f"({summary['size_bytes'] / 1e9:.2f} GB, sha256 {summary['sha256'][:16]}...)")
    print(f"wrote {OUT_DIR / 'had100_background_manifest.csv'} ({summary['n_patches']} rows)")

    fail = []
    if summary["shape"] != [2088, 64, 64, 184]:
        fail.append(f"shape = {summary['shape']}, expected [2088, 64, 64, 184]")
    if summary["n_ng_scenes"] != 260:
        fail.append(f"n_ng_scenes = {summary['n_ng_scenes']}, expected 260")
    if summary["n_classic_scenes"] != 262:
        fail.append(f"n_classic_scenes = {summary['n_classic_scenes']}, expected 262")

    if fail:
        print("\nD11 INVARIANTS VIOLATED:", file=sys.stderr)
        for f in fail:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("all D11 background-pool invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
