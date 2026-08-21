#!/usr/bin/env python3
"""Pipeline-proof training run for the pretext arm (3B.2/3B.3/3B.4), on the
full real train/val scene split. NOT the §3B.8 experiment (that's a full
multi-arm LODO sweep to convergence + report); this is a bounded
demonstration run -- background pool -> scene split -> reduce_bands fit ->
synthetic pretext dataset -> LightUNet training on the real GTX 1650 --
actually works and shows a loss trend on the full split, not just the tiny
subset in tests/test_train_unet_real_gpu.py (which is the actual proof that
the chain works and fits in 4GB VRAM; this script only adds a visible trend
on more data). Kept short (8 epochs, num_workers=4, verbose per-epoch
printing) on purpose: an earlier 15-epoch/num_workers=0 version was killed
after ~20min on a mistaken hang diagnosis -- it was actually just slow
(~67ms/item x 2085 items/epoch, single-threaded) with zero progress output
until the very end. num_workers=4 and verbose=True fix both problems.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.datasets import (  # noqa: E402
    fit_reduce_bands_transformer,
    SyntheticSegDataset,
    train_val_scene_split,
)
from segmentation.train_unet import train_unet  # noqa: E402

OUT_DIR = ROOT / "experiments" / "seg_arch"


def main() -> int:
    t0 = time.time()
    split = train_val_scene_split(val_fraction=0.2, seed=0)
    print(f"split: {len(split['train_scene_ids'])} train scenes / "
          f"{len(split['val_scene_ids'])} val scenes, "
          f"{len(split['train_array_indices'])} / {len(split['val_array_indices'])} patches")

    transformer, n_sampled = fit_reduce_bands_transformer(
        train_array_indices=split["train_array_indices"], n_components=30,
        n_sample_pixels=200_000, seed=0)
    print(f"reduce_bands fit on {n_sampled} train-split pixels in {time.time() - t0:.1f}s")

    train_ds = SyntheticSegDataset(
        mode="pretext", array_indices=split["train_array_indices"],
        transformer=transformer, n_components=30, seed=0)
    val_ds = SyntheticSegDataset(
        mode="pretext", array_indices=split["val_array_indices"],
        transformer=transformer, n_components=30, seed=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    model, history = train_unet(
        train_ds, val_ds, epochs=8, batch_size=16, patience=8, num_workers=4,
        verbose=True, checkpoint_path=OUT_DIR / "unet_pretext_smoke.pt")
    print(f"trained {len(history)} epochs in {time.time() - t1:.1f}s")

    (OUT_DIR / "unet_pretext_smoke_history.json").write_text(json.dumps(history, indent=2))

    first, last = history[0], history[-1]
    print(f"\ntrain_loss {first['train_loss']:.4f} -> {last['train_loss']:.4f}")
    print(f"val_loss   {first['val_loss']:.4f} -> {last['val_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
