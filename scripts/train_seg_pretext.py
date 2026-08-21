#!/usr/bin/env python3
"""§3B.8 pretext arm, trained to convergence — the deliverable checkpoint.

This is the real run, not `smoke_train_pretext.py`'s 8-epoch trend
demonstration. It produces the checkpoint that `segmentation/infer.py`
(§3B.6), Phase 4's end-to-end re-run and Phase 7's `demo.py` all consume.

**Only one of §3B.8's five learned arms is trainable today**, and that is
worth stating where the next reader will hit it. `unet_lodo_abu`,
`unet_lodo_hyd` and `unet_all_real` are suspended pending O9 (D19 — ABU and
HYDICE ship no wavelength array, so real target spectra cannot be put on the
canonical grid). `unet_implanted_lib` is blocked on a *different* and much
more recoverable gap: `SPECTRA_POOLS["lib"]` has never been fetched, because
`scripts/fetch_speclib.py` (§1.6) does not exist yet. So §3B.8's headline
`implanted_lib` vs `pretext` comparison is currently a one-sided table.
Build the fetcher and this becomes a two-arm run.

The transformer is persisted alongside the checkpoint. It is fit on the
TRAIN SPLIT ONLY (D15) and inference MUST reuse this exact fitted object —
re-fitting it at inference time on the scoring scene is precisely the leak
D15 deferred `reduce_bands` out of §3A.1 to prevent.
"""
import json
import pickle
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
EPOCHS = 60
PATIENCE = 12
N_COMPONENTS = 30


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split = train_val_scene_split(val_fraction=0.2, seed=0)
    print(f"split: {len(split['train_scene_ids'])} train scenes / "
          f"{len(split['val_scene_ids'])} val scenes, "
          f"{len(split['train_array_indices'])} / {len(split['val_array_indices'])} patches",
          flush=True)

    transformer, n_sampled = fit_reduce_bands_transformer(
        train_array_indices=split["train_array_indices"], n_components=N_COMPONENTS,
        n_sample_pixels=200_000, seed=0)
    print(f"reduce_bands fit on {n_sampled} TRAIN-split pixels in {time.time() - t0:.1f}s",
          flush=True)

    with open(OUT_DIR / "reduce_bands_transformer.pkl", "wb") as fh:
        pickle.dump(transformer, fh)

    train_ds = SyntheticSegDataset(
        mode="pretext", array_indices=split["train_array_indices"],
        transformer=transformer, n_components=N_COMPONENTS, seed=0)
    val_ds = SyntheticSegDataset(
        mode="pretext", array_indices=split["val_array_indices"],
        transformer=transformer, n_components=N_COMPONENTS, seed=1)

    t1 = time.time()
    model, history = train_unet(
        train_ds, val_ds, epochs=EPOCHS, batch_size=16, patience=PATIENCE,
        num_workers=4, verbose=True,
        checkpoint_path=OUT_DIR / "unet_pretext.pt")
    elapsed = time.time() - t1
    print(f"trained {len(history)} epochs in {elapsed:.1f}s", flush=True)

    best = min(history, key=lambda h: h["val_loss"])
    meta = dict(
        arm="unet_pretext",
        spectra_provenance="none",
        scored_on="had100/test only",
        epochs_run=len(history), epochs_requested=EPOCHS, patience=PATIENCE,
        n_components=N_COMPONENTS,
        train_scenes=len(split["train_scene_ids"]),
        val_scenes=len(split["val_scene_ids"]),
        train_patches=len(split["train_array_indices"]),
        val_patches=len(split["val_array_indices"]),
        reduce_bands_fit_pixels=int(n_sampled),
        best_epoch=best["epoch"], best_val_loss=best["val_loss"],
        train_seconds=elapsed, train_host="local",
        device="cuda", amp=False,
    )
    (OUT_DIR / "unet_pretext_history.json").write_text(json.dumps(history, indent=2))
    (OUT_DIR / "unet_pretext_meta.json").write_text(json.dumps(meta, indent=2))

    first, last = history[0], history[-1]
    print(f"\ntrain_loss {first['train_loss']:.4f} -> {last['train_loss']:.4f}")
    print(f"val_loss   {first['val_loss']:.4f} -> {last['val_loss']:.4f}"
          f"  (best {best['val_loss']:.4f} @ epoch {best['epoch']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
