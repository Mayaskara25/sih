"""§3B.4 real-data smoke test: the actual GTX 1650 (D8), the actual cached
background pool (D17), the actual pretext synthetic pipeline (3B.2/3B.3) --
proving the full chain fits in 4GB VRAM and trains, not just that the pieces
type-check in isolation. Small and fast on purpose (a few epochs, a small
patch subset) -- this is a pipeline smoke test, not the §3B.8 experiment.

Also pins a real, verified GPU/driver-stack bug found while building this:
under torch.autocast (AMP fp16), this model's dec2 conv produces NaN on this
GTX 1650 (cuDNN 9.2.0, torch 2.13.0+cu130) even on fp32-clean input --
root-caused by disabling cuDNN entirely, which removes the NaN. See
train_unet()'s docstring. amp now defaults to False; test_amp_is_currently_
unsafe_on_this_gpu below is the regression pin -- if cuDNN/driver updates
ever fix this, that test starts failing, which is the signal to revisit the
default and re-check §3B.5's SegFormer arm (which the plan sizes against
4GB assuming AMP works).
"""
from pathlib import Path

import pytest
import torch

from segmentation.datasets import (
    MANIFEST_PATH,
    POOL_PATH,
    SyntheticSegDataset,
    fit_reduce_bands_transformer,
    train_val_scene_split,
)
from segmentation.train_unet import LightUNet, combined_loss, train_unet

_have_pool = POOL_PATH.exists() and MANIFEST_PATH.exists()
_have_cuda = torch.cuda.is_available()


@pytest.mark.skipif(not _have_pool, reason="background pool not built")
def test_pretext_training_smoke_test_on_real_pool_and_real_gpu(tmp_path):
    split = train_val_scene_split(val_fraction=0.2, seed=0)

    transformer, n_sampled = fit_reduce_bands_transformer(
        train_array_indices=split["train_array_indices"], n_components=30,
        n_sample_pixels=50_000, seed=0)
    assert n_sampled > 0

    train_ds = SyntheticSegDataset(
        mode="pretext", array_indices=split["train_array_indices"][:64],
        transformer=transformer, n_components=30, seed=0)
    val_ds = SyntheticSegDataset(
        mode="pretext", array_indices=split["val_array_indices"][:16],
        transformer=transformer, n_components=30, seed=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, history = train_unet(
        train_ds, val_ds, epochs=3, batch_size=16, device=device,   # amp defaults False
        checkpoint_path=tmp_path / "smoke_ckpt.pt")

    assert isinstance(model, LightUNet)
    assert len(history) == 3
    for row in history:
        assert row["train_loss"] == row["train_loss"]   # not NaN
        assert row["val_loss"] == row["val_loss"]
    assert (tmp_path / "smoke_ckpt.pt").exists()

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        assert peak_mb < 3500, f"peak VRAM {peak_mb:.0f} MB exceeds the 4 GB GTX 1650 budget (D8)"


@pytest.mark.skipif(not _have_cuda, reason="no CUDA GPU available")
def test_amp_is_currently_unsafe_on_this_gpu():
    """The regression pin for the finding above. If this ever starts
    failing (i.e. AMP stops producing NaN), that's good news -- it means a
    driver/cuDNN update fixed the bug, and amp's default in train_unet()
    should be reconsidered, and §3B.5's SegFormer AMP assumption re-checked.
    """
    torch.manual_seed(0)
    model = LightUNet(in_channels=30).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    saw_nan = False
    for _ in range(10):
        x = torch.randn(16, 30, 64, 64, device="cuda")
        target = (torch.rand(16, 1, 64, 64, device="cuda") > 0.7).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            loss = combined_loss(model(x), target)
        if torch.isnan(loss):
            saw_nan = True
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    assert saw_nan, (
        "expected AMP to reproduce NaN on this GPU (known cuDNN 9.2.0 bug) -- "
        "if this fails, the bug may be fixed; see this file's module docstring")
