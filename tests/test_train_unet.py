"""§3B.4 segmentation/train_unet.py -- fast CPU unit tests. The real GPU
smoke test against the cached background pool lives in
test_train_unet_real_gpu.py.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from segmentation.train_unet import LightUNet, combined_loss, dice_loss, train_unet


def test_light_unet_output_shape():
    model = LightUNet(in_channels=30)
    x = torch.randn(2, 30, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_light_unet_param_count_matches_spec():
    model = LightUNet(in_channels=30)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1_700_000 <= n_params <= 2_100_000, f"expected ~1.9M params, got {n_params}"


def test_dice_loss_near_zero_for_perfect_prediction():
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    logits = (target * 2 - 1) * 20.0   # saturated sigmoid -> ~target
    loss = dice_loss(logits, target)
    assert loss.item() < 0.01


def test_dice_loss_high_for_inverted_prediction():
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, :4, :4] = 1.0
    logits = (1 - target) * 20.0 - 10.0   # predicts the opposite region
    loss = dice_loss(logits, target)
    assert loss.item() > 0.8


def test_combined_loss_is_bce_dice_average():
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    logits = torch.randn(2, 1, 8, 8)
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target)
    combined = combined_loss(logits, target)
    assert combined.item() == pytest.approx(0.5 * bce.item() + 0.5 * dice.item(), abs=1e-5)


class _TinyDataset(Dataset):
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.patches = rng.normal(size=(n, 30, 64, 64)).astype(np.float32)
        self.masks = (rng.random((n, 1, 64, 64)) > 0.7).astype(np.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        return torch.from_numpy(self.patches[i]), torch.from_numpy(self.masks[i])


def test_train_unet_runs_on_cpu_and_returns_history_with_expected_keys():
    train_ds = _TinyDataset(12, seed=0)
    val_ds = _TinyDataset(6, seed=1)
    model, history = train_unet(
        train_ds, val_ds, epochs=2, batch_size=4, device="cpu", amp=False, num_workers=0)

    assert isinstance(model, LightUNet)
    assert len(history) == 2
    for row in history:
        assert set(row) == {"epoch", "train_loss", "val_loss", "lr"}
        assert np.isfinite(row["train_loss"])
        assert np.isfinite(row["val_loss"])


def test_train_unet_early_stops_within_patience(tmp_path):
    train_ds = _TinyDataset(8, seed=2)
    val_ds = _TinyDataset(4, seed=3)
    _model, history = train_unet(
        train_ds, val_ds, epochs=50, batch_size=4, device="cpu", amp=False,
        patience=2, checkpoint_path=tmp_path / "ckpt.pt")
    assert len(history) < 50   # patience=2 on a tiny dataset should stop well short of 50
    assert (tmp_path / "ckpt.pt").exists()
