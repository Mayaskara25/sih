"""§3B.4 -- LightUNet + training loop.

Runs are kept separate and never averaged together (§3B.4): each spectra
pool / mode combination is its own training run, never merged into one
number. "Early stop on the synthetic validation split" means synthetic
patches built from HELD-OUT SCENES (same GroupShuffleSplit as the train
patches) -- never real GT. RealSegDataset is not touched anywhere in this
module; it belongs to scoring (segmentation/infer.py, not built here).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightUNet(nn.Module):
    """Encoder 32-64-128-256, decoder mirror + skips, GroupNorm, output 1
    logit. ~1.9M params. Input [B, 30, 64, 64]."""

    def __init__(self, in_channels: int = 30):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.bottleneck = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))

        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.out_conv(d1)   # [B, 1, 64, 64] logits


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(-1)
    union = probs.sum(-1) + target.sum(-1)
    return 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + BCE, 0.5/0.5 (§3B.4)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * bce + 0.5 * dice_loss(logits, target)


def train_unet(train_dataset, val_dataset, *, epochs: int = 50, batch_size: int = 16,
                lr: float = 3e-4, weight_decay: float = 1e-4, patience: int = 10,
                amp: bool = False, device: str | None = None,
                checkpoint_path: Path | None = None, in_channels: int = 30,
                num_workers: int = 0, seed: int = 0,
                verbose: bool = False) -> tuple[LightUNet, list[dict]]:
    """AdamW lr=3e-4 wd=1e-4, cosine schedule, batch 16 (§3B.4 defaults; all
    overridable). Early stops on val_dataset -- caller is responsible for
    val_dataset being SYNTHETIC (held-out scenes), never RealSegDataset;
    this function has no opinion on where the data came from, only that it
    must not be refit on it (it never touches reduce_bands or any fitting
    step at all).

    amp DEFAULTS TO FALSE, not §3B.4's stated "AMP fp16" -- verified on this
    machine's actual GPU (GTX 1650, cuDNN 9.2.0, torch 2.13.0+cu130): under
    `torch.autocast`, this model's dec2 Conv2d produces NaN from a fresh
    forward pass with fp32-clean input. Disabling cuDNN entirely
    (`torch.backends.cudnn.enabled = False`) makes the NaN disappear, which
    is consistent with a cuDNN fp16 conv kernel bug on this GPU/driver stack
    -- though a genuine fp16-range/accumulation limitation specific to that
    kernel path is an equally consistent read of the same evidence; either
    way it is not a data-scale or architecture bug on our side, and the fix
    (disable AMP here) is the same regardless of which explanation is right.
    `cudnn.benchmark = True` hides it for the first few calls but it recurs
    once cuDNN's algorithm cache settles (confirmed: 0/20 clean iterations
    in a real optimizer loop).
    fp32 training is fully stable (0/30 NaN) and uses ~244 MB VRAM at
    batch=16 -- nowhere near the 4 GB budget, so AMP buys nothing here
    memory-wise; it was only ever needed for the model this small because
    §3B.4 asked for it, not because fp32 doesn't fit. **This is a real
    concern for §3B.5's SegFormer arm**, which the plan explicitly sizes
    against 4 GB *assuming* AMP works ("batch 8 + grad-accum 2 + AMP at
    4 GB", D8) -- re-run this same isolation test against that model before
    trusting AMP there.

    verbose=True prints one flush=True line per epoch as it happens. Off by
    default (unit tests run dozens of tiny fits and don't want the noise),
    but turn it on for any run long enough that "no output yet" is
    otherwise indistinguishable from a hang -- history was previously only
    printed by the caller after this function returned ALL epochs, which
    on a full-scene-split run made a normally-slow-but-working run look
    identical to a stalled one from the outside.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device == "cuda"

    model = LightUNet(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=len(train_dataset) > batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for patches, masks in train_loader:
            patches, masks = patches.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(patches)
                loss = combined_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for patches, masks in val_loader:
                patches, masks = patches.to(device), masks.to(device)
                val_losses.append(combined_loss(model(patches), masks).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                             lr=scheduler.get_last_lr()[0]))
        if verbose:
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return model, history
