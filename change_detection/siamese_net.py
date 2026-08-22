"""PLAN.md §3C.5 -- SiameseChangeNet, the learned change-detection arm.

Shared-weight encoder over both epochs, channel-attention fusion of the two
feature stacks, lightweight decoder to a per-pixel change logit. Trained on
SYNTHETIC change pairs (t2-only implant via `segmentation/synth.py`, same D7
discipline as 3B: train on synthetic, score on real) and scored patch-wise
on full scenes by `predict_change_map`.

AMP note: fp32 only -- see segmentation/train_unet.py for the measured
cuDNN fp16 conv NaN on this machine's GPU; the same reasoning applies here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segmentation.synth import implant_targets


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SiameseChangeNet(nn.Module):
    """Shared-weight siamese encoder + attention fusion + light decoder.

    Input: two [N, B, H, W] epoch stacks. Output: [N, 1, H, W] change logit.
    Sized ~2.4 M params at in_channels=30 (measured, see __init__).
    """

    def __init__(self, in_channels: int = 30, width: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        w1, w2, w3 = width
        # shared across BOTH epochs AND all encoder stages' first layer is
        # deliberately per-stage -- sharing means ONE encoder applied twice.
        self.enc1 = _ConvBlock(in_channels, w1)
        self.enc2 = _ConvBlock(w1, w2)
        self.enc3 = _ConvBlock(w2, w3)
        self.pool = nn.MaxPool2d(2)

        # channel attention over the two feature stacks: gate = sigmoid of a
        # learned function of [|f_t1 - f_t2| ; f_t1 * f_t2] per stage.
        self.att1 = nn.Conv2d(2 * w1, w1, 1)
        self.att2 = nn.Conv2d(2 * w2, w2, 1)
        self.att3 = nn.Conv2d(2 * w3, w3, 1)

        self.dec2 = _ConvBlock(w3 + w2, w2)
        self.dec1 = _ConvBlock(w2 + w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        e1a, e2a, e3a = self._encode(t1)
        e1b, e2b, e3b = self._encode(t2)

        f3 = self._fuse(e3a, e3b, self.att3)
        u2 = F.interpolate(f3, size=e2a.shape[-2:], mode="bilinear", align_corners=False)
        f2 = self.dec2(torch.cat([u2, self._fuse(e2a, e2b, self.att2)], dim=1))
        u1 = F.interpolate(f2, size=e1a.shape[-2:], mode="bilinear", align_corners=False)
        f1 = self.dec1(torch.cat([u1, self._fuse(e1a, e1b, self.att1)], dim=1))
        return self.head(f1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        d1 = self.enc1(x)
        d2 = self.enc2(self.pool(d1))
        d3 = self.enc3(self.pool(d2))
        return d1, d2, d3

    @staticmethod
    def _fuse(a: torch.Tensor, b: torch.Tensor,
              att: nn.Conv2d) -> torch.Tensor:
        diff = torch.abs(a - b)
        prod = a * b
        gate = torch.sigmoid(att(torch.cat([diff, prod], dim=1)))
        return gate * diff + (1.0 - gate) * prod


def make_change_pair(background: np.ndarray, target_spectra: np.ndarray, *,
                     n_targets: int, illumination_gain: float = 0.0,
                     seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build one synthetic bi-temporal training pair from a clean background.

    t1 = background; t2 = t1 with (a) an optional uniform illumination gain
    (pseudo-change -- NO ground-truth change pixels) and (b) `n_targets`
    implanted ONLY into t2 (ground truth mask comes free).

    Returns (t2, change_mask[C3 uint8], meta).
    """
    if illumination_gain != 0.0:
        base = background.astype(np.float64) * (1.0 + illumination_gain)
        base = np.clip(base, 0.0, None).astype(np.float32)
    else:
        base = background.copy()
    t2, mask, meta = implant_targets(base, target_spectra,
                                     n_targets=n_targets, seed=seed)
    meta["illumination_gain"] = illumination_gain
    return t2, mask, meta


def train_siamese(train_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                  val_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], *,
                  epochs: int = 20, batch_size: int = 8, lr: float = 3e-4,
                  weight_decay: float = 1e-4, device: str | None = None,
                  in_channels: int = 30, seed: int = 0,
                  verbose: bool = False
                  ) -> tuple[SiameseChangeNet, list[dict]]:
    """Fit on synthetic (t1, t2, mask) triples. FP32 only (see module docstring).

    Pairs are [C, H, W] float32 arrays with a common small size; masks are
    [H, W] uint8. Returns (model, history).
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseChangeNet(in_channels=in_channels).to(device)

    def _tensor(cube: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(cube)).float()

    train_loader = DataLoader(_PairDataset(train_pairs), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(_PairDataset(val_pairs), batch_size=batch_size,
                            shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=max(epochs, 1))

    # class imbalance control computed over the WHOLE train set once --
    # anchoring on the first batch lets a positives-free batch produce an
    # enormous pos_weight that wrecks optimization.
    pos = sum(float(m.sum()) for _, _, m in train_pairs)
    neg = sum(m.size * 1.0 for _, _, m in train_pairs) - pos
    pos_weight = torch.tensor(max(neg / max(pos, 1.0), 1.0)).to(device)

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        tr_loss = n_tr = 0
        for batch in train_loader:
            t1, t2, mask = (batch["t1"].to(device), batch["t2"].to(device),
                            batch["mask"].to(device))
            logits = model(t1, t2).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(
                logits, mask, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.detach())
            n_tr += 1
        scheduler.step()

        model.eval()
        va_loss = n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                t1 = batch["t1"].to(device)
                t2 = batch["t2"].to(device)
                mask = batch["mask"].to(device)
                logits = model(t1, t2).squeeze(1)
                va_loss += float(F.binary_cross_entropy_with_logits(
                    logits, mask, pos_weight=pos_weight))
                n_va += 1
        rec = dict(epoch=epoch, train_loss=tr_loss / max(n_tr, 1),
                   val_loss=va_loss / max(n_va, 1))
        history.append(rec)
        if verbose:
            print(f"epoch {rec['epoch']}: train {rec['train_loss']:.4f} "
                  f"val {rec['val_loss']:.4f}", flush=True)

    return model, history


class _PairDataset(torch.utils.data.Dataset):
    """(t1, t2, mask) triples -> dict of tensors for the siamese loader."""

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        t1, t2, mask = self.pairs[i]
        return {
            "t1": torch.from_numpy(np.ascontiguousarray(t1)).float(),
            "t2": torch.from_numpy(np.ascontiguousarray(t2)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).float(),
        }


def predict_change_map(model: SiameseChangeNet, cube_t1: np.ndarray,
                       cube_t2: np.ndarray, *, patch: int = 64,
                       stride: int | None = None,
                       device: str | None = None) -> np.ndarray:
    """Score a full scene pair patch-wise -> [H, W] float32 change probability.

    NaN nodata propagates positionally: output is NaN wherever either input
    cube has a NaN band at that pixel.
    """
    stride = stride or patch
    was_training = model.training
    model.eval()
    dev = device or next(model.parameters()).device
    h, w = cube_t1.shape[:2]
    prob = np.full((h, w), np.nan, dtype=np.float32)
    valid = ~(np.isnan(cube_t1).any(axis=-1) | np.isnan(cube_t2).any(axis=-1))

    with torch.no_grad():
        for r0 in range(0, h, stride):
            for c0 in range(0, w, stride):
                r1, c1 = min(r0 + patch, h), min(c0 + patch, w)
                pr, pc = r1 - r0, c1 - c0
                if pr < 2 or pc < 2:
                    continue
                a = cube_t1[r0:r1, c0:c1]
                b = cube_t2[r0:r1, c0:c1]
                nan_a = np.isnan(a).any(axis=-1)
                nan_b = np.isnan(b).any(axis=-1)
                fill_a = np.nan_to_num(a, nan=float(np.nanmean(a)))
                fill_b = np.nan_to_num(b, nan=float(np.nanmean(b)))
                ta = torch.from_numpy(fill_a[None].transpose(0, 3, 1, 2)).float().to(dev)
                tb = torch.from_numpy(fill_b[None].transpose(0, 3, 1, 2)).float().to(dev)
                p = torch.sigmoid(model(ta, tb))[0, 0].cpu().numpy()
                local_nan = (nan_a | nan_b)[None]
                p = np.where(local_nan, np.nan, p).astype(np.float32)
                prob[r0:r1, c0:c1] = p
    if was_training:
        model.train()
    prob[~valid] = np.nan
    return prob
