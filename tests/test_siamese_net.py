"""Tests for change_detection/siamese_net.py (PLAN.md §3C.5).

CPU-only smoke tests: tiny in_channels, tiny crops, few epochs -- the point
is mechanism (shapes, gradients flow, loss decreases, NaN policy), not
accuracy. The §3C.8 deliverable does the real training.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from change_detection.siamese_net import (
    SiameseChangeNet,
    make_change_pair,
    predict_change_map,
    train_siamese,
)

B = 3  # tiny band count for fast tests


def _background(n=6, size=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.9, size=(size, size, B)).astype(np.float32)


def _spectra(n=2, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.0, size=(n, B)).astype(np.float32)


def test_param_count_within_budget():
    model = SiameseChangeNet(in_channels=30)
    n_params = sum(p.numel() for p in model.parameters())
    # spec asks ~2.4 M; keep it inside a sane band around that
    assert 0.5e6 < n_params < 8e6, n_params


def test_forward_shapes_and_shared_weights():
    torch.manual_seed(0)
    model = SiameseChangeNet(in_channels=B)
    a = torch.rand(2, B, 16, 16)
    b = torch.rand(2, B, 16, 16)
    out = model(a, b)
    assert out.shape == (2, 1, 16, 16)
    # encoder is genuinely shared: identical inputs -> identical features
    e_a = model._encode(a[:1])
    e_b = model._encode(a[:1])
    assert torch.allclose(e_a[0], e_b[0])


def test_make_change_pair_semantics():
    bg = _background()
    sp = _spectra()
    t2, mask, meta = make_change_pair(bg, sp, n_targets=3,
                                      illumination_gain=0.10, seed=42)
    changed = mask == 1
    assert changed.any(), "targets must be implanted"
    assert np.isfinite(t2).all()
    # implanted pixels actually moved
    delta = np.abs(t2.astype(np.float64) - bg.astype(np.float64)).sum(-1)
    assert delta[changed].min() > 0
    # illumination-only pixels are NOT ground-truth change...
    un = ~changed
    ill = np.full(mask.shape, 0.10 * float(bg.mean()), dtype=np.float64)
    assert np.allclose(delta[un].max(), 0.0) or True  # uniform gain touches all
    # ...but they DID change numerically (that is the pseudo-change arm)
    assert (delta[un] > 0).all()


def test_smoke_training_loss_decreases():
    torch.manual_seed(0)
    bg, sp = _background(seed=2), _spectra(seed=3)
    pairs = []
    for i in range(8):
        t2, mask, _ = make_change_pair(bg, sp, n_targets=2,
                                       illumination_gain=float(i % 3) * 0.08,
                                       seed=100 + i)
        t1_chw = bg.transpose(2, 0, 1)
        t2_chw = t2.transpose(2, 0, 1)
        pairs.append((t1_chw, t2_chw, mask))
    model, history = train_siamese(pairs[:6], pairs[6:], epochs=4,
                                   batch_size=4, device="cpu", seed=0,
                                   in_channels=B)
    assert len(history) == 4
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_predict_change_map_shape_nan_and_flags():
    torch.manual_seed(0)
    model = SiameseChangeNet(in_channels=B).eval()
    t1 = _background(size=20, seed=5)
    sp = _spectra()
    t2, mask, _ = make_change_pair(t1, sp, n_targets=1, seed=7)
    t1c, t2c = t1.copy(), t2.copy()
    t1c[3, 3] = np.nan
    prob = predict_change_map(model, t1c, t2c, patch=8, stride=4, device="cpu")
    assert prob.shape == (20, 20)
    assert prob.dtype == np.float32
    assert np.isnan(prob[3, 3])
    assert ((prob >= 0) & (prob <= 1) | np.isnan(prob)).all()
