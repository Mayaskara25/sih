"""Tests for change_detection/temporal_baseline.py (PLAN.md §3C.6)."""
from __future__ import annotations

import numpy as np
import pytest

from change_detection.temporal_baseline import TemporalBaseline


def _stack(n=6, h=4, w=5, b=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 1.0, size=(h, w, b)).astype(np.float32)
            for _ in range(n)]


def test_median_mad_match_batch_computation():
    epochs = _stack()
    tb = TemporalBaseline()
    for e in epochs:
        tb.add_epoch(e)
    med, mad = tb.baseline()
    ref_stack = np.stack(epochs)
    assert np.allclose(med, np.median(ref_stack, axis=0), atol=1e-5)
    ref_mad = np.median(np.abs(ref_stack - med[None]), axis=0)
    assert np.allclose(mad, ref_mad, atol=1e-5)


def test_streaming_equals_batch_with_window():
    epochs = _stack(n=8, seed=2)
    tb_stream, tb_batch = TemporalBaseline(window=4), TemporalBaseline(window=4)
    for i, e in enumerate(epochs):
        tb_stream.add_epoch(e)          # rolling window evicts oldest
        if i >= len(epochs) - 4:
            tb_batch.add_epoch(e)
    m1, d1 = tb_stream.baseline()
    m2, d2 = tb_batch.baseline()
    assert np.allclose(m1, m2) and np.allclose(d1, d2)


def test_change_score_flags_implanted_outlier():
    epochs = _stack(seed=4)
    tb = TemporalBaseline()
    for e in epochs:
        tb.add_epoch(e)
    probe = epochs[-1].copy()
    probe[1, 1] += 50.0                 # huge departure from a stable history
    score = tb.change_score(probe)
    assert score.shape == probe.shape
    background = np.ones(probe.shape[:2], dtype=bool)
    background[1, 1] = False
    assert score[1, 1].min() > 10 * score[background].mean()


def test_zero_mad_pixel_still_scores():
    flat = np.full((3, 3, 2), 0.5, dtype=np.float32)
    tb = TemporalBaseline()
    for _ in range(5):
        tb.add_epoch(flat.copy())
    probe = flat.copy()
    probe[2, 2] += 1.0
    score = tb.change_score(probe)
    assert (score[0, 0] == 0.0).all()   # no departure -> zero score, no NaN
    assert score[2, 2].min() > 100.0    # floor prevents division by zero


def test_nan_in_any_epoch_poisons_baseline_positionally():
    epochs = _stack(seed=6)
    epochs[2][0, 0] = np.nan
    tb = TemporalBaseline()
    for e in epochs:
        tb.add_epoch(e)
    med, mad = tb.baseline()
    assert np.isnan(med[0, 0]).all() and np.isnan(mad[0, 0]).all()
    assert np.isfinite(med[1:, ...]).all()


def test_nan_input_pixel_scores_nan():
    epochs = _stack(seed=8)
    tb = TemporalBaseline()
    for e in epochs:
        tb.add_epoch(e)
    probe = epochs[-1].copy()
    probe[3, 4] = np.nan
    score = tb.change_score(probe)
    assert np.isnan(score[3, 4]).all()
    assert np.isfinite(score[:-1, :4]).all()


def test_validation_errors():
    with pytest.raises(ValueError):
        TemporalBaseline(window=0)
    tb = TemporalBaseline()
    with pytest.raises(RuntimeError):
        tb.change_score(np.zeros((2, 2, 2), dtype=np.float32))
    tb.add_epoch(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError):     # shape drift
        tb.add_epoch(np.zeros((3, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError):     # wrong ndim
        tb.add_epoch(np.zeros((2, 2), dtype=np.float32))
