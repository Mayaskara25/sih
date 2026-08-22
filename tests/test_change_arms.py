"""Tests for scripts/run_change_arms.py metric helpers (PLAN.md §3C.8)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_change_arms import (  # noqa: E402
    pseudo_change_rate,
    rank_auc,
    threshold_at_recall,
)


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / max(hi - lo, 1e-9)


def test_rank_auc_perfect_and_reversed():
    gt = np.array([0, 0, 1, 1], dtype=bool)
    assert rank_auc(_norm([0.1, 0.2, 8.0, 9.0]), gt) == 1.0
    assert rank_auc(_norm([8.0, 9.0, 0.1, 0.2]), gt) == 0.0


def test_rank_auc_nan_ignored():
    gt = np.array([0, 1, 0, 1], dtype=bool)
    s = _norm([5.0, 6.0, np.nan, np.nan])
    assert rank_auc(s, gt) == 1.0


def test_rank_auc_degenerate_returns_nan():
    gt = np.zeros(4, dtype=bool)
    s = _norm([1, 2, 3, 4])
    assert np.isnan(rank_auc(s, gt))


def test_threshold_at_recall_hits_target_and_metrics_consistent():
    rng = np.random.default_rng(0)
    score = _norm(rng.uniform(size=1000))
    gt = np.zeros(1000, dtype=bool)
    gt[:50] = True
    score[gt] += 1.0                       # positives separably higher
    thr, m = threshold_at_recall(score.astype(np.float32), gt, recall_target=0.95)
    pred = score >= thr
    rec = (pred & gt).sum() / gt.sum()
    assert rec >= 0.95
    assert m["recall"] == pytest.approx(rec, abs=1e-6)
    assert 0.0 <= m["precision"] <= 1.0 and 0.0 <= m["f1"] <= 1.0


def test_pseudo_change_rate_counts_flagged_fraction():
    score = np.full((10,), 0.5, dtype=np.float32)
    score[:5] = 0.9                        # flagged region
    illum = np.zeros(10, dtype=bool)
    illum[:] = True
    rate = pseudo_change_rate(score, illum, thr=0.75)
    assert rate == pytest.approx(0.5)


def test_pseudo_change_rate_empty_region_is_nan():
    score = np.ones(4, dtype=np.float32)
    illum = np.zeros(4, dtype=bool)
    assert np.isnan(pseudo_change_rate(score, illum, thr=0.5))
