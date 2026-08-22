"""PLAN.md §3A.10 -- benchmark harness pooling and reporting rules.

§3A.10 makes pooling a correctness question, not a formatting one: ABU's
anomaly density spans 0.084%-2.72%, a 32x range (D13.2), so a pixel-weighted
pool is effectively a report on two scenes. Scene-macro is PRIMARY, micro is
secondary and labelled, and an unlabelled "pooled" figure is banned. These
tests pin that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_benchmark import DEFERRED, Row, _local_rx_params, _metrics, pool


def _row(scene, det, auc, n_anom, n_px=10000, status="ok"):
    return dict(scene=scene, dataset="abu", detector=det, standardized=True,
                n_px=n_px, n_anom=n_anom, anom_frac=n_anom / n_px,
                roc_auc=auc, pr_auc=auc, precision=auc, recall=auc, f1=auc,
                runtime_s=1.0, peak_rss_mb=1.0, components="", status=status, note="")


def test_macro_and_micro_differ_when_scene_densities_differ():
    """The D13.2 property that motivates the whole rule. If a change ever makes
    these identical, the micro column has stopped being weighted and the
    32x-density argument silently stopped applying."""
    df = pd.DataFrame([
        _row("dense", "global_rx", 0.60, n_anom=2720),   # 27.2% -- dominates micro
        _row("sparse", "global_rx", 0.99, n_anom=8),     # 0.08%
    ])
    p = pool(df).iloc[0]
    assert p.roc_auc_MACRO == pytest.approx(0.795)       # equal scene weight
    assert p.roc_auc_micro < p.roc_auc_MACRO             # dragged toward the dense scene
    assert p.roc_auc_micro == pytest.approx((0.60 * 2720 + 0.99 * 8) / 2728)


def test_macro_and_micro_columns_are_distinctly_labelled():
    """An unlabelled 'pooled' column is banned by §3A.10."""
    df = pd.DataFrame([_row("a", "global_rx", 0.9, 100), _row("b", "global_rx", 0.8, 200)])
    cols = set(pool(df).columns)
    assert "roc_auc_MACRO" in cols and "roc_auc_micro" in cols
    assert "roc_auc" not in cols, "a bare 'roc_auc' column would read as an unlabelled pool"
    assert "roc_auc_pooled" not in cols


def test_failed_rows_are_counted_not_dropped():
    """A detector that crashes is a FINDING (D22 found global_rx crashing on
    3/13 ABU scenes). Dropping the row would hide it and inflate the mean."""
    df = pd.DataFrame([
        _row("ok1", "global_rx", 0.9, 100),
        _row("boom", "global_rx", None, 100, status="FAILED:LinAlgError"),
    ])
    p = pool(df).iloc[0]
    assert p.n_scenes == 1
    assert p.n_failed == 1


def test_deferred_rows_name_a_reason():
    """§11.1 P3 items and O9/D21-blocked arms are emitted as marked rows, not
    silently omitted -- a missing row reads as an oversight."""
    assert "kpca_autoencoder" in DEFERRED
    assert "unet_implanted_lib" in DEFERRED
    for name, why in DEFERRED.items():
        assert why.strip(), f"{name} deferred without a reason"


@pytest.mark.parametrize("shape,expected_outer", [((64, 64), 15), ((100, 100), 21),
                                                   ((150, 150), 21), ((120, 120), 21)])
def test_local_rx_window_is_per_dataset(shape, expected_outer):
    """§3A.2: the 64x64 HAD100 params must NOT be carried to larger scenes --
    'at 120x120 it needlessly starves the annulus.' Carrying them over made
    local_rx rank last of five on ABU in the first harness run."""
    cube = np.zeros((*shape, 20), dtype=np.float32)
    assert _local_rx_params(cube)["outer"] == expected_outer


def test_metrics_match_a_hand_computed_reference():
    gt = np.array([[0, 0], [1, 1]], dtype=bool)
    score = np.array([[0.1, 0.2], [0.9, 0.8]], dtype=np.float32)
    row = Row(scene="t", dataset="d", detector="x", standardized=True,
              n_px=4, n_anom=2, anom_frac=0.5)
    _metrics(score, gt, row)
    assert row.roc_auc == pytest.approx(1.0)      # perfectly separated
    assert row.recall == pytest.approx(1.0)


def test_all_nan_score_is_flagged_not_silently_scored():
    gt = np.array([[0, 1], [1, 0]], dtype=bool)
    row = Row(scene="t", dataset="d", detector="x", standardized=True,
              n_px=4, n_anom=2, anom_frac=0.5)
    _metrics(np.full((2, 2), np.nan, dtype=np.float32), gt, row)
    assert row.status == "all_nan"
    assert row.roc_auc is None


def test_degenerate_labels_flagged():
    """A scene with no positives cannot produce an AUC; say so rather than
    emitting a number."""
    gt = np.zeros((2, 2), dtype=bool)
    row = Row(scene="t", dataset="d", detector="x", standardized=True,
              n_px=4, n_anom=0, anom_frac=0.0)
    _metrics(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), gt, row)
    assert row.status == "degenerate_labels"
