"""D3 -- percentile + rank normalization, invertibility, NaN handling."""
import numpy as np
from scipy import stats

from anomaly.scoring import percentile_normalize, rank_normalize, threshold_by_percentile


def test_percentile_normalize_range_and_nan_preserving():
    rng = np.random.default_rng(0)
    score = rng.normal(size=(50, 50)).astype(np.float32)
    score[0, 0] = np.nan
    norm, v_lo, v_hi = percentile_normalize(score)
    assert np.isnan(norm[0, 0])
    valid = ~np.isnan(norm)
    assert np.nanmin(norm[valid]) >= 0.0
    assert np.nanmax(norm[valid]) <= 1.0


def test_percentile_normalize_invertible_within_clip_range():
    rng = np.random.default_rng(1)
    score = rng.normal(size=(200, 200)).astype(np.float32)
    norm, v_lo, v_hi = percentile_normalize(score)
    recovered = norm * (v_hi - v_lo) + v_lo
    clipped = np.clip(score, v_lo, v_hi)
    np.testing.assert_allclose(recovered, clipped, atol=1e-4)


def test_rank_normalize_is_uniform():
    rng = np.random.default_rng(2)
    score = rng.normal(size=10_000).astype(np.float32).reshape(100, 100)
    out = rank_normalize(score)
    stat, p = stats.kstest(out.ravel(), "uniform")
    assert p > 0.05


def test_rank_normalize_nan_preserving():
    score = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32)
    out = rank_normalize(score)
    assert np.isnan(out[0, 2])
    assert not np.any(np.isnan(out[~np.isnan(score)]))


def test_threshold_by_percentile_produces_c3_mask():
    norm_score = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
    mask = threshold_by_percentile(norm_score, pct=90.0)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() == 10   # top 10% of 100 pixels
