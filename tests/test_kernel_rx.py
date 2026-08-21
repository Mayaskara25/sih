"""PLAN.md §3A.3 -- kernel_rx correctness gates + the ABU-Beach-2 AUC report.

local_rx (also named in §3A.3's accept criterion, "reported alongside global
and local RX") does not exist in this repo yet (anomaly/ only ships rx.py,
scoring.py, kernel_rx.py, crd.py as of this branch) -- see the discrepancy
noted in this branch's report. Only global_rx is available as a baseline.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from anomaly.kernel_rx import _median_heuristic_gamma, kernel_rx
from anomaly.rx import global_rx
from preprocessing.raster_loader import load_scene

ROOT = Path(__file__).resolve().parents[1]
ABU_BEACH_2 = ROOT / "data" / "benchmark" / "abu" / "abu-beach-2.mat"
_have_abu_beach_2 = ABU_BEACH_2.exists()


# --- determinism ------------------------------------------------------------

def test_deterministic_at_fixed_seed():
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(20, 20, 8)).astype(np.float32)

    s1 = kernel_rx(cube, n_background=100, seed=7)
    s2 = kernel_rx(cube, n_background=100, seed=7)

    np.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-12)


def test_different_seeds_generally_give_different_backgrounds_and_scores():
    """Not a correctness requirement by itself, but confirms the seed is
    actually wired into the subsample draw rather than ignored."""
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(24, 24, 6)).astype(np.float32)

    s1 = kernel_rx(cube, n_background=50, seed=1)
    s2 = kernel_rx(cube, n_background=50, seed=2)

    assert not np.allclose(s1, s2)


# --- gamma=None median heuristic --------------------------------------------

def test_median_heuristic_gamma_is_positive_finite():
    rng = np.random.default_rng(2)
    bg = rng.normal(size=(200, 10))
    gamma = _median_heuristic_gamma(bg)
    assert np.isfinite(gamma)
    assert gamma > 0


def test_gamma_none_runs_and_differs_from_a_fixed_gamma():
    """gamma=None must be resolved to a computed value INSIDE kernel_rx, not
    passed through as None into the RBF kernel (which would crash the
    `exp(-gamma * ...)` call, so this also doubles as a not-None guard)."""
    rng = np.random.default_rng(3)
    cube = rng.normal(size=(16, 16, 5)).astype(np.float32)

    scores_auto = kernel_rx(cube, gamma=None, n_background=100, seed=0)
    scores_fixed = kernel_rx(cube, gamma=1.0, n_background=100, seed=0)

    assert np.all(np.isfinite(scores_auto))
    assert not np.allclose(scores_auto, scores_fixed)


# --- shape / dtype contract --------------------------------------------------

def test_shape_and_dtype_contract():
    rng = np.random.default_rng(4)
    cube = rng.normal(size=(12, 15, 7)).astype(np.float32)
    scores = kernel_rx(cube, n_background=50)
    assert scores.shape == (12, 15)
    assert scores.dtype == np.float32


# --- NaN in -> NaN out, no cross-pixel poisoning ----------------------------

def test_nan_in_nan_out_without_poisoning():
    rng = np.random.default_rng(5)
    cube = rng.normal(size=(10, 10, 6)).astype(np.float32)
    cube[4, 4, :] = np.nan

    scores = kernel_rx(cube, n_background=60)

    assert np.isnan(scores[4, 4])
    flat = scores.ravel()
    bad_idx = 4 * 10 + 4
    assert not np.any(np.isnan(np.delete(flat, bad_idx)))


def test_nan_pixel_excluded_from_background_sample():
    """A NaN pixel must never be drawn into the background subsample -- if it
    were, arithmetic over it would poison the whole Gram matrix (D15)."""
    rng = np.random.default_rng(6)
    cube = rng.normal(size=(10, 10, 6)).astype(np.float32)
    cube[0, 0, :] = np.nan

    # n_background == number of valid pixels forces the sampler to draw
    # every valid pixel; if the NaN pixel leaked in, the whole result NaNs.
    scores = kernel_rx(cube, n_background=99, seed=0)
    assert np.isnan(scores[0, 0])
    assert not np.any(np.isnan(np.delete(scores.ravel(), 0)))


# --- synthetic implanted-anomaly sanity test --------------------------------

def test_implanted_target_ranks_near_top():
    rng = np.random.default_rng(8)
    h, w, b = 30, 30, 10
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[15, 15] += 50.0   # far outside the background cluster

    scores = kernel_rx(cube, n_background=400, seed=0)
    flat = scores.ravel()
    rank = (flat > flat[15 * w + 15]).sum()
    assert rank < 0.01 * flat.size


# --- AUC on ABU-Beach-2, alongside global_rx --------------------------------

@pytest.mark.skipif(not _have_abu_beach_2, reason="ABU-Beach-2 not fetched")
def test_auc_on_abu_beach_2_vs_global_rx():
    cube, _meta = load_scene(ABU_BEACH_2, source="abu")
    raw = loadmat(ABU_BEACH_2)
    labels = (raw["map"] > 0).astype(int).ravel()

    krx_scores = kernel_rx(cube, n_background=2000, seed=0).ravel()
    grx_scores = global_rx(cube).ravel()

    krx_auc = roc_auc_score(labels, krx_scores)
    grx_auc = roc_auc_score(labels, grx_scores)

    print(f"\nABU-Beach-2 ROC-AUC: kernel_rx={krx_auc:.4f}  global_rx={grx_auc:.4f}  "
          f"(local_rx not implemented in this repo yet)")

    assert 0.0 <= krx_auc <= 1.0
    assert 0.0 <= grx_auc <= 1.0
