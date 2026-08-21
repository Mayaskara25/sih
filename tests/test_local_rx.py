"""§3A.2 -- local_rx correctness, NaN contract, and the accept criteria:
ROC-AUC on ABU-Airport-1 >= global RX on the same scene, the annulus-too-small
assertion fires at outer=7/inner=5/n_components=20, and 145x145x200 runs
under 60s single-core.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from sklearn.metrics import roc_auc_score

from anomaly.local_rx import local_rx
from anomaly.rx import global_rx
from preprocessing.harmonize import reduce_bands

ROOT = Path(__file__).resolve().parents[1]
ABU_AIRPORT_1 = ROOT / "data" / "benchmark" / "abu" / "abu-airport-1.mat"
_have_abu = ABU_AIRPORT_1.exists()


def test_assertion_fires_on_undersized_annulus():
    # outer=7, inner=5 -> annulus has 7**2 - 5**2 = 24 samples, which is not
    # > n_components*2 == 40. This is the literal accept-criterion case from
    # plan.md §3A.2, not an arbitrary undersized example.
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(20, 20, 10)).astype(np.float32)
    with pytest.raises(AssertionError):
        local_rx(cube, inner=5, outer=7, n_components=20)


def test_assertion_fires_when_outer_not_greater_than_inner():
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(20, 20, 10)).astype(np.float32)
    with pytest.raises(AssertionError):
        local_rx(cube, inner=9, outer=9, n_components=2)


def test_matches_brute_force_annulus_mahalanobis():
    # test_rx.py's convention: the first real test of a detector proves the
    # arithmetic against an independent brute-force reference, not just that
    # the output "looks" anomaly-shaped. Everything risky in local_rx --
    # window clipping, annulus = outer_box - inner_box via inclusion-
    # exclusion, the second-moment table -- is index arithmetic that a good
    # AUC would not catch (an inner/outer swap can still rank an implant
    # highest). Params keep every pixel in the 12x12 grid scoreable: annulus
    # = 7**2 - 3**2 = 40 > n_components*2 == 6 everywhere, including corners
    # (worst case clips to 16 - 4 = 12 samples).
    rng = np.random.default_rng(7)
    h, w, b = 12, 12, 6
    inner, outer, n_components, reg = 3, 7, 3, 1e-4
    cube = rng.normal(size=(h, w, b)).astype(np.float32)

    scores = local_rx(cube, inner=inner, outer=outer, n_components=n_components, reg=reg)
    assert not np.any(np.isnan(scores))

    # Same PCA basis local_rx uses internally (fit_on=None, full SVD is
    # deterministic at this size), so the reference differs from local_rx
    # ONLY in how the annulus is gathered and solved -- explicit python
    # loops and np.cov/np.linalg.solve instead of integral images.
    reduced, _ = reduce_bands(cube, n_components=n_components, method="pca", fit_on=None)
    half_outer, half_inner = outer // 2, inner // 2
    ref = np.empty((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            r0o, r1o = max(0, r - half_outer), min(h, r + half_outer + 1)
            c0o, c1o = max(0, c - half_outer), min(w, c + half_outer + 1)
            r0i, r1i = max(0, r - half_inner), min(h, r + half_inner + 1)
            c0i, c1i = max(0, c - half_inner), min(w, c + half_inner + 1)
            inner_pts = {(rr, cc) for rr in range(r0i, r1i) for cc in range(c0i, c1i)}
            annulus_pts = [(rr, cc) for rr in range(r0o, r1o) for cc in range(c0o, c1o)
                           if (rr, cc) not in inner_pts]
            x = np.stack([reduced[rr, cc] for rr, cc in annulus_pts]).astype(np.float64)
            mu = x.mean(axis=0)
            sigma = np.cov(x.T, bias=True) + reg * np.eye(n_components)
            dv = reduced[r, c].astype(np.float64) - mu
            ref[r, c] = dv @ np.linalg.solve(sigma, dv)

    np.testing.assert_allclose(scores, ref, rtol=1e-4, atol=1e-6)


def test_sample_count_guard_returns_nan_below_threshold():
    # Plan-spec'd behavior: "pixels with fewer than n_components*2 valid
    # neighbours return NaN." At defaults (outer=21, inner=5, n_components=20,
    # threshold=40) on an 8x8 grid, the center pixel's outer window clips to
    # the full 8x8 (64 samples) with a 5x5 inner guard (25) -> annulus 39,
    # just under the threshold; a corner clips to 64 outer / 9 inner -> 55,
    # comfortably over it. Both branches of the guard, exercised at the
    # actual default parameters (not a rescaled test-only config).
    rng = np.random.default_rng(9)
    cube = rng.normal(size=(8, 8, 30)).astype(np.float32)
    scores = local_rx(cube)
    assert np.isnan(scores[4, 4])
    assert not np.isnan(scores[0, 0])


def test_shape_and_dtype_contract():
    rng = np.random.default_rng(1)
    h, w, b = 24, 24, 12
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    scores = local_rx(cube, inner=3, outer=9, n_components=4)
    assert scores.shape == (h, w)
    assert scores.dtype == np.float32


def test_nan_in_nan_out_without_poisoning_neighbours():
    rng = np.random.default_rng(2)
    h, w, b = 24, 24, 10
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[12, 12, :] = np.nan

    scores = local_rx(cube, inner=3, outer=9, n_components=4)

    assert np.isnan(scores[12, 12])
    # A single nodata pixel must not poison any OTHER pixel's score, even
    # ones whose annulus overlaps it -- the module docstring's whole point
    # about masking/gathering instead of `0.0 * NaN` arithmetic.
    flat = scores.ravel()
    poisoned_idx = 12 * w + 12
    others = np.delete(flat, poisoned_idx)
    assert not np.any(np.isnan(others))


def test_synthetic_bright_anomaly_scores_highest():
    rng = np.random.default_rng(3)
    h, w, b = 32, 32, 10
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[16, 16] += 50.0   # implant: far outside the local background cluster

    scores = local_rx(cube, inner=3, outer=11, n_components=4)
    flat = scores.ravel()
    target_idx = 16 * w + 16
    assert flat[target_idx] == np.nanmax(flat)


def test_runs_fast_on_indian_pines_scale():
    rng = np.random.default_rng(4)
    cube = rng.normal(size=(145, 145, 200)).astype(np.float32)
    t0 = time.perf_counter()
    local_rx(cube)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0


@pytest.mark.skipif(not _have_abu, reason="ABU benchmark data not fetched")
def test_auc_beats_global_rx_on_abu_airport_1():
    d = sio.loadmat(ABU_AIRPORT_1)
    cube = d["data"].astype(np.float32)
    gt = (d["map"] > 0).ravel()

    local_scores = local_rx(cube)
    global_scores = global_rx(cube)

    assert not np.any(np.isnan(local_scores))
    assert not np.any(np.isnan(global_scores))
    # Mahalanobis-style scores must be non-negative. Not structurally
    # guaranteed here: Sigma is the one-pass M2/n - mu*mu^T form, differenced
    # out of integral images, and solved with np.linalg.solve (LU) rather
    # than a Cholesky factorization -- LU will happily "solve" an indefinite
    # matrix and return a negative score where Cholesky would have raised.
    # A high AUC alone would not catch this.
    assert np.all(local_scores >= 0)

    auc_local = roc_auc_score(gt, local_scores.ravel())
    auc_global = roc_auc_score(gt, global_scores.ravel())

    assert auc_local >= auc_global
