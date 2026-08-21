"""§2.3 -- global_rx correctness gate, not an accuracy gate."""
import numpy as np
from scipy.spatial.distance import mahalanobis

from anomaly.rx import global_rx


def test_matches_reference_mahalanobis():
    rng = np.random.default_rng(0)
    h, w, b = 6, 6, 4
    cube = rng.normal(size=(h, w, b)).astype(np.float32)

    scores = global_rx(cube, reg=1e-6)

    flat = cube.reshape(-1, b).astype(np.float64)
    mu = flat.mean(axis=0)
    sigma = np.cov(flat.T, bias=True) + 1e-6 * np.eye(b)
    vi = np.linalg.inv(sigma)
    ref = np.array([mahalanobis(x, mu, vi) ** 2 for x in flat]).reshape(h, w)

    np.testing.assert_allclose(scores, ref, rtol=1e-6, atol=1e-6)


def test_implanted_target_ranks_in_top_point_one_percent():
    rng = np.random.default_rng(1)
    h, w, b = 32, 32, 10
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[16, 16] += 50.0   # a=1.0 implant: far outside the background cluster

    scores = global_rx(cube)
    flat_scores = scores.ravel()
    rank = (flat_scores > flat_scores[16 * w + 16]).sum()
    assert rank < 0.001 * flat_scores.size


def test_nan_in_nan_out():
    rng = np.random.default_rng(2)
    cube = rng.normal(size=(8, 8, 5)).astype(np.float32)
    cube[3, 3, :] = np.nan
    scores = global_rx(cube)
    assert np.isnan(scores[3, 3])
    assert not np.any(np.isnan(np.delete(scores.ravel(), 3 * 8 + 3)))


def test_runs_fast_on_indian_pines_scale():
    import time

    rng = np.random.default_rng(3)
    cube = rng.normal(size=(145, 145, 200)).astype(np.float32)
    t0 = time.perf_counter()
    global_rx(cube)
    assert time.perf_counter() - t0 < 5.0
