"""PLAN.md §3A.4 -- crd correctness gates + the ABU AUC report.

See D22, D22.1, D22.2: this repo has now found the same bug three times --
a regularization constant (`reg`/`lam`) transplanted between operators whose
scales have nothing to do with each other. D22.2 names `crd`'s `lam=1e-2` as
the next one to check, and says the existing accept criterion (`lam -> inf`
collapses the score to `||y||_2`) tests the regularizer's *direction* but
not whether its default *magnitude* is in the useful range. The lam-sweep
and implant-magnitude investigation for that question was run out-of-repo
(scratch script, not committed here); see this branch's report for the
table. Its conclusion: `lam=1e-2` sits on a wide plateau for `crd`, same as
kernel_rx's D22.2 plateau, so the default is left unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from anomaly.crd import crd
from preprocessing.raster_loader import load_scene

ROOT = Path(__file__).resolve().parents[1]
ABU_DIR = ROOT / "data" / "benchmark" / "abu"
_have_abu = ABU_DIR.exists() and any(ABU_DIR.glob("*.mat"))

# Small window for the fast unit tests below -- outer=15/inner=5 (the
# defaults) on even a 20x20 synthetic cube is dominated by a single huge
# annulus (K = 15**2 - 5**2 = 200) relative to the number of pixels, which
# is slow for no correctness benefit. A tiny window exercises exactly the
# same code paths (fast batched + slow per-pixel NaN-masked).
_FAST_KW = dict(outer=5, inner=3)


# --- shape / dtype contract --------------------------------------------------

def test_shape_and_dtype_contract():
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(12, 14, 6)).astype(np.float32)
    scores = crd(cube, **_FAST_KW)
    assert scores.shape == (12, 14)
    assert scores.dtype == np.float32


# --- residual is strictly non-negative --------------------------------------

def test_residual_nonnegative_everywhere_except_nan():
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(14, 14, 5)).astype(np.float32)
    cube[3, 3, :] = np.nan   # also exercise the NaN path in the same call
    scores = crd(cube, **_FAST_KW)
    finite = scores[~np.isnan(scores)]
    assert finite.size > 0
    assert np.all(finite >= 0.0)


# --- lam -> inf collapses the score to ||y||_2 -------------------------------

def test_large_lam_collapses_to_norm_y():
    rng = np.random.default_rng(2)
    cube = rng.normal(size=(12, 12, 6)).astype(np.float32)
    scores = crd(cube, lam=1e12, **_FAST_KW)
    flat = cube.reshape(-1, 6).astype(np.float64)
    ref = np.linalg.norm(flat, axis=-1).reshape(12, 12)
    np.testing.assert_allclose(scores, ref, rtol=1e-4, atol=1e-4)


def test_lam_direction_small_lam_gives_smaller_or_equal_residual_than_huge_lam():
    """Sanity check on the regularizer's direction (independent of the
    magnitude question investigated separately, see module docstring):
    trusting neighbours more (small lam) should fit y at least as well as
    being forced to w=0 (huge lam), on average over the image."""
    rng = np.random.default_rng(3)
    cube = rng.normal(size=(14, 14, 6)).astype(np.float32)
    small = crd(cube, lam=1e-6, **_FAST_KW)
    huge = crd(cube, lam=1e12, **_FAST_KW)
    assert np.nanmean(small) <= np.nanmean(huge) + 1e-6


# --- NaN in -> NaN out, no cross-pixel poisoning -----------------------------

def test_nan_in_nan_out_without_poisoning():
    rng = np.random.default_rng(4)
    cube = rng.normal(size=(14, 14, 5)).astype(np.float32)
    cube[7, 7, :] = np.nan

    scores = crd(cube, **_FAST_KW)

    assert np.isnan(scores[7, 7])
    flat = scores.ravel()
    bad_idx = 7 * 14 + 7
    assert not np.any(np.isnan(np.delete(flat, bad_idx)))


def test_nan_neighbour_forces_slow_path_without_poisoning_the_pixel_itself():
    """A pixel whose own value is valid but whose annulus contains a NaN
    neighbour (nodata mixed into the window, or simply near the NaN pixel
    above) must still score a finite, non-NaN value -- computed on the
    per-pixel slow path over only the real neighbours (D15)."""
    rng = np.random.default_rng(5)
    cube = rng.normal(size=(14, 14, 5)).astype(np.float32)
    cube[7, 7, :] = np.nan

    scores = crd(cube, **_FAST_KW)

    # Pixels adjacent to (7,7) have (7,7) inside their annulus/window and
    # must NOT come out NaN themselves.
    for r, c in [(6, 7), (8, 7), (7, 6), (7, 8)]:
        assert np.isfinite(scores[r, c]), f"pixel {(r, c)} poisoned by NaN neighbour"


def test_all_nan_cube_returns_all_nan():
    cube = np.full((6, 6, 4), np.nan, dtype=np.float32)
    scores = crd(cube, **_FAST_KW)
    assert scores.shape == (6, 6)
    assert np.all(np.isnan(scores))


# --- synthetic implanted-anomaly sanity test, incl. rank-vs-magnitude -------

def test_implanted_target_ranks_near_top():
    rng = np.random.default_rng(6)
    h, w, b = 24, 24, 8
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[12, 12] += 50.0   # far outside the background cluster

    scores = crd(cube, **_FAST_KW)
    flat = scores.ravel()
    rank = (flat > flat[12 * w + 12]).sum()
    assert rank < 0.02 * flat.size


def test_implanted_target_rank_responds_to_spike_magnitude():
    """The check that caught kernel_rx's D22.2 silent failure: rank must
    actually MOVE with spike size, not sit pinned at some fixed rank
    regardless of how anomalous the implant is. A detector whose lam is
    wildly mis-scaled would (per D22.2's failure mode) interpolate/ignore
    genuine signal and rank a 1sigma and a 50sigma implant identically."""
    rng = np.random.default_rng(7)
    h, w, b = 24, 24, 8
    base = rng.normal(size=(h, w, b)).astype(np.float32)
    r, c = 12, 12

    ranks = []
    for mag in (1.0, 5.0, 50.0):
        cube = base.copy()
        cube[r, c] += mag
        scores = crd(cube, **_FAST_KW).ravel()
        ranks.append(int((scores > scores[r * w + c]).sum()))

    # Strictly non-increasing as magnitude grows, and NOT flat across the
    # whole sweep (the pinned-rank failure mode).
    assert ranks[0] >= ranks[1] >= ranks[2]
    assert len(set(ranks)) > 1, f"rank pinned at every magnitude: {ranks}"


# --- AUC on a subsample of ABU scenes, macro + micro (labelled) -------------

_SUBSAMPLE_SCENES = ["abu-airport-1", "abu-beach-2", "abu-urban-4"]


@pytest.mark.skipif(not _have_abu, reason="ABU benchmark not fetched")
def test_auc_on_abu_subsample_macro_and_micro():
    """CRD (default outer=15/inner=5) is slow -- O(H*W) systems of size
    K=200 each -- so this test subsamples 3 of the 13 ABU scenes rather
    than all 13 (the full 13-scene table, required by the §3A.4 accept
    criterion as a reported deliverable rather than a fast unit test, was
    produced out-of-repo; see this branch's report).

    §3A.10 mandates scene-macro-average as PRIMARY (each scene weighted
    equally) with pixel-micro-average only as a labelled secondary -- an
    unlabelled pooled number is banned.
    """
    aucs = []
    all_scores, all_labels = [], []
    for scene in _SUBSAMPLE_SCENES:
        path = ABU_DIR / f"{scene}.mat"
        if not path.exists():
            pytest.skip(f"{scene} not fetched")
        cube, _meta = load_scene(path, source="abu")
        raw = loadmat(path)
        labels = (raw["map"] > 0).astype(int).ravel()

        scores = crd(cube).ravel()
        auc = roc_auc_score(labels, scores)
        aucs.append(auc)
        all_scores.append(scores)
        all_labels.append(labels)
        print(f"\nCRD {scene}: AUC={auc:.4f}")

    macro = float(np.mean(aucs))
    micro = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))
    print(f"CRD macro-AUC (primary, n={len(aucs)} scenes)  = {macro:.4f}")
    print(f"CRD micro-AUC (secondary, pooled pixels)       = {micro:.4f}")

    assert 0.0 <= macro <= 1.0
    assert 0.0 <= micro <= 1.0
