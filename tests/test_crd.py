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
    """np.linalg.norm can't return a negative number, so this alone can't
    fail from an arithmetic mistake -- kept as a cheap contract guard, not
    load-bearing by itself. The load-bearing companion is the interpolation
    check below: a residual that is non-negative but numerically collapsed
    to ~0 everywhere would still pass this and still be broken."""
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(14, 14, 5)).astype(np.float32)
    cube[3, 3, :] = np.nan   # also exercise the NaN path in the same call
    scores = crd(cube, **_FAST_KW)
    finite = scores[~np.isnan(scores)]
    assert finite.size > 0
    assert np.all(finite >= 0.0)


def test_default_lam_does_not_interpolate_at_native_radiance_scale():
    """D22.2's silent-failure mode, reproduced synthetically: if `lam` is
    negligible relative to the Gram's scale, X^T X alone (near-OLS) fits y
    almost exactly for every pixel and the residual collapses uniformly
    towards 0 regardless of whether the pixel is anomalous -- the score
    stops discriminating anything. Scale the cube up to ABU's native
    radiance order (D13.2: max ~1e4-2e4) rather than unit-scale, because
    D22's own lesson is that unit-scale synthetic fixtures hide exactly
    this bug. K = outer**2 - inner**2 = 16 here with a small 5x5 window and
    B=6 bands (K > B, the more permissive-to-interpolate regime), so a
    residual collapse would show up clearly if the default were unsafe at
    this K/B ratio."""
    rng = np.random.default_rng(9)
    cube = (rng.normal(loc=5000.0, scale=2000.0, size=(16, 16, 6))
            .astype(np.float32))
    scores = crd(cube, outer=5, inner=3, lam=1e-2)
    y_norms = np.linalg.norm(cube.reshape(-1, 6).astype(np.float64), axis=-1)
    ratio = scores.ravel() / y_norms
    # Real (non-collapsed) discrimination: residual should be a real
    # fraction of ||y||, not uniformly pinned near machine-epsilon.
    assert np.median(ratio) > 1e-3, (
        f"median score/||y|| = {np.median(ratio):.2e} -- looks collapsed "
        "(interpolating) at native scale, D22.2's failure mode")


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

    # Not a strict monotonic chain (seed-fragile with only 3 points on
    # unit-normal data) -- the two properties that actually distinguish
    # "working" from kernel_rx's D22.2 pinned-rank failure: the biggest
    # spike must clearly outrank the smallest, and it must land near the
    # very top rather than merely somewhere unremarkable.
    assert ranks[2] < ranks[0], f"50-sigma spike did not outrank 1-sigma spike: {ranks}"
    assert ranks[2] < 0.02 * (h * w), f"50-sigma spike not near top: rank {ranks[2]}"
    assert len(set(ranks)) > 1, f"rank pinned at every magnitude: {ranks}"


@pytest.mark.skipif(not _have_abu, reason="ABU benchmark not fetched")
def test_implanted_target_rank_responds_to_magnitude_on_real_abu_crop():
    """The synthetic version above uses unit-scale data, which is exactly
    the regime D22 warns hides scale-blindness bugs ("the fixture's scale
    hid the bug the fixture existed to find"). Repeat the same rank-vs-
    magnitude check on a small crop of real, native-radiance ABU data
    (abu-beach-4, B=102, the most K>B/over-complete scene in the set --
    see this branch's report) at the shipped default lam=1e-2, so this
    suite can actually catch a D22.2-style silent failure rather than only
    a synthetic stand-in for one."""
    path = ABU_DIR / "abu-beach-4.mat"
    if not path.exists():
        pytest.skip("abu-beach-4 not fetched")
    cube, _meta = load_scene(path, source="abu")
    # small crop, away from the labelled anomaly region (rows 13-98,
    # cols 0-30, per this branch's investigation) so the implant is
    # planted on genuine background.
    crop = cube[100:120, 100:120, :]
    h, w, b = crop.shape
    flat = crop.reshape(-1, b).astype(np.float64)
    std = flat.std(axis=0)
    r, c = h // 2, w // 2

    ranks = []
    for mag in (1.0, 8.0, 50.0):
        test_cube = crop.copy().astype(np.float64)
        test_cube[r, c] = crop[r, c].astype(np.float64) + mag * std
        scores = crd(test_cube.astype(np.float32), lam=1e-2).ravel()
        ranks.append(int((scores > scores[r * w + c]).sum()))

    assert ranks[2] < ranks[0], (
        f"on real ABU data, a 50-sigma implant did not outrank a 1-sigma "
        f"one: ranks={ranks} -- this is the exact D22.2 pinned-rank "
        f"signature (kernel_rx's rank stuck at 481 of 900 for every "
        f"magnitude from 1sigma to 50sigma)")


# --- AUC on a subsample of ABU scenes, macro + micro (labelled) -------------

# CRD at the default window (outer=15/inner=5, K=200) costs ~30ms/pixel even
# on an otherwise-idle machine -- a full 100x100 scene is ~5 minutes, and
# the full 13-scene table (2 of which are 150x150) is well over an hour.
# These crops are chosen (see this branch's report for the row/col ranges
# pulled from each scene's `map`) to retain EVERY positive pixel while
# cutting the pixel count ~3-6x, so the AUC number below is a faithful
# (if partial-coverage) measurement on real data rather than a synthetic
# stand-in -- not the full-frame number that belongs in a proper
# experiments/rx_vs_ae/ deliverable table. The full 13-scene, full-frame
# table (produced out-of-repo, not part of this fast suite) is in this
# branch's report.
_SUBSAMPLE_CROPS = {
    "abu-airport-1": (slice(0, 60), slice(0, 100)),    # 144/144 positives
    "abu-beach-2":   (slice(0, 70), slice(15, 65)),    # 202/202 positives
    "abu-beach-4":   (slice(0, 100), slice(0, 35)),    #  68/68 positives
}


@pytest.mark.skipif(not _have_abu, reason="ABU benchmark not fetched")
def test_auc_on_abu_subsample_macro_and_micro():
    """CRD (default outer=15/inner=5) is slow -- O(H*W) systems of size
    K=200 each -- so this test uses positive-retaining crops of 3 of the
    13 ABU scenes rather than all 13 full frames (see module-level comment
    above and this branch's report for the full 13-scene table, produced
    out-of-repo).

    §3A.10 mandates scene-macro-average as PRIMARY (each scene weighted
    equally) with pixel-micro-average only as a labelled secondary -- an
    unlabelled pooled number is banned.
    """
    aucs = []
    all_scores, all_labels = [], []
    for scene, (rsl, csl) in _SUBSAMPLE_CROPS.items():
        path = ABU_DIR / f"{scene}.mat"
        if not path.exists():
            pytest.skip(f"{scene} not fetched")
        cube, _meta = load_scene(path, source="abu")
        raw = loadmat(path)
        labels_full = (raw["map"] > 0).astype(int)

        cube_c = cube[rsl, csl, :]
        labels = labels_full[rsl, csl].ravel()

        scores = crd(cube_c).ravel()
        auc = roc_auc_score(labels, scores)
        aucs.append(auc)
        all_scores.append(scores)
        all_labels.append(labels)
        print(f"\nCRD {scene} (crop, n_pos={int(labels.sum())}): AUC={auc:.4f}")

    macro = float(np.mean(aucs))
    micro = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))
    print(f"CRD macro-AUC (primary, n={len(aucs)} scenes, crops) = {macro:.4f}")
    print(f"CRD micro-AUC (secondary, pooled pixels, crops)      = {micro:.4f}")

    assert 0.0 <= macro <= 1.0
    assert 0.0 <= micro <= 1.0
