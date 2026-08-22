"""Tests for change_detection/physics_fusion.py (PLAN.md §3C.4)."""
from __future__ import annotations

import numpy as np
import pytest

from anomaly.scoring import rank_normalize
from change_detection.physics_fusion import difference_structure, fuse_change_signals


@pytest.fixture(scope="module")
def crafted_pair():
    """Static lower half (t2 == t1), noisy genuine-change upper half."""
    rng = np.random.default_rng(7)
    t1 = rng.uniform(0.2, 1.0, size=(8, 8, 4)).astype(np.float32)
    t2 = t1.copy()
    t2[:4] += rng.normal(0.0, 0.25, size=(4, 8, 4)).astype(np.float32)
    return t1, t2


def test_structure_maps_shapes_and_dtypes(crafted_pair):
    t1, t2 = crafted_pair
    out = difference_structure(t1, t2, patch=3)
    assert set(out) == {"variance", "entropy", "coherence"}
    for key, m in out.items():
        assert m.shape == (8, 8)
        assert m.dtype == np.float32
        assert np.isfinite(m).all()
    assert (out["variance"] >= 0).all()
    assert (out["entropy"] >= 0).all()
    assert ((out["coherence"] >= 0) & (out["coherence"] <= 1)).all()


def test_change_region_has_higher_variance_than_static(crafted_pair):
    t1, t2 = crafted_pair
    out = difference_structure(t1, t2, patch=5)
    # interiors only -- border windows of one region bleed into the other
    var_static = out["variance"][6:, :]
    var_change = out["variance"][1:3, 2:6]
    assert var_change.min() > var_static.max()
    assert np.allclose(var_static, 0.0, atol=1e-12)


def test_change_region_has_higher_entropy_than_static(crafted_pair):
    t1, t2 = crafted_pair
    out = difference_structure(t1, t2, patch=5)
    ent_static = out["entropy"][6:, :]
    ent_change = out["entropy"][1:3, 2:6]
    assert ent_change.mean() > ent_static.mean()


def test_nan_propagates_positionally():
    rng = np.random.default_rng(11)
    t1 = rng.uniform(size=(5, 5, 3)).astype(np.float32)
    t2 = rng.uniform(size=(5, 5, 3)).astype(np.float32)
    t1[2, 2, 1] = np.nan
    out = difference_structure(t1, t2, patch=3)
    for m in out.values():
        assert np.isnan(m[2, 2])
        m[2, 2] = 0.0
        assert np.isfinite(m).all()   # no NaN leaked into neighbouring patches


def test_fuse_all_weight_on_sam_reproduces_rank_normalize():
    rng = np.random.default_rng(13)
    t1 = rng.uniform(0.1, 1.0, size=(6, 6, 4)).astype(np.float32)
    t2 = t1 + rng.normal(0.0, 0.1, size=(6, 6, 4)).astype(np.float32)
    sam = rng.uniform(size=(6, 6)).astype(np.float32)
    structure = difference_structure(t1, t2, patch=3)
    fused = fuse_change_signals(sam, structure, np.zeros((6, 6), np.uint8),
                                weights={"sam": 1.0, "variance": 0.0,
                                         "entropy": 0.0, "coherence": 0.0})
    ref = rank_normalize(sam)
    assert fused.dtype == np.float32
    assert np.allclose(fused, ref, atol=1e-6)


def test_fuse_default_weights_are_convex_and_weighted_sum_holds():
    rng = np.random.default_rng(17)
    h = w = 6
    t1 = rng.uniform(0.1, 1.0, size=(h, w, 4)).astype(np.float32)
    t2 = t1 + rng.normal(0.0, 0.15, size=(h, w, 4)).astype(np.float32)
    structure = difference_structure(t1, t2, patch=3)
    signals = {"sam": rng.uniform(size=(h, w)), **structure}
    normed = {k: rank_normalize(v) for k, v in signals.items()}
    weights = {"sam": 0.50, "variance": 0.20, "entropy": 0.15, "coherence": 0.15}
    fused = fuse_change_signals(signals["sam"], structure,
                                np.zeros((h, w), np.uint8))
    ref = sum(weights[k] * normed[k] for k in weights)
    assert abs(sum(weights.values()) - 1.0) < 1e-12   # defaults are convex
    assert np.allclose(fused, ref, atol=1e-6)


def test_cloud_mask_zeroes_output_but_nan_wins():
    rng = np.random.default_rng(19)
    t1 = rng.uniform(0.1, 1.0, size=(6, 6, 4)).astype(np.float32)
    t2 = t1 + rng.normal(0.0, 0.1, size=(6, 6, 4)).astype(np.float32)
    structure = difference_structure(t1, t2, patch=3)
    sam = rng.uniform(size=(6, 6)).astype(np.float32)
    sam[0, 0] = np.nan                # nodata under cloud must stay NaN

    cloud = np.zeros((6, 6), dtype=np.uint8)
    cloud[3:, :] = 1                  # cloudy bottom half (all finite above)
    fused = fuse_change_signals(sam, structure, cloud)

    assert (fused[3:] == 0.0).sum() == 18            # zeroed...
    assert np.isnan(fused[0, 0])                     # ...but NaN beats zero
    assert (fused[:3] > 0).any()                     # clear sky kept its score


def test_bad_cloud_mask_raises():
    from core.contracts import ContractViolation
    t = np.zeros((2, 2, 3), dtype=np.float32)
    structure = difference_structure(t, t, patch=3)
    with pytest.raises(ContractViolation):
        fuse_change_signals(np.zeros((2, 2)), structure,
                            np.zeros((2, 2), dtype=np.float64))


def test_bad_weights_raise():
    t = np.zeros((2, 2, 3), dtype=np.float32)
    structure = difference_structure(t, t, patch=3)
    mask = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        fuse_change_signals(np.zeros((2, 2)), structure, mask,
                            weights={"sam": 1.0, "nope": 0.0})
    with pytest.raises(ValueError):
        fuse_change_signals(np.zeros((2, 2)), structure, mask,
                            weights={"sam": -1.0, "variance": 0.2,
                                     "entropy": 0.4, "coherence": 0.4})


def test_invalid_patch_raises(crafted_pair):
    t1, t2 = crafted_pair
    with pytest.raises(ValueError):
        difference_structure(t1, t2, patch=4)     # even
    with pytest.raises(ValueError):
        difference_structure(t1, t2, bins=1)
