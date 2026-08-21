"""S3A.9 / D20 -- fuse_scores is component-adaptive: weights renormalize over
whatever components are present, the active set is recorded, and a missing
component is dropped rather than zero-filled (D20's central guard -- a
zero-filled channel is a CONSTANT after rank normalization, not "no
information", and would silently bias the fused ranking)."""
import numpy as np
import pytest

from anomaly.fusion import DEFAULT_WEIGHTS, FusionResult, fuse_scores
from anomaly.scoring import rank_normalize


def _components(rng, h=8, w=8):
    return {
        "rx": rng.uniform(0, 10, size=(h, w)).astype(np.float32),
        "ace": rng.uniform(0, 1, size=(h, w)).astype(np.float32),
        "index": rng.uniform(0, 1, size=(h, w)).astype(np.float32),
        "spatial": rng.uniform(0, 5, size=(h, w)).astype(np.float32),
    }


def test_fuse_scores_returns_fusion_result_with_all_four_on_had100_style_input():
    rng = np.random.default_rng(0)
    comp = _components(rng)
    result = fuse_scores(comp)
    assert isinstance(result, FusionResult)
    assert result.components == ("ace", "index", "rx", "spatial")
    assert result.weights == pytest.approx(DEFAULT_WEIGHTS)   # nothing dropped -> unchanged


def test_fuse_scores_output_in_unit_interval():
    rng = np.random.default_rng(1)
    comp = _components(rng)
    result = fuse_scores(comp)
    valid = ~np.isnan(result.score)
    assert np.all(result.score[valid] >= 0.0)
    assert np.all(result.score[valid] <= 1.0)


def test_fuse_scores_weights_renormalize_when_index_absent_abu_case():
    """D20: on ABU/HYDICE/Indian Pines, spectral_index_score is unavailable
    (D13.4), so `index` is simply never in `components`. The three remaining
    weights must renormalize to sum to 1.0."""
    rng = np.random.default_rng(2)
    comp = _components(rng)
    del comp["index"]

    result = fuse_scores(comp)
    assert result.components == ("ace", "rx", "spatial")
    assert sum(result.weights.values()) == pytest.approx(1.0)

    total = DEFAULT_WEIGHTS["rx"] + DEFAULT_WEIGHTS["ace"] + DEFAULT_WEIGHTS["spatial"]
    assert result.weights["rx"] == pytest.approx(DEFAULT_WEIGHTS["rx"] / total)
    assert result.weights["ace"] == pytest.approx(DEFAULT_WEIGHTS["ace"] / total)
    assert result.weights["spatial"] == pytest.approx(DEFAULT_WEIGHTS["spatial"] / total)


def test_fuse_scores_active_component_set_is_recorded_and_sorted():
    rng = np.random.default_rng(3)
    comp = _components(rng)
    del comp["ace"]
    result = fuse_scores(comp)
    assert result.components == ("index", "rx", "spatial")
    assert set(result.components) == {"index", "rx", "spatial"}
    assert "ace" not in result.weights


def test_fuse_scores_missing_component_is_not_zero_filled():
    """Construct rx/ace/spatial such that zero-filling a missing `index`
    channel at its ORIGINAL (un-renormalized) weight would give a
    detectably different fused score than correctly dropping it and
    renormalizing the remaining three weights. Assert fuse_scores matches
    the drop-and-renormalize answer, not the zero-fill one."""
    rng = np.random.default_rng(4)
    h, w = 8, 8
    comp = _components(rng, h, w)
    del comp["index"]

    result = fuse_scores(comp)

    valid = np.ones((h, w), dtype=bool)
    total = DEFAULT_WEIGHTS["rx"] + DEFAULT_WEIGHTS["ace"] + DEFAULT_WEIGHTS["spatial"]
    correct = (
        (DEFAULT_WEIGHTS["rx"] / total) * rank_normalize(comp["rx"], valid=valid)
        + (DEFAULT_WEIGHTS["ace"] / total) * rank_normalize(comp["ace"], valid=valid)
        + (DEFAULT_WEIGHTS["spatial"] / total) * rank_normalize(comp["spatial"], valid=valid)
    )
    np.testing.assert_allclose(result.score, correct, atol=1e-6)

    zero_index = np.zeros((h, w), dtype=np.float32)
    wrong_zero_fill = (
        DEFAULT_WEIGHTS["rx"] * rank_normalize(comp["rx"], valid=valid)
        + DEFAULT_WEIGHTS["ace"] * rank_normalize(comp["ace"], valid=valid)
        + DEFAULT_WEIGHTS["spatial"] * rank_normalize(comp["spatial"], valid=valid)
        + DEFAULT_WEIGHTS["index"] * rank_normalize(zero_index, valid=valid)
    )
    assert not np.allclose(result.score, wrong_zero_fill, atol=1e-3)


def test_fuse_scores_components_are_rank_normalized_before_weighting():
    rng = np.random.default_rng(5)
    h, w = 10, 10
    # Wildly different native scales -- if fuse_scores weighted raw values
    # instead of rank-normalizing first, rx (~1e6 scale) would completely
    # dominate the tiny ace/spatial values.
    rx = rng.uniform(0, 1e6, size=(h, w)).astype(np.float32)
    ace = rng.uniform(0, 1, size=(h, w)).astype(np.float32)
    spatial = rng.uniform(0, 1e-3, size=(h, w)).astype(np.float32)

    result = fuse_scores({"rx": rx, "ace": ace, "spatial": spatial})

    valid = np.ones((h, w), dtype=bool)
    total = DEFAULT_WEIGHTS["rx"] + DEFAULT_WEIGHTS["ace"] + DEFAULT_WEIGHTS["spatial"]
    expected = (
        (DEFAULT_WEIGHTS["rx"] / total) * rank_normalize(rx, valid=valid)
        + (DEFAULT_WEIGHTS["ace"] / total) * rank_normalize(ace, valid=valid)
        + (DEFAULT_WEIGHTS["spatial"] / total) * rank_normalize(spatial, valid=valid)
    )
    np.testing.assert_allclose(result.score, expected, atol=1e-6)
    # And it must NOT match a raw-value weighted sum (which would be
    # dominated by rx's ~1e6 scale and clip to ~1.0 almost everywhere).
    raw_weighted = (
        (DEFAULT_WEIGHTS["rx"] / total) * rx
        + (DEFAULT_WEIGHTS["ace"] / total) * ace
        + (DEFAULT_WEIGHTS["spatial"] / total) * spatial
    )
    assert not np.allclose(result.score, np.clip(raw_weighted, 0, 1), atol=1e-3)


def test_fuse_scores_nan_locality():
    rng = np.random.default_rng(6)
    comp = _components(rng)
    comp["rx"][2, 2] = np.nan
    result = fuse_scores(comp)
    assert np.isnan(result.score[2, 2])
    assert not np.any(np.isnan(np.delete(result.score.ravel(), 2 * 8 + 2)))


def test_fuse_scores_raises_on_empty_components():
    with pytest.raises(ValueError):
        fuse_scores({})


def test_fuse_scores_raises_when_weight_missing_for_active_component():
    rng = np.random.default_rng(7)
    comp = {"rx": rng.uniform(size=(4, 4)).astype(np.float32),
            "unknown_component": rng.uniform(size=(4, 4)).astype(np.float32)}
    with pytest.raises(ValueError):
        fuse_scores(comp)


def test_fuse_scores_raises_on_shape_mismatch():
    rng = np.random.default_rng(8)
    comp = {"rx": rng.uniform(size=(4, 4)).astype(np.float32),
            "ace": rng.uniform(size=(5, 5)).astype(np.float32)}
    with pytest.raises(ValueError):
        fuse_scores(comp)
