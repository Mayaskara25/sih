"""§12 test_synth.py -- mixing-model correctness at a=0 and a=1; mask exactness."""
import numpy as np
import pytest

from segmentation.synth import (
    SPECTRA_POOLS,
    implant_targets,
    load_target_spectra,
    pseudo_anomaly_patch,
)


def _background(seed=0, h=32, w=32, b=10):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=5.0, scale=1.0, size=(h, w, b)).astype(np.float32)


def _targets(k=3, b=10, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(loc=50.0, scale=2.0, size=(k, b)).astype(np.float32))


# --- load_target_spectra: named, verified gaps, not silent failures --------

def test_load_target_spectra_lib_raises_not_fetched():
    with pytest.raises(FileNotFoundError, match="fetch_speclib"):
        load_target_spectra(("lib",))


@pytest.mark.parametrize("pool", ["abu_real", "hyd_real"])
def test_load_target_spectra_real_pools_raise_suspended(pool):
    with pytest.raises(NotImplementedError, match="O9"):
        load_target_spectra((pool,))


def test_load_target_spectra_rejects_unknown_pool():
    with pytest.raises(ValueError):
        load_target_spectra(("bogus",))


def test_spectra_pools_keys_match_load_target_spectra_branches():
    assert set(SPECTRA_POOLS) == {"lib", "abu_real", "hyd_real"}


# --- implant_targets: mixing model at a=0 / a=1, mask exactness ------------

def test_implant_targets_a_equals_zero_leaves_background_unchanged():
    bg = _background()
    targets = _targets()
    cube, mask, meta = implant_targets(
        bg, targets, n_targets=3, abundance_range=(0.0, 0.0),
        size_range_px=(10, 20), seed=0)

    assert meta["n_targets_placed"] > 0
    np.testing.assert_allclose(cube, bg)   # a=0 -> m = 0*t + 1*s = s, exactly
    for t in meta["targets"]:
        assert t["abundance"] == pytest.approx(0.0)


def test_implant_targets_a_equals_one_writes_pure_target_spectrum():
    bg = _background()
    targets = _targets()
    cube, mask, meta = implant_targets(
        bg, targets, n_targets=3, abundance_range=(1.0, 1.0),
        size_range_px=(10, 20), seed=0)

    implanted = mask.astype(bool)
    assert implanted.any()
    for r, c in zip(*np.where(implanted)):
        # exactly one of the K target spectra, bit-for-bit (a=1 -> m = t)
        matches = [np.allclose(cube[r, c], t) for t in targets]
        assert any(matches)
    # untouched pixels are exactly the original background
    np.testing.assert_array_equal(cube[~implanted], bg[~implanted])


def test_implant_targets_mask_exactness_matches_modified_pixels():
    bg = _background()
    targets = _targets()
    cube, mask, meta = implant_targets(
        bg, targets, n_targets=2, abundance_range=(0.5, 0.5),
        size_range_px=(15, 25), seed=3)

    changed = ~np.all(np.isclose(cube, bg), axis=-1)
    np.testing.assert_array_equal(mask.astype(bool), changed)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}


def test_implant_targets_rejects_band_count_mismatch():
    bg = _background(b=10)
    targets = _targets(b=5)
    with pytest.raises(ValueError, match="band count"):
        implant_targets(bg, targets, n_targets=1, seed=0)


def test_implant_targets_records_provenance_when_given():
    bg = _background()
    targets = _targets(k=2)
    _cube, _mask, meta = implant_targets(
        bg, targets, n_targets=1, abundance_range=(0.8, 0.8),
        size_range_px=(5, 5), seed=0, spectrum_ids=["usgs_001", "usgs_002"])
    assert meta["targets"][0]["spectrum_id"] in ("usgs_001", "usgs_002")


# --- pseudo_anomaly_patch: zero real target spectra, mask exactness --------

@pytest.mark.parametrize("kind", ["shift", "scale", "swap", "noise", "mixed"])
def test_pseudo_anomaly_patch_mask_exactness(kind):
    bg = _background()
    cube, mask = pseudo_anomaly_patch(bg, n_regions=3, perturbation=kind, seed=0)

    changed = ~np.all(np.isclose(cube, bg), axis=-1)
    np.testing.assert_array_equal(mask.astype(bool), changed)
    assert mask.dtype == np.uint8


def test_pseudo_anomaly_patch_uses_no_real_target_spectra():
    """Structural check: the function signature has no target_spectra
    parameter at all -- the zero-prior property is enforced by the API,
    not just a convention."""
    import inspect

    params = inspect.signature(pseudo_anomaly_patch).parameters
    assert "target_spectra" not in params


def test_pseudo_anomaly_patch_swap_pool_draws_from_pool_not_self():
    bg = _background(seed=0)
    pool = np.full((1, 32, 32, 10), 999.0, dtype=np.float32)   # unmistakable donor
    cube, mask = pseudo_anomaly_patch(
        bg, n_regions=5, perturbation="swap", seed=0, swap_pool=pool)
    implanted = mask.astype(bool)
    assert implanted.any()
    assert np.any(cube[implanted] == 999.0)
