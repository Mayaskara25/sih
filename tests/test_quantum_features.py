"""PLAN.md 3E -- Stage 0 quantum-branch foundation tests. CONTRIBUTING.md:
"tests for the failure, not just the success" and `skipif` on anything needing
data/. A fresh clone must stay green.

Two tests here deliberately AVOID data/ even though they exercise the same
code path build_split() uses (classical_reduce / harmonize): a synthetic
wavelength array dense enough to satisfy harmonize's coverage_ok gate lets
"does classical_reduce refit" and "does clipping actually clip" run without
HAD100, so those failure modes are caught on every clone, not only one with
data/ fetched.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio.crs
import rasterio.transform
from sklearn.metrics import average_precision_score

from core.contracts import SceneMeta
from quantum.data import (
    FLIGHTLINE_SPLIT,
    QuantumSplit,
    _natural_prevalence_sample,
    build_split,
    flightline_of,
    stratified_share,
)
from quantum.feature_map import (
    BASIS_GATES,
    QuantumFeatureTransformer,
    build_feature_map,
    classical_reduce,
    transpiled_depth,
)
from quantum.qiskit_basics import run_bell

ROOT = Path(__file__).resolve().parents[1]
_NG_DATA = ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data" / "aviris_ng_target"
_have_had100 = _NG_DATA.exists() and any(_NG_DATA.glob("*.hdr"))

_DUMMY_CRS = rasterio.crs.CRS.from_epsg(4326)
_DUMMY_TRANSFORM = rasterio.transform.from_origin(0, 0, 1, 1)


def _synthetic_scene(seed: int = 0, h: int = 12, w: int = 12):
    """[H, W, B] float32 cube + SceneMeta with a DENSE synthetic wavelength
    array (every 4 nm from 380-2504) so harmonize()'s coverage_ok gate passes
    without touching data/ -- this is what lets the classical_reduce tests
    below run on a fresh clone.
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(380, 2505, 4, dtype=np.float64)
    cube = rng.normal(size=(h, w, wl.size)).astype(np.float32)
    meta = SceneMeta(scene_id=f"synthetic_{seed}", crs=_DUMMY_CRS, transform=_DUMMY_TRANSFORM,
                      wavelengths=wl, bad_bands=np.zeros(wl.size, dtype=bool),
                      gsd_m=1.0, source="had100", georef="real")
    return cube, meta


# --------------------------------------------------------------- 3E.1 Bell circuit --

def test_bell_within_3sigma():
    result = run_bell()
    assert result["within_3sigma"], (
        f"Bell counts {result['counts']} not within 3 sigma "
        f"({result['sigma']:.1f}) of 50/50 at {result['shots']} shots")
    # the failure this guards against: a broken CX would leave '01'/'10' with
    # substantial mass instead of ~0.
    assert set(result["counts"]) <= {"00", "11"}


# --------------------------------------------------------------- 3E.2 feature maps --

@pytest.mark.parametrize("bad_n", [0, 1, 7, 17, 32])
def test_build_feature_map_rejects_bad_n_features(bad_n):
    with pytest.raises(ValueError):
        build_feature_map(bad_n)


def test_build_feature_map_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_feature_map(8, kind="bogus")


@pytest.mark.parametrize("kind", ["zz", "z", "pauli"])
def test_build_feature_map_no_deprecation_warning(kind):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qc = build_feature_map(8, kind=kind, reps=2, entanglement="linear")
    deprecated = [w for w in caught
                  if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))]
    assert not deprecated, f"kind={kind!r} emitted: {[str(w.message) for w in deprecated]}"
    assert qc.num_qubits == 8


def test_transpiled_depth_matches_measured_zz_8_reps2_linear():
    """Locks in the measured fact this module's docstring cites: at
    optimization_level=1, zero-bound rz gates are NOT folded away, so depth
    and gate counts are a real (if arbitrary) measurement, not an artifact of
    binding parameters to zero.
    """
    qc = build_feature_map(8, kind="zz", reps=2, entanglement="linear")
    d = transpiled_depth(qc)
    assert d["depth"] == 33
    assert d["ops"] == {"rz": 46, "cx": 28, "sx": 16}
    assert d["num_qubits"] == 8
    assert d["basis_gates"] == BASIS_GATES
    assert d["optimization_level"] == 1


# --------------------------------------------------------- classical_reduce (synthetic) --

def test_classical_reduce_given_fitted_transformer_does_not_refit():
    cube0, meta0 = _synthetic_scene(seed=0)
    cube1, meta1 = _synthetic_scene(seed=1)

    _, transformer = classical_reduce(cube0, meta0, n_features=8)
    components_before = transformer.pca.components_.copy()
    lo_before, hi_before = transformer.lo.copy(), transformer.hi.copy()

    _, transformer2 = classical_reduce(cube1, meta1, n_features=8, fit_on=transformer)

    assert transformer2 is transformer, "reused transformer should be the SAME object, not a copy/refit"
    assert np.array_equal(transformer.pca.components_, components_before), "PCA was refit"
    assert np.array_equal(transformer.lo, lo_before) and np.array_equal(transformer.hi, hi_before)


def test_classical_reduce_clips_outlier_to_0_pi():
    cube0, meta0 = _synthetic_scene(seed=0)
    _, transformer = classical_reduce(cube0, meta0, n_features=8)

    # A pixel 1000x + 5000 the training scale is WAY outside [lo, hi] in every
    # reduced component -- exactly the case clipping exists for.
    outlier_cube = cube0.copy()
    outlier_cube[0, 0, :] = cube0[0, 0, :] * 1000.0 + 5000.0
    angles, _ = classical_reduce(outlier_cube, meta0, n_features=8, fit_on=transformer)

    px = angles[0, 0]
    assert np.all(px >= 0.0) and np.all(px <= np.pi), f"outlier pixel escaped [0, pi]: {px}"
    # and it should actually be AT an extreme (clipped), not coincidentally interior
    assert np.any(np.isclose(px, 0.0)) or np.any(np.isclose(px, np.pi))


def test_classical_reduce_output_bounded_in_0_pi_on_own_train_pixels():
    """Every value classical_reduce emits for the pixels it fit ON must itself
    lie in [0, pi] -- the basic angle-encoding contract, checked before any
    clipping edge case.
    """
    cube0, meta0 = _synthetic_scene(seed=2)
    angles, _ = classical_reduce(cube0, meta0, n_features=8)
    valid = ~np.isnan(angles).any(axis=-1)
    assert valid.any()
    assert np.nanmin(angles[valid]) >= 0.0
    assert np.nanmax(angles[valid]) <= np.pi


# --------------------------------------------------------- FLIGHTLINE_SPLIT (data-free) --

def test_flightline_split_no_flightline_in_two_splits():
    seen: dict[str, str] = {}
    for split_name, flightlines in FLIGHTLINE_SPLIT.items():
        for fl in flightlines:
            assert fl not in seen, (
                f"{fl!r} appears in both {seen[fl]!r} and {split_name!r} splits")
            seen[fl] = split_name


def test_flightline_split_covers_exactly_ten_flightlines():
    union = {fl for fls in FLIGHTLINE_SPLIT.values() for fl in fls}
    assert len(union) == 10, f"expected exactly 10 flightlines, got {len(union)}: {sorted(union)}"


# --------------------------------------------------------- data-dependent (skipif) --

@pytest.mark.skipif(not _have_had100, reason="HAD100 aviris_ng_target/ not fetched")
def test_flightline_split_union_matches_disk():
    """The failure this catches: a typo'd flightline id in FLIGHTLINE_SPLIT
    that a hardcoded-vs-hardcoded comparison could never find.
    """
    union = {fl for fls in FLIGHTLINE_SPLIT.values() for fl in fls}
    on_disk = {flightline_of(p.stem) for p in _NG_DATA.glob("*.hdr")}
    assert union == on_disk, f"FLIGHTLINE_SPLIT vs disk mismatch: {union ^ on_disk}"


@pytest.mark.skipif(not _have_had100, reason="HAD100 aviris_ng_target/ not fetched")
def test_flightline_of_roundtrips_on_real_filenames():
    hdrs = sorted(_NG_DATA.glob("*.hdr"))
    assert hdrs
    for hdr in hdrs[:20]:
        fl = flightline_of(hdr.stem)
        assert hdr.stem.startswith(fl + "_")


def test_flightline_of_rejects_malformed_scene_id():
    # deliberately NOT skipif'd: this is a pure string-parsing failure test,
    # needs no data/, and CONTRIBUTING.md's skipif rule is for tests that
    # NEED data/ -- gating this one would silently drop it on a fresh clone.
    with pytest.raises(ValueError):
        flightline_of("not_a_had100_scene_id")


@pytest.mark.skipif(not _have_had100, reason="HAD100 aviris_ng_target/ not fetched")
def test_build_split_values_in_0_pi():
    split = build_split(n_features=8, limit_scenes=2, max_train_total=40,
                         max_test_total=40, seed=0)
    assert isinstance(split, QuantumSplit)
    for name in ("X_train", "X_val", "X_test"):
        X = getattr(split, name)
        if X.size:
            assert X.min() >= 0.0, f"{name} has values below 0"
            assert X.max() <= np.pi, f"{name} has values above pi"

    # clipping test: a synthetic pixel deliberately outside the train range,
    # pushed through the SAME transformer build_split fit -- must still land
    # in [0, pi], not alias past it.
    raw_dim = split.transformer.pca.n_features_in_
    outlier = np.full((1, raw_dim), 1e6, dtype=np.float64)
    angle = split.transformer.transform(outlier)
    assert np.all(angle >= 0.0) and np.all(angle <= np.pi)


@pytest.mark.skipif(not _have_had100, reason="HAD100 aviris_ng_target/ not fetched")
def test_build_split_max_train_total_honoured_stratified():
    """ang20170821t183707 alone has 38 of HAD100's 94 patches -- if capping
    were proportional (or a first-N truncation) instead of stratified, it
    would dominate the pooled train budget. This asserts NO flightline's row
    count exceeds stratified_share, AND that the total never exceeds `cap`
    -- cap=42 over 4 flightlines is deliberately NOT divisible (42 % 4 == 2),
    exactly the case a naive ceil(cap/n)-per-flightline share with no
    redistribution overshoots on: 4 * ceil(42/4) = 4*11 = 44 > 42.
    """
    from quantum.data import _assemble_split, _build_scene_records

    rng = np.random.default_rng(0)
    records = _build_scene_records(rng, n_bg_per_scene=40, max_anom_per_scene=40, limit_scenes=3)
    cap = 42
    X, y, sid = _assemble_split(records, "train", cap, rng)

    assert X.shape[0] <= cap, f"total {X.shape[0]} exceeds cap {cap}"
    assert set(np.unique(y)) == {0, 1}, "capping destroyed class balance"

    share = stratified_share(cap, len(FLIGHTLINE_SPLIT["train"]))
    fls = np.array([flightline_of(s) for s in sid])
    for fl in FLIGHTLINE_SPLIT["train"]:
        count = int((fls == fl).sum())
        assert count <= share, f"{fl} has {count} rows, exceeds stratified share {share}"


# --------------------------------------------------------- natural-prevalence sampling --

def test_natural_prevalence_weighting_recovers_pr_auc():
    """The whole point of score_scene_natural's sampling: on a population with
    real (low) anomaly prevalence, scoring only a subsample and reporting
    UNWEIGHTED PR-AUC systematically inflates it; weighting by
    n_background_total / n_background_sampled recovers something close to the
    full-population number instead. Both halves of that asymmetry are
    asserted, per the correction to this module's spec.
    """
    rng = np.random.default_rng(0)
    n = 50_000
    prevalence = 0.004
    gt_flat = rng.random(n) < prevalence
    valid = np.ones(n, dtype=bool)
    assert gt_flat.sum() >= 20, "synthetic population needs enough positives to be meaningful"

    # separable-ish synthetic scores: anomalies score higher on average, plus noise
    scores_all = rng.normal(0, 1, n) + gt_flat * 2.5
    full_ap = average_precision_score(gt_flat, scores_all)

    idx, weight = _natural_prevalence_sample(gt_flat, valid, max_bg_per_scene=1500, seed=0)
    scores_sub = scores_all[idx]
    labels_sub = gt_flat[idx].astype(int)

    weighted_ap = average_precision_score(labels_sub, scores_sub, sample_weight=weight)
    unweighted_ap = average_precision_score(labels_sub, scores_sub)

    assert abs(weighted_ap - full_ap) < abs(unweighted_ap - full_ap), (
        f"weighted ({weighted_ap:.4f}) should be closer to full-population "
        f"({full_ap:.4f}) than unweighted ({unweighted_ap:.4f}) is")
    # unweighted inflates PR-AUC toward the balanced-population regime -- assert
    # the direction, not just "different", so a bug that shrinks it instead
    # still fails this test.
    assert unweighted_ap > full_ap
