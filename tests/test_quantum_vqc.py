"""PLAN.md 3E.3 (see D27) -- tests for quantum/vqc_encoder.py::VQCArm.

SMOKE TEST ONLY, DELIBERATELY. D27.0 finding 6: VQC costs 10.35 s per COBYLA
objective evaluation at N=600 / 8 qubits / 32 ansatz params (3.64 s at
N=200); a real maxiter=200 fit is ~35 min and MUST NOT run here -- that is a
background job run separately. Every fit below uses `ansatz_reps=0` (8
parameters on 8 qubits, the smallest real_amplitudes ansatz at n_features=8,
since n_features itself is pinned to >=8 by build_feature_map's hard qubit-
count constraint) and a small N (<=20), so the maxfun clamp (scipy clamps
`maxiter` up to `n_params + 2` -- here 10, regardless of what is requested)
still keeps each `fit()` call to ~2-3 s.

Most tests build a synthetic QuantumSplit directly rather than calling
quantum.data.build_split -- VQCArm.fit only ever reads split.X_train /
split.y_train, so a synthetic split exercises the same code path without
touching data/, keeping this file green on a fresh clone (same discipline as
tests/test_quantum_features.py). One integration test does call build_split
with limit_scenes, skipif'd on data/ not being present, to prove the real
split machinery's angles (already in [0, pi], not [-1, 1] or similar) work
end to end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from quantum.data import QuantumSplit, build_split, score_scene_natural, test_scene_ids
from quantum.vqc_encoder import VQCArm, _anomaly_column

ROOT = Path(__file__).resolve().parents[1]
_NG_DATA = ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data" / "aviris_ng_target"
_have_had100 = _NG_DATA.exists() and any(_NG_DATA.glob("*.hdr"))

# Smallest real ansatz at the smallest legal n_features: 8 qubits, ansatz_reps=0
# -> real_amplitudes(8, reps=0).num_parameters == 8 -> COBYLA maxfun clamps to
# 10 regardless of `maxiter` (see module docstring / vqc_encoder.py's trap note).
_SMOKE_KW = dict(n_features=8, reps=1, ansatz_reps=0, maxiter=1, seed=0)


def _dummy_split(X_train: np.ndarray, y_train: np.ndarray) -> QuantumSplit:
    """A minimal QuantumSplit carrying only what VQCArm.fit reads (X_train,
    y_train) -- val/test/transformer/scene_test are never touched by fit(),
    so they are filled with cheap placeholders rather than requiring a real
    build_split() call.
    """
    n = X_train.shape[1]
    return QuantumSplit(
        X_train=X_train, y_train=y_train,
        X_val=np.zeros((0, n)), y_val=np.zeros((0,), dtype=np.uint8),
        X_test=np.zeros((0, n)), y_test=np.zeros((0,), dtype=np.uint8),
        scene_test=np.zeros((0,), dtype="<U32"),
        n_features=n, transformer=None, seed=0)


def _separable_split(seed: int = 0, n_per_class: int = 8, n_features: int = 8) -> QuantumSplit:
    """Two clusters pinned near the angle-encoding extremes (0 and pi) so a
    tiny (8-parameter, 10-evaluation) VQC fit can actually separate them --
    this is what makes test_score_direction_not_flipped a fast AND reliable
    sign check rather than a flaky one (measured: AUC 1.0 with this exact
    shape, ansatz_reps=0, maxiter=1 -> 10 real evals, ~2.5 s).
    """
    rng = np.random.default_rng(seed)
    X_bg = rng.uniform(0.0, 0.4, size=(n_per_class, n_features))
    X_an = rng.uniform(np.pi - 0.4, np.pi, size=(n_per_class, n_features))
    X = np.vstack([X_bg, X_an])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.uint8)
    return _dummy_split(X, y)


# ------------------------------------------------------------------- direction --

def test_score_direction_not_flipped():
    """The frozen interface's own required test: AUC on the split the model
    was TRAINED on must be > 0.5, not < 0.5. A sign-flipped score() would
    give a mirrored AUC (e.g. 0.05 instead of 0.95) that still looks like a
    plausible-if-bad number -- this is exactly what would slip past a test
    that only checked "AUC is a number in [0, 1]".
    """
    split = _separable_split()
    arm = VQCArm(**_SMOKE_KW)
    arm.fit(split)
    scores = arm.score(split.X_train)
    auc = roc_auc_score(split.y_train, scores)
    assert auc > 0.5, f"AUC {auc:.4f} <= 0.5 -- score() is very likely sign-flipped"
    assert auc > 0.9, f"AUC {auc:.4f} unexpectedly low for a maximally-separated split"


def test_anomaly_column_matches_manual_onehot_order():
    """Locks in the mechanism test_score_direction_not_flipped's pass depends
    on: predict_proba's column 1 really is P(class == 1) for {0, 1} labels,
    because sklearn's OneHotEncoder(categories='auto') sorts fitted labels
    ascending. Checked directly against _anomaly_column rather than only
    indirectly through an AUC number.
    """
    split = _separable_split()
    arm = VQCArm(**_SMOKE_KW)
    arm.fit(split)
    assert arm._anomaly_col == 1
    assert _anomaly_column(arm._vqc) == 1


# ------------------------------------------------------------- honest wall-clock --

def test_fit_records_seconds_and_real_eval_count():
    """D27.0 finding 6's maxfun clamp trap, as a regression test: `maxiter=1`
    with an 8-parameter ansatz must NOT report 1 (or anything <= 1) objective
    evaluation -- scipy silently clamps maxfun to n_params + 2 = 10. A caller
    trusting self.maxiter for wall-clock accounting would under-report real
    cost by 10x here.
    """
    split = _separable_split()
    arm = VQCArm(**_SMOKE_KW)
    assert arm.fit_seconds is None
    assert arm.n_objective_evals is None
    arm.fit(split)
    assert arm.fit_seconds is not None and arm.fit_seconds > 0
    n_params = arm._ansatz.num_parameters
    assert n_params == 8                       # real_amplitudes(8, reps=0)
    assert arm.n_objective_evals == n_params + 2, (
        f"expected the clamped count {n_params + 2}, got {arm.n_objective_evals} "
        "-- either scipy's clamping behaviour changed or the callback isn't "
        "counting every objective evaluation")


# ------------------------------------------------------------------------ seeding --

def test_seed_reproducible():
    """Same seed -> bit-identical predict_proba. VQC's own default
    initial_point draws from qiskit_machine_learning's global algorithm_globals
    RNG, which this class does NOT rely on (see vqc_encoder.py's SEEDING
    note) -- an explicit np.random.default_rng(seed)-drawn initial_point is
    passed instead, and this test is what would catch a regression back to
    the unseeded default.
    """
    split = _separable_split(seed=1)
    a = VQCArm(**_SMOKE_KW)
    b = VQCArm(**_SMOKE_KW)
    a.fit(split)
    b.fit(split)
    sa = a.score(split.X_train)
    sb = b.score(split.X_train)
    np.testing.assert_array_equal(sa, sb)
    assert a.n_objective_evals == b.n_objective_evals


def test_different_seed_can_change_result():
    """Not a strict requirement, but guards against seed being silently
    ignored entirely (e.g. a bug that always seeds StatevectorSampler(0)
    regardless of self.seed).
    """
    split = _separable_split(seed=2)
    a = VQCArm(n_features=8, reps=1, ansatz_reps=0, maxiter=1, seed=0)
    b = VQCArm(n_features=8, reps=1, ansatz_reps=0, maxiter=1, seed=1)
    a.fit(split)
    b.fit(split)
    sa = a.score(split.X_train)
    sb = b.score(split.X_train)
    assert not np.array_equal(sa, sb)


# --------------------------------------------------------------------- circuit_info --

def test_circuit_info_fields():
    split = _separable_split()
    arm = VQCArm(**_SMOKE_KW)
    arm.fit(split)
    info = arm.circuit_info()
    assert info is not None
    assert info["n_qubits"] == 8
    assert info["n_parameters"] == 8            # ansatz_reps=0 -> real_amplitudes(8, reps=0)
    assert info["optimizer"] == "COBYLA"
    assert info["maxiter"] == 1
    assert info["depth"] > 0
    assert info["num_qubits"] == 8
    assert isinstance(info["ops"], dict) and info["ops"]
    assert info["basis_gates"] == ["rz", "sx", "x", "cx"]
    assert info["optimization_level"] == 1


def test_circuit_info_before_fit_raises():
    arm = VQCArm(**_SMOKE_KW)
    with pytest.raises(RuntimeError):
        arm.circuit_info()


def test_score_before_fit_raises():
    arm = VQCArm(**_SMOKE_KW)
    with pytest.raises(RuntimeError):
        arm.score(np.zeros((3, 8)))


# --------------------------------------------------------------------- constraints --

def test_n_features_out_of_range_raises_on_fit():
    """n_features is directly the qubit count (build_feature_map's own hard
    8..16 constraint, PLAN.md 3E.2) -- VQCArm doesn't duplicate the check,
    it just has to propagate it rather than swallowing it.
    """
    split = _separable_split(n_features=4)
    arm = VQCArm(n_features=4, reps=1, ansatz_reps=0, maxiter=1, seed=0)
    with pytest.raises(ValueError):
        arm.fit(split)


def test_class_metadata():
    assert VQCArm.name == "vqc"
    assert VQCArm.supervision == "supervised"


# ---------------------------------------------------------- real split integration --

@pytest.mark.skipif(not _have_had100, reason="requires data/benchmark/had100/HAD100")
def test_end_to_end_with_real_split_limit_scenes():
    """One real-data smoke test, capped hard via limit_scenes so it stays a
    smoke test: build_split(limit_scenes=1) loads exactly one patch per
    flightline (~4 train scenes), producing a small angle-encoded [N, 8]
    array already scaled to [0, pi] by QuantumFeatureTransformer -- this is
    the actual object every other test's _dummy_split hand-builds, so this
    test is what would catch a real-split shape/range mismatch the synthetic
    tests can't see.
    """
    split = build_split(limit_scenes=1, n_bg_per_scene=5, max_anom_per_scene=5,
                         max_train_total=20, max_test_total=20)
    assert split.X_train.shape[0] > 0
    assert split.X_train.min() >= 0.0 and split.X_train.max() <= np.pi

    arm = VQCArm(n_features=split.n_features, reps=1, ansatz_reps=0, maxiter=1, seed=0)
    arm.fit(split)
    scores = arm.score(split.X_train)
    assert scores.shape == (split.X_train.shape[0],)
    assert np.all(np.isfinite(scores))
    assert scores.min() >= 0.0 and scores.max() <= 1.0


@pytest.mark.skipif(not _have_had100, reason="requires data/benchmark/had100/HAD100")
def test_end_to_end_through_score_scene_natural():
    """The reported numbers never come from arm.score(split.X_train) directly
    -- they come from quantum.data.score_scene_natural(arm, scene_id, split),
    which transforms raw pixels through split.transformer itself (not the
    pre-transformed X_train/X_test this file's other tests hand VQCArm) and
    returns (scores, labels, sample_weight) for a weighted roc_auc_score /
    average_precision_score call (D27.7). This is the one integration the
    frozen interface exists to guarantee, so it gets its own test rather than
    being inferred from the X_train-only tests above. max_bg_per_scene is cut
    to 15 (from score_scene_natural's default 400) purely to stay a smoke
    test -- D27.7's reweighting is already covered on synthetic data by
    quantum.data's own test suite, not re-tested here.
    """
    split = build_split(limit_scenes=1, n_bg_per_scene=5, max_anom_per_scene=5,
                         max_train_total=20, max_test_total=20)
    arm = VQCArm(n_features=split.n_features, reps=1, ansatz_reps=0, maxiter=1, seed=0)
    arm.fit(split)

    scene_id = test_scene_ids()[0]
    scores, labels, weight = score_scene_natural(arm, scene_id, split, max_bg_per_scene=15, seed=0)
    assert scores.shape == labels.shape == weight.shape
    assert scores.dtype == np.float64
    assert np.all(np.isfinite(scores))
    assert set(np.unique(labels)) <= {0, 1}
    assert np.all(weight >= 1.0)
