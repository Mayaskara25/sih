"""PLAN.md 3E.6 -- tests for quantum/classical_baselines.py and
quantum/classical_vs_quantum.py.

DATA-FREE WHERE POSSIBLE, SAME DISCIPLINE AS tests/test_quantum_kernel.py /
tests/test_quantum_vqc.py. The frozen-interface and sign-orientation checks
build a synthetic QuantumSplit directly (no HAD100 fetch needed, green on a
fresh clone). Tests that exercise `quantum.data.score_scene_natural` or
`quantum.classical_vs_quantum.main` (the D27.7 weighting regression guard,
the OCSVM reference-number reproduction, the `--smoke` end-to-end run) touch
real HAD100 data and are `skipif`'d on `data/benchmark/had100` not being
present, matching `tests/test_quantum_vqc.py`'s own convention.

WHY THE WEIGHTING TEST IS THE IMPORTANT ONE (D27.7). An unweighted PR-AUC on
`score_scene_natural`'s anomaly-enriched subsample reads roughly double the
weighted (correct, natural-prevalence) number -- "0.87 where the truth is
0.34" is D27.7's own measured example. `test_natural_prevalence_pr_auc_is_
weighted_not_unweighted` reproduces that gap on a real scene and then checks
`classical_vs_quantum._score_test_scenes` reports the WEIGHTED number, not
the unweighted one -- a regression here is exactly the kind of "a check that
passes for the wrong reason is worse than one that fails" defect D27/D28
both restate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from quantum.classical_baselines import DenseAEArm, MahalanobisArm, OCSVMArm, SVCArm
from quantum.data import QuantumSplit, build_split, score_scene_natural
from quantum.data import test_scene_ids as _test_scene_ids
from quantum.quantum_autoencoder import QuantumAutoencoderArm
from quantum.quantum_kernel import QuantumKernelArm
from quantum.vqc_encoder import VQCArm
from quantum import classical_vs_quantum as cvq

ROOT = Path(__file__).resolve().parents[1]
_NG_DATA = ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data" / "aviris_ng_target"
_have_had100 = _NG_DATA.exists() and any(_NG_DATA.glob("*.hdr"))
_needs_had100 = pytest.mark.skipif(not _have_had100, reason="requires data/benchmark/had100/HAD100")

N_FEATURES = 8


# --------------------------------------------------------------- synthetic splits --

def _dummy_split(X_train: np.ndarray, y_train: np.ndarray, *,
                  X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                  n_features: int = N_FEATURES) -> QuantumSplit:
    """Minimal QuantumSplit carrying only what every arm here actually
    reads (X_train/y_train always; X_val/y_val when a test asks for a
    val-scored arm) -- same convention as tests/test_quantum_vqc.py's
    _dummy_split / tests/test_quantum_kernel.py's _synthetic_split.
    """
    empty_x = np.zeros((0, n_features), dtype=np.float64)
    empty_y = np.zeros((0,), dtype=np.uint8)
    return QuantumSplit(
        X_train=X_train, y_train=y_train,
        X_val=X_val if X_val is not None else empty_x,
        y_val=y_val if y_val is not None else empty_y,
        X_test=empty_x, y_test=empty_y,
        scene_test=np.zeros((0,), dtype="<U32"),
        n_features=n_features, transformer=None, seed=0)


def _separable_split(seed: int = 0, n_bg: int = 40, n_an: int = 40,
                      n_features: int = N_FEATURES) -> QuantumSplit:
    """Background clustered near pi/2 (small spread), anomalies drawn from
    the full [0, pi] range -- same construction tests/test_quantum_kernel.py
    uses, verified there to give full-margin (AUC 1.0 correctly-signed / 0.0
    flipped) separation rather than a coin-flip-close one, so a sign bug is
    unambiguous rather than a 1-sigma BLAS wobble away from passing anyway.
    """
    rng = np.random.default_rng(seed)
    bg = np.clip(rng.normal(loc=np.pi / 2, scale=0.25, size=(n_bg, n_features)), 0.0, np.pi)
    an = rng.uniform(0.0, np.pi, size=(n_an, n_features))
    X = np.concatenate([bg, an]).astype(np.float64)
    y = np.concatenate([np.zeros(n_bg, dtype=np.uint8), np.ones(n_an, dtype=np.uint8)])
    perm = rng.permutation(X.shape[0])
    return _dummy_split(X[perm], y[perm], n_features=n_features)


# Every constructor here builds a FROZEN-INTERFACE arm (name, supervision,
# fit, score, circuit_info, fit_seconds) at a size cheap enough to fit
# instantly (classical) or within the maxfun-clamp smoke budget
# (VQC/QAE -- ansatz_reps=0, matching tests/test_quantum_vqc.py's own
# smallest-real-ansatz convention).
_CLASSICAL_BUILDERS = {
    "classical_ae": lambda: DenseAEArm(n_latent=4, n_epochs=5, seed=0),
    "classical_ocsvm": lambda: OCSVMArm(gamma="scale", nu=0.1, seed=0),
    "classical_svc": lambda: SVCArm(gamma="scale", C=1.0, seed=0),
    "rx_8feat": lambda: MahalanobisArm(reg=1e-6, seed=0),
}
_QUANTUM_BUILDERS = {
    "vqc": lambda: VQCArm(n_features=N_FEATURES, reps=1, ansatz_reps=0, maxiter=1, seed=0),
    "qae": lambda: QuantumAutoencoderArm(n_features=N_FEATURES, n_latent=4, reps=1,
                                          ansatz_reps=0, maxiter=1, seed=0),
    "quantum_kernel": lambda: cvq._ScaledQuantumKernelArm(kind="zz", reps=1, angle_scale=1.0, seed=0),
}
_ALL_BUILDERS = {**_CLASSICAL_BUILDERS, **_QUANTUM_BUILDERS}


# ------------------------------------------------------------------- interface --

@pytest.mark.parametrize("row_id", sorted(_ALL_BUILDERS))
def test_frozen_interface(row_id):
    """Every arm has name (non-empty str), supervision (in {supervised,
    unsupervised}), and callable fit/score/circuit_info -- and after
    fit+score, circuit_info() is None for classical arms and a dict with
    'depth'/'n_qubits' for quantum ones (D27.4: depth without a named basis,
    or a missing-vs-zero distinction, is not comparable).
    """
    arm = _ALL_BUILDERS[row_id]()
    assert isinstance(arm.name, str) and arm.name
    assert arm.supervision in ("supervised", "unsupervised")
    for meth in ("fit", "score", "circuit_info"):
        assert callable(getattr(arm, meth))

    split = _separable_split(seed=1)
    arm.fit(split)
    assert arm.fit_seconds is not None and arm.fit_seconds >= 0.0

    scores = arm.score(split.X_train)
    scores = np.asarray(scores)
    assert scores.shape == (split.X_train.shape[0],)
    assert np.all(np.isfinite(scores))

    info = arm.circuit_info()
    if row_id in _CLASSICAL_BUILDERS:
        assert info is None
    else:
        assert isinstance(info, dict)
        assert "depth" in info and "n_qubits" in info


@pytest.mark.parametrize("row_id", sorted(_ALL_BUILDERS))
def test_score_before_fit_raises(row_id):
    arm = _ALL_BUILDERS[row_id]()
    with pytest.raises(RuntimeError):
        arm.score(np.zeros((3, N_FEATURES)))


# -------------------------------------------------------------- sign orientation --

_UNSUPERVISED_ROW_IDS = sorted(k for k in _CLASSICAL_BUILDERS if k != "classical_svc") + ["quantum_kernel"]


@pytest.mark.parametrize("row_id", _UNSUPERVISED_ROW_IDS)
def test_sign_orientation_unsupervised(row_id):
    """Unsupervised arms (background-only fit): scored on the SAME separable
    split's train rows, anomalies must rank higher than background -- AUC >
    0.5. A flipped score would give a mirrored (and equally plausible-
    looking) AUC below 0.5, exactly the trap OCSVMArm/QuantumKernelArm's own
    docstrings warn about.
    """
    split = _separable_split(seed=2)
    arm = _ALL_BUILDERS[row_id]()
    arm.fit(split)
    scores = arm.score(split.X_train)
    auc = roc_auc_score(split.y_train, scores)
    assert auc > 0.5, f"{row_id}: AUC {auc:.4f} <= 0.5 -- sign likely flipped"


def test_sign_orientation_svc_supervised():
    split = _separable_split(seed=2)
    arm = SVCArm(gamma="scale", C=1.0, seed=0)
    arm.fit(split)
    scores = arm.score(split.X_train)
    auc = roc_auc_score(split.y_train, scores)
    assert auc > 0.5, f"classical_svc: AUC {auc:.4f} <= 0.5 -- sign likely flipped"


def test_sign_orientation_svc_flip_fails():
    """Negating SVCArm's score (simulating the sign-flip mistake its
    docstring warns about NOT making) must mirror the AUC below 0.5 -- proof
    the test above actually catches a flip rather than passing regardless.
    """
    split = _separable_split(seed=2)
    arm = SVCArm(gamma="scale", C=1.0, seed=0)
    arm.fit(split)
    scores = arm.score(split.X_train)
    flipped_auc = roc_auc_score(split.y_train, -scores)
    correct_auc = roc_auc_score(split.y_train, scores)
    assert flipped_auc < 0.5 < correct_auc


def test_ocsvm_decision_function_is_negated():
    """Direct check on the sign-flip fix itself: OCSVMArm.score must equal
    -decision_function, not decision_function (the trap the module docstring
    names).
    """
    split = _separable_split(seed=3)
    arm = OCSVMArm(gamma="scale", nu=0.1, seed=0)
    arm.fit(split)
    raw = arm._model.decision_function(split.X_train)
    assert np.allclose(arm.score(split.X_train), -raw)


# ------------------------------------------------------------- reference numbers --

@_needs_had100
def test_ocsvm_reproduces_reference_numbers():
    """D27's own measured reference (this class's job is to REPRODUCE it,
    not to be adjusted toward a different target if it doesn't):
    gamma="scale", nu=0.1 -> train 0.6053 / val 0.4508 / test 0.5568;
    gamma=1.0, nu=0.1 -> train 0.7573 / val 0.5613 / test 0.5797.
    """
    split = build_split(seed=0, n_features=N_FEATURES)
    expected = {
        ("scale", 0.1): (0.6053, 0.4508, 0.5568),
        (1.0, 0.1): (0.7573, 0.5613, 0.5797),
    }
    for (gamma, nu), (etr, eva, ete) in expected.items():
        arm = OCSVMArm(gamma=gamma, nu=nu, seed=0)
        arm.fit(split)
        tr = roc_auc_score(split.y_train, arm.score(split.X_train))
        va = roc_auc_score(split.y_val, arm.score(split.X_val))
        te = roc_auc_score(split.y_test, arm.score(split.X_test))
        assert tr == pytest.approx(etr, abs=1e-3), (gamma, nu, "train")
        assert va == pytest.approx(eva, abs=1e-3), (gamma, nu, "val")
        assert te == pytest.approx(ete, abs=1e-3), (gamma, nu, "test")


@_needs_had100
def test_mahalanobis_reproduces_reference_numbers():
    """D28's rewrite: Mahalanobis/RX on the 8 features, reg=1e-6 -> train
    0.8255 / val 0.7396 / test 0.6635 -- the branch's strongest measured
    baseline.
    """
    split = build_split(seed=0, n_features=N_FEATURES)
    arm = MahalanobisArm(reg=1e-6, seed=0)
    arm.fit(split)
    tr = roc_auc_score(split.y_train, arm.score(split.X_train))
    va = roc_auc_score(split.y_val, arm.score(split.X_val))
    te = roc_auc_score(split.y_test, arm.score(split.X_test))
    assert tr == pytest.approx(0.8255, abs=1e-3)
    assert va == pytest.approx(0.7396, abs=1e-3)
    assert te == pytest.approx(0.6635, abs=1e-3)


@_needs_had100
def test_kernel_angle_scale_moves_test_auc():
    """D28's rewrite, the branch's largest measured single effect: the
    angle-encoding scale multiplying zz/reps=2 input moves test AUC from
    0.378 (scale=1.0, as specified) to ~0.69 (scale=0.5, val-selected).
    Reproduces both ends, cheaply (statevector backend), as the regression
    guard for `_ScaledQuantumKernelArm`.
    """
    split = build_split(seed=0, n_features=N_FEATURES)
    spec = cvq._ScaledQuantumKernelArm(kind="zz", reps=2, angle_scale=1.0, seed=0)
    spec.fit(split)
    te_spec = roc_auc_score(split.y_test, spec.score(split.X_test))
    assert te_spec == pytest.approx(0.3780, abs=1e-2)

    scaled = cvq._ScaledQuantumKernelArm(kind="zz", reps=2, angle_scale=0.5, seed=0)
    scaled.fit(split)
    te_scaled = roc_auc_score(split.y_test, scaled.score(split.X_test))
    assert te_scaled == pytest.approx(0.6926, abs=1e-2)
    assert te_scaled - te_spec > 0.2


# ------------------------------------------------------------------- weighting --

@_needs_had100
def test_natural_prevalence_pr_auc_is_weighted_not_unweighted():
    """D27.7's own regression guard. On a real test scene, the WEIGHTED
    (natural-prevalence) PR-AUC and the UNWEIGHTED one must differ
    materially -- and `classical_vs_quantum._score_test_scenes` must report
    the weighted one.
    """
    split = build_split(seed=0, n_features=N_FEATURES, limit_scenes=2)
    scene_id = _test_scene_ids()[0]
    arm = MahalanobisArm(reg=1e-6, seed=0)
    arm.fit(split)

    scores, labels, weight = score_scene_natural(arm, scene_id, split, max_bg_per_scene=400, seed=0)
    assert labels.sum() > 0, "fixture scene has no anomaly pixels -- pick another"

    weighted_pr = average_precision_score(labels, scores, sample_weight=weight)
    unweighted_pr = average_precision_score(labels, scores)
    assert abs(weighted_pr - unweighted_pr) > 0.05, (
        f"weighted {weighted_pr:.4f} vs unweighted {unweighted_pr:.4f} -- expected a "
        "material gap (D27.7); if this shrinks, the reweighting may have stopped doing "
        "anything")

    rows = cvq._score_test_scenes(arm, "rx_8feat", arm.name, arm.supervision, [scene_id],
                                   split, seed=0, max_bg_per_scene=400, threshold=None)
    assert len(rows) == 1 and rows[0].status == "ok"
    assert rows[0].pr_auc == pytest.approx(weighted_pr, abs=1e-9)
    assert rows[0].pr_auc != pytest.approx(unweighted_pr, abs=1e-3)


# ---------------------------------------------------------------- row validation --

@pytest.mark.parametrize("bad", [None, "", "quantum", "SUPERVISED", 0])
def test_row_missing_or_empty_supervision_fails(bad):
    with pytest.raises(ValueError):
        cvq._validate_supervision(bad, "some_row")


def test_row_valid_supervision_passes():
    cvq._validate_supervision("supervised", "some_row")
    cvq._validate_supervision("unsupervised", "some_row")


# -------------------------------------------------------------------- val-only sweep --

def test_select_on_val_never_touches_test():
    """`_select_on_val` must select using only X_val/y_val -- constructed so
    that val and "test-shaped" data disagree, and checking the returned
    sweep table's val_auc entries were computed from `split.X_val`, not
    leaked from elsewhere (the function signature doesn't even accept a test
    set, but this pins the contract structurally).
    """
    split = _separable_split(seed=4, n_bg=30, n_an=30)
    # give it a real (non-empty) val split too
    val_split = _dummy_split(split.X_train, split.y_train, X_val=split.X_train[:10],
                              y_val=split.y_train[:10])
    candidates = [dict(reg=r) for r in (1e-6, 1e-1, 10.0)]
    best_arm, best_params, table = cvq._select_on_val(
        cvq._build_rx, candidates, val_split, seed=0)
    assert len(table) == len(candidates)
    assert all("val_auc" in row and "params" in row for row in table)
    assert best_params in candidates


def test_select_on_val_degenerates_to_single_fit_when_one_candidate():
    """A single-candidate list (VQC/QAE/'quantum_kernel_spec', 'not swept')
    reuses the exact same code path -- this pins that it still fits and
    returns, val_auc possibly NaN if the val split is degenerate, without
    raising.
    """
    split = _separable_split(seed=5)
    arm, params, table = cvq._select_on_val(cvq._build_rx, [dict(reg=1e-6)], split, seed=0)
    assert len(table) == 1
    assert params == dict(reg=1e-6)


# --------------------------------------------------------------------- pooling --

def test_pool_keeps_fully_failed_arm():
    """D27.7 / run_benchmark.py's own rule extended to the pooled level
    (module docstring's explanation for why _pool is not an import of
    scripts/run_benchmark.py::pool): an arm with ZERO ok rows must still
    appear in the pooled table, with NaN metrics, not vanish.
    """
    import pandas as pd
    df = pd.DataFrame([
        dict(row_id="ok_arm", arm="ok_arm", supervision="unsupervised", scene="s1",
             n_px_scored=10, n_anom=2, roc_auc=0.7, pr_auc=0.3, precision=0.5, recall=0.5,
             f1=0.5, status="ok", note=""),
        dict(row_id="dead_arm", arm="dead_arm", supervision="unsupervised", scene="(all)",
             n_px_scored=0, n_anom=0, roc_auc=None, pr_auc=None, precision=None, recall=None,
             f1=None, status="FAILED:RuntimeError", note="boom"),
    ])
    pooled = cvq._pool(df, ["ok_arm", "dead_arm"])
    assert set(pooled.row_id) == {"ok_arm", "dead_arm"}
    dead = pooled[pooled.row_id == "dead_arm"].iloc[0]
    assert dead.n_scenes == 0 and dead.n_failed == 1
    assert np.isnan(dead.roc_auc_MACRO) and np.isnan(dead.roc_auc_micro)


# --------------------------------------------------------------------------- smoke --

@_needs_had100
def test_smoke_runs_end_to_end_and_writes_parseable_csvs(tmp_path):
    """`--smoke` must run all eight rows to completion, well under a minute,
    and write results.csv / results_pooled.csv / run_manifest.json that all
    parse -- and every results.csv row must carry a non-empty `supervision`.
    """
    import time
    import pandas as pd

    t0 = time.perf_counter()
    rc = cvq.main(["--smoke", "--out", str(tmp_path), "--seed", "0"])
    elapsed = time.perf_counter() - t0
    assert rc == 0
    assert elapsed < 60, f"--smoke took {elapsed:.1f}s, expected well under 60s"

    results = pd.read_csv(tmp_path / "results.csv")
    pooled = pd.read_csv(tmp_path / "results_pooled.csv")
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())

    expected_row_ids = {"classical_ae", "classical_ocsvm", "classical_svc", "rx_8feat",
                         "vqc", "qae", "quantum_kernel_spec", "quantum_kernel_valscale"}
    assert set(results.row_id.unique()) == expected_row_ids
    assert set(pooled.row_id.unique()) == expected_row_ids
    assert results.supervision.isin(["supervised", "unsupervised"]).all()
    assert "swept" in pooled.columns and "wall_clock_kind" in pooled.columns
    assert manifest["smoke"] is True
    assert set(manifest["arms"].keys()) == expected_row_ids
    assert "quantum_kernel_spec" in manifest["kernel_concentration"]
    assert "quantum_kernel_valscale" in manifest["kernel_concentration"]
    for label, diag in manifest["kernel_concentration"].items():
        assert "mean_offdiag" in diag and "p99_offdiag" in diag
