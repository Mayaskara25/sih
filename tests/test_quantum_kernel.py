"""PLAN.md 3E.5 -- tests for quantum/quantum_kernel.py.

Data-free by construction: quantum_kernel.py never touches data/ (it consumes
angles in [0, pi], not raw cubes), so every test here builds a synthetic
QuantumSplit / angle array directly rather than calling quantum.data.build_split
-- a fresh clone with no HAD100 fetched stays green (CONTRIBUTING.md, matching
tests/test_quantum_features.py's own data-free convention).

THE EQUIVALENCE TEST IS THE IMPORTANT ONE (module docstring, D27.0 finding 5):
gram_statevector and gram_fidelity are claimed to be the SAME mathematical
object, not two competing approximations, and that claim is exactly what would
silently break if either function's convention (which overlap, which
normalization, which double-centering if any) drifted. Kept to N=40 per the
brief -- FidelityQuantumKernel costs ~28 ms/pair there, ~23 s for the 820
pairs of a 40x40 symmetric Gram, the slowest thing any test here runs.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from quantum.data import QuantumSplit
from quantum.feature_map import build_feature_map
from quantum.quantum_kernel import (
    QuantumKernelArm,
    fit_ocsvm,
    gram_fidelity,
    gram_statevector,
    score_ocsvm,
)

N_FEATURES = 8
FM = build_feature_map(N_FEATURES, kind="zz", reps=2, entanglement="linear")


def _angles(rng: np.random.Generator, n: int, n_features: int = N_FEATURES) -> np.ndarray:
    """[n, n_features] float64 angles in [0, pi] -- the only shape/range
    contract this module's Gram functions rely on (QuantumFeatureTransformer's
    actual output convention, quantum/feature_map.py).
    """
    return rng.uniform(0.0, np.pi, size=(n, n_features))


def _synthetic_split(rng: np.random.Generator, *, n_bg: int, n_an: int,
                      n_features: int = N_FEATURES, seed: int = 0) -> QuantumSplit:
    """A QuantumSplit with only X_train/y_train populated -- QuantumKernelArm.fit
    reads nothing else. Background clustered near pi/2 (small spread) and
    anomalies drawn from the FULL [0, pi] range give the two classes visibly
    different angle-encoded states without hand-deriving quantum-kernel
    separability, which the sign-orientation test needs to see AUC > 0.5 for
    a REASON, not by chance. Measured on this exact construction (test_sign_
    orientation_auc_above_half, seed=3): correctly-signed AUC 1.0, sign
    flipped 0.0 -- full-margin separation, not a coin-flip 0.55 one BLAS
    wobble from passing with a broken sign.
    """
    bg = np.clip(rng.normal(loc=np.pi / 2, scale=0.25, size=(n_bg, n_features)), 0.0, np.pi)
    an = rng.uniform(0.0, np.pi, size=(n_an, n_features))
    X = np.concatenate([bg, an]).astype(np.float64)
    y = np.concatenate([np.zeros(n_bg, dtype=np.uint8), np.ones(n_an, dtype=np.uint8)])
    empty_x = np.zeros((0, n_features), dtype=np.float64)
    empty_y = np.zeros((0,), dtype=np.uint8)
    return QuantumSplit(
        X_train=X, y_train=y, X_val=empty_x, y_val=empty_y, X_test=empty_x, y_test=empty_y,
        scene_test=np.zeros((0,), dtype="<U8"), n_features=n_features, transformer=None, seed=seed)


# --------------------------------------------------------------------- equivalence --

def test_gram_statevector_matches_gram_fidelity():
    """The central claim of this module (D27.0 finding 5): gram_statevector and
    gram_fidelity compute the SAME Gram matrix. N=40, atol=1e-8 -- generous
    above the measured 1.13e-11 max-abs-diff, so ordinary float64/BLAS jitter
    across machines does not trip it while a real convention drift would.
    """
    rng = np.random.default_rng(0)
    X = _angles(rng, 40)
    g_sv = gram_statevector(X, feature_map=FM)
    g_fq = gram_fidelity(X, feature_map=FM)
    assert g_sv.shape == g_fq.shape == (40, 40)
    max_diff = np.max(np.abs(g_sv - g_fq))
    assert max_diff < 1e-8, f"max|delta G| = {max_diff:.3e}, expected ~1e-11"


# ------------------------------------------------------------------- Gram properties --

@pytest.mark.parametrize("backend", ["statevector", "fidelity"])
def test_gram_symmetric_diag_one_psd_ish(backend):
    """diag == 1 (each state's self-fidelity), symmetric, and PSD up to float
    slop (module docstring: |M|^2 is PSD by the Schur product theorem applied
    to the exact object; a returned Gram can have eigenvalues a hair below
    zero in finite precision). fidelity backend kept tiny (N=6, 15 pairs,
    well under a second) -- the N=40 size is reserved for the equivalence test.
    """
    rng = np.random.default_rng(1)
    n = 20 if backend == "statevector" else 6
    X = _angles(rng, n)
    gram_fn = gram_statevector if backend == "statevector" else gram_fidelity
    g = gram_fn(X, feature_map=FM)

    assert np.allclose(np.diag(g), 1.0, atol=1e-8)
    assert np.allclose(g, g.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(g)
    assert eigvals.min() > -1e-6, f"most negative eigenvalue {eigvals.min():.3e}"


@pytest.mark.parametrize("backend", ["statevector", "fidelity"])
def test_gram_rectangular_matches_symmetric_block(backend):
    """gram_*(X, Y) has shape [len(X), len(Y)] and equals the corresponding
    off-diagonal block of the symmetric Gram of concat(X, Y) -- the rectangular
    and symmetric code paths must agree on what "the Gram between two sets of
    points" means. fidelity backend kept tiny (4+3 points, 21-pair symmetric
    Gram) to stay fast.
    """
    rng = np.random.default_rng(2)
    nx, ny = (10, 7) if backend == "statevector" else (4, 3)
    X = _angles(rng, nx)
    Y = _angles(rng.spawn(1)[0], ny)
    gram_fn = gram_statevector if backend == "statevector" else gram_fidelity

    g_xy = gram_fn(X, Y, feature_map=FM)
    assert g_xy.shape == (nx, ny)

    g_full = gram_fn(np.concatenate([X, Y]), feature_map=FM)
    block = g_full[:nx, nx:]
    assert np.allclose(g_xy, block, atol=1e-8)


# ------------------------------------------------------------------- OneClassSVM --

def test_sign_orientation_auc_above_half():
    """The trap this module's docstring names: OneClassSVM.decision_function is
    POSITIVE for inliers, so an unnegated score would rank background ABOVE
    anomalies -- a mirrored AUC (< 0.5) that reads as a plausible bad result,
    not an obvious bug. Fits on background only (nu=0.1), scores the full
    (background + anomaly) synthetic split, and requires AUC > 0.5: a
    reintroduced sign flip flips this to < 0.5 and fails loudly.
    """
    rng = np.random.default_rng(3)
    split = _synthetic_split(rng, n_bg=60, n_an=25, seed=0)
    X_bg = split.X_train[split.y_train == 0]

    gram_train = gram_statevector(X_bg, feature_map=FM)
    model = fit_ocsvm(gram_train, nu=0.1)

    gram_all = gram_statevector(split.X_train, X_bg, feature_map=FM)
    scores = score_ocsvm(model, gram_all)

    auc = roc_auc_score(split.y_train, scores)
    assert auc > 0.5, f"AUC {auc:.4f} <= 0.5 -- score_ocsvm's sign negation likely broken"


def test_quantum_kernel_arm_end_to_end_sign_and_shape():
    """Same sign-orientation check through the frozen QuantumKernelArm interface
    (fit/score), plus the contract shape/finite checks and circuit_info().
    """
    rng = np.random.default_rng(4)
    split = _synthetic_split(rng, n_bg=50, n_an=20, seed=0)

    arm = QuantumKernelArm(n_features=N_FEATURES, reps=2, kind="zz", nu=0.1,
                            backend="statevector", seed=0)
    assert arm.name == "quantum_kernel"
    assert arm.supervision == "unsupervised"

    arm.fit(split)
    assert arm.fit_seconds is not None and arm.fit_seconds >= 0.0

    scores = arm.score(split.X_train)
    assert scores.shape == (split.X_train.shape[0],)
    assert np.all(np.isfinite(scores))
    assert arm.score_seconds >= 0.0
    assert arm.score_calls == 1

    scores_again = arm.score(split.X_train)
    assert arm.score_calls == 2
    assert arm.score_seconds >= 0.0            # accumulated, not overwritten -- monotonic non-decreasing
    assert np.array_equal(scores, scores_again)

    auc = roc_auc_score(split.y_train, scores)
    assert auc > 0.5, f"AUC {auc:.4f} <= 0.5 through QuantumKernelArm -- sign likely broken"

    info = arm.circuit_info()
    assert info is not None
    assert info["n_qubits"] == N_FEATURES
    assert info["depth"] > 0
    assert info["basis_gates"] == ["rz", "sx", "x", "cx"]


def test_quantum_kernel_arm_fidelity_backend_dispatches_correctly():
    """The module-level gram_fidelity/gram_statevector equivalence test does
    NOT exercise QuantumKernelArm's own backend dispatch
    (`self._gram_fn = gram_statevector if ... else gram_fidelity` in
    __post_init__) -- every other arm test here uses backend="statevector".
    Because the two Grams agree to ~1e-11, an inverted dispatch would still
    pass every score/shape/AUC assertion; the only observable difference is
    wall-clock, which is this arm's whole `backend` column (D27.4). So this
    test checks the dispatch DIRECTLY (which function object got bound) and
    cross-checks that a fidelity-backend arm's scores agree with a
    statevector-backend arm's on the same tiny split, kept small (8 background
    rows, one score call) since gram_fidelity is the O(N^2)/28-ms-per-pair path.
    """
    rng = np.random.default_rng(7)
    split = _synthetic_split(rng, n_bg=8, n_an=4, seed=0)

    arm_sv = QuantumKernelArm(n_features=N_FEATURES, nu=0.2, backend="statevector", seed=0)
    arm_fq = QuantumKernelArm(n_features=N_FEATURES, nu=0.2, backend="fidelity", seed=0)
    assert arm_sv._gram_fn is gram_statevector
    assert arm_fq._gram_fn is gram_fidelity

    arm_sv.fit(split)
    arm_fq.fit(split)
    s_sv = arm_sv.score(split.X_train)
    s_fq = arm_fq.score(split.X_train)
    assert np.allclose(s_sv, s_fq, atol=1e-6)


def test_score_before_fit_raises():
    arm = QuantumKernelArm(n_features=N_FEATURES, backend="statevector", seed=0)
    with pytest.raises(RuntimeError):
        arm.score(_angles(np.random.default_rng(5), 3))


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        QuantumKernelArm(n_features=N_FEATURES, backend="not_a_backend")


# --------------------------------------------------------------------- determinism --

def test_determinism_same_seed_identical_scores():
    """Two independently constructed QuantumKernelArm instances, same params
    and seed, fit on the same split: identical scores. Both Gram backends are
    deterministic given X (module docstring: gram_fidelity's default fidelity
    primitive evaluates exactly, not via shots; gram_statevector has no RNG at
    all), so this also indirectly guards against an accidental shot-based
    fidelity primitive being introduced later.
    """
    rng = np.random.default_rng(6)
    split = _synthetic_split(rng, n_bg=40, n_an=15, seed=0)

    arm1 = QuantumKernelArm(n_features=N_FEATURES, reps=2, kind="zz", nu=0.1,
                             backend="statevector", seed=0)
    arm2 = QuantumKernelArm(n_features=N_FEATURES, reps=2, kind="zz", nu=0.1,
                             backend="statevector", seed=0)
    arm1.fit(split)
    arm2.fit(split)

    s1 = arm1.score(split.X_train)
    s2 = arm2.score(split.X_train)
    assert np.array_equal(s1, s2)
