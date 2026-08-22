"""PLAN.md 3E.4 -- SWAP-test quantum autoencoder tests. CONTRIBUTING.md: "tests
for the failure, not just the success". Every test here is data-free (no
data/ dependency, no build_split call) and well under a minute: fit/score are
exercised on tiny synthetic QuantumSplit instances rather than the real
HAD100 split, on the same reasoning tests/test_quantum_features.py already
uses for its data-free classical_reduce checks -- and because a real fit is
~15 min (this module's own measured per-eval cost times maxiter=150), which
the brief explicitly says NOT to run here (the coordinator runs it separately
as a background job).

Two correctness axes, kept SEPARATE per the brief:
  1. The SWAP test itself, pinned independent of any trained model
     (test_swap_test_*): identical states -> fidelity 1, orthogonal -> 0.
  2. The autoencoder's reported DIRECTION, pinned on data
     (test_auc_direction_on_training_split): infidelity must rank anomalies
     above background (AUC > 0.5). This is the harder failure mode -- a
     flipped score (reporting fidelity instead of infidelity) gives a
     MIRRORED AUC that still looks like a plausible, if bad, result, rather
     than an obvious crash.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.metrics import roc_auc_score

from quantum.data import QuantumSplit
from quantum.quantum_autoencoder import (
    QuantumAutoencoderArm,
    _build_circuit,
    _fidelity_batch,
    _swap_test_block,
)


def _fidelity_of_block(block: QuantumCircuit, ancilla: int) -> float:
    """Run a (fully bound, state-prepped) swap-test block through the exact
    statevector and return fidelity := 1 - 2*P(ancilla=1) -- see
    quantum_autoencoder.py's module docstring for why this, not raw
    P(ancilla=0), is "fidelity" (the SWAP test's own floor is P(0)=0.5 for
    ORTHOGONAL states, not 0).
    """
    sv = Statevector.from_instruction(block)
    p0, p1 = sv.probabilities([ancilla])
    return 1.0 - 2.0 * p1


def _synthetic_split(seed: int = 0, n_bg: int = 6, n_an: int = 6, n_features: int = 8
                      ) -> QuantumSplit:
    """A tiny, deliberately well-separated QuantumSplit for the AUC-direction
    test: background rows cluster near angle pi (all dims in [pi-0.3, pi]),
    anomaly rows cluster near angle 0 (all dims in [0, 0.3]).

    THIS ASYMMETRY IS DELIBERATE, NOT ARBITRARY, and the two clusters are NOT
    interchangeable -- ``zz_feature_map``'s pairwise interaction term is
    ``(pi - x_i) * (pi - x_j)`` (the standard ZZFeatureMap formula), which is
    ~0 at x=pi and MAXIMAL at x=0. So x~pi encodes a near-product (unentangled,
    "simple") state that a shallow, lightly-trained ansatz can plausibly rotate
    close to the trash reference |0...0>, while x~0 encodes a heavily entangled
    state that the same shallow ansatz, trained ONLY on the x~pi cluster,
    should generalize to much less well. Empirically verified while building
    this test: putting BACKGROUND at x~0 and ANOMALY at x~pi (the naive
    "background is the origin" choice), at ansatz_reps=1, gives AUC well
    below 0.5 even after 150 COBYLA evaluations across 5 random seeds -- not
    a score-direction bug, just this feature map's own geometry pulling the
    "easy to compress" cluster to the opposite end from where naive intuition
    puts it, and ansatz_reps=1 not being expressive enough to overcome that
    prior. At ansatz_reps=3/maxiter=300 the naive (unflipped) placement DOES
    recover the correct direction (measured: background infidelity 0.575 <
    anomaly infidelity 0.623) -- proof that training-with-enough-expressivity
    genuinely overcomes the feature map's geometric prior, not just that the
    prior always wins -- but 300 evals at ansatz_reps=3 is a ~45s single
    test, too slow for this file. Swapping the clusters (this version) keeps
    the AUC-direction test cheap (ansatz_reps=1/maxiter=20, ~2s) while still
    exercising a genuine (if geometry-assisted) fit -- see
    test_fit_reduces_the_training_objective for the check that pins
    training itself doing something, independent of this cluster placement.
    transformer/scene_test are unused by QuantumAutoencoderArm and left as
    placeholders.
    """
    rng = np.random.default_rng(seed)
    bg = rng.uniform(np.pi - 0.3, np.pi, size=(n_bg, n_features))
    an = rng.uniform(0.0, 0.3, size=(n_an, n_features))
    X = np.concatenate([bg, an])
    y = np.concatenate([np.zeros(n_bg, dtype=np.uint8), np.ones(n_an, dtype=np.uint8)])
    return QuantumSplit(
        X_train=X, y_train=y,
        X_val=np.zeros((0, n_features)), y_val=np.zeros((0,), dtype=np.uint8),
        X_test=np.zeros((0, n_features)), y_test=np.zeros((0,), dtype=np.uint8),
        scene_test=np.zeros((0,), dtype="<U1"),
        n_features=n_features, transformer=None, seed=seed)


# ------------------------------------------------------ SWAP test itself (data-free) --

def test_swap_test_identical_states_fidelity_one():
    """trash and ref both left at |0...0> (no state prep at all) -- the SWAP
    test's own base case. Must read fidelity ~1.0, not P(ancilla=0) ~1.0
    (which would ALSO be ~1.0 here and not distinguish the *2-1 formula from
    a bug that reports raw P(0)) -- the orthogonal case below is what
    actually pins the formula; this one pins that identical states don't
    ALREADY fail.
    """
    n_trash = 4
    block, trash, ref, ancilla = _swap_test_block(n_trash)
    fidelity = _fidelity_of_block(block, ancilla)
    assert fidelity == pytest.approx(1.0, abs=1e-9)


def test_swap_test_orthogonal_states_fidelity_zero():
    """trash flipped to |1111>, ref left at |0000> -- <1111|0000> = 0, exactly
    orthogonal. Raw P(ancilla=0) for this case is 0.5 (the SWAP test's own
    floor), so this is the test that actually catches a bug that reports raw
    P(ancilla=0) as "fidelity": that bug would read 0.5 here, not ~0.
    """
    n_trash = 4
    block, trash, ref, ancilla = _swap_test_block(n_trash)
    prep = QuantumCircuit(block.num_qubits)
    for t in trash:
        prep.x(t)
    full = prep.compose(block)
    fidelity = _fidelity_of_block(full, ancilla)
    assert fidelity == pytest.approx(0.0, abs=1e-9)


def test_swap_test_partial_overlap_between_zero_and_one():
    """A single trash qubit in |+> (H|0>) against a |0> reference: overlap
    |<+|0>|^2 = 0.5, so fidelity should land at 0.5 -- neither of the two
    extremes above, catching a formula that only happens to work at 0/1.
    """
    n_trash = 1
    block, trash, ref, ancilla = _swap_test_block(n_trash)
    prep = QuantumCircuit(block.num_qubits)
    prep.h(trash[0])
    full = prep.compose(block)
    fidelity = _fidelity_of_block(full, ancilla)
    assert fidelity == pytest.approx(0.5, abs=1e-9)


# ----------------------------------------------------------- circuit construction --

def test_build_circuit_register_layout_matches_brief():
    """8 data + 4 trash-reference + 1 ancilla = 13 qubits, per the brief's
    register layout, at the frozen defaults n_features=8/n_latent=4.
    """
    qc, fmap_params, ansatz_params, ancilla = _build_circuit(
        n_features=8, n_latent=4, reps=2, ansatz_reps=3, kind="zz")
    assert qc.num_qubits == 13
    assert ancilla == 12
    assert len(fmap_params) == 8
    assert len(ansatz_params) == 32, "real_amplitudes(8, reps=3) must have 32 params (D27.0)"


def test_build_circuit_rejects_n_latent_ge_n_features():
    with pytest.raises(ValueError):
        _build_circuit(n_features=8, n_latent=8, reps=2, ansatz_reps=3, kind="zz")


# --------------------------------------------------------------- frozen interface --

def test_circuit_info_reports_full_register_and_params():
    arm = QuantumAutoencoderArm()  # all defaults: n_features=8, n_latent=4, ansatz_reps=3
    info = arm.circuit_info()
    assert info is not None
    assert info["n_qubits"] == 13
    assert info["n_latent"] == 4
    assert info["n_trash"] == 4
    assert info["n_parameters"] == 32
    assert info["optimizer"] == "COBYLA"
    assert info["maxiter"] == 150
    assert "depth" in info and info["depth"] > 0
    assert "basis_gates" in info and "optimization_level" in info


def test_score_before_fit_raises():
    arm = QuantumAutoencoderArm()
    with pytest.raises(RuntimeError):
        arm.score(np.zeros((3, 8)))


def test_fit_rejects_split_with_no_background_rows():
    arm = QuantumAutoencoderArm(ansatz_reps=1, maxiter=5)
    split = _synthetic_split(n_bg=0, n_an=4)
    with pytest.raises(ValueError):
        arm.fit(split)


def test_fit_records_actual_clamped_eval_count():
    """maxiter=1 with ansatz_reps=1 (real_amplitudes(8, reps=1) has 16
    params) must actually run >= 16+2=18 objective evaluations -- the
    n_params+2 scipy maxfun floor the brief's TRAP note describes -- and
    n_objective_evals must record THAT count, not the requested maxiter=1.
    """
    arm = QuantumAutoencoderArm(ansatz_reps=1, maxiter=1, seed=0)
    assert len(arm._ansatz_params) == 16
    split = _synthetic_split(n_bg=3, n_an=3)
    arm.fit(split)
    assert arm.n_objective_evals >= 18, (
        f"expected >= n_params+2=18 (scipy maxfun floor), got {arm.n_objective_evals}")
    assert arm.n_objective_evals != 1
    assert arm.fit_seconds > 0.0


def test_fit_then_score_shape_and_dtype():
    arm = QuantumAutoencoderArm(ansatz_reps=1, maxiter=1, seed=0)
    split = _synthetic_split(n_bg=3, n_an=3)
    arm.fit(split)
    scores = arm.score(split.X_train)
    assert scores.shape == (6,)
    assert scores.dtype == np.float64
    assert np.all(scores >= -1e-9) and np.all(scores <= 1.0 + 1e-9)


def test_auc_direction_on_training_split():
    """THE flip-catching test the brief asks for. Background clusters near
    angle pi, anomalies near angle 0 (see _synthetic_split's docstring for
    why that placement, not the naive reverse, is what makes this cheap to
    train) -- an AE trained on background-only should compress the
    background cluster it has seen much better than the anomaly cluster it
    has never seen, so infidelity (this module's score) should rank
    anomalies ABOVE background: roc_auc_score(y, score) > 0.5. A flipped
    score (fidelity reported as the anomaly score instead of infidelity)
    would give AUC < 0.5 here -- a mirrored, still-plausible-looking number,
    not a crash, which is exactly why this check exists rather than relying
    on eyeballing one AUC value.

    Verified while building this test: ansatz_reps=1 (16 params) / maxiter=20
    on 6 background + 6 anomaly rows reliably reaches AUC > 0.9 in ~2s;
    kept exactly at that scale here rather than the frozen defaults
    (ansatz_reps=3 / maxiter=150), which would take minutes.
    """
    arm = QuantumAutoencoderArm(ansatz_reps=1, maxiter=20, seed=0)
    split = _synthetic_split(seed=2, n_bg=6, n_an=6)
    arm.fit(split)
    scores = arm.score(split.X_train)
    auc = roc_auc_score(split.y_train, scores)
    assert auc > 0.5, f"AUC {auc:.4f} <= 0.5 -- score direction is likely flipped"


def test_fit_reduces_the_training_objective():
    """test_auc_direction_on_training_split alone does not prove `fit`
    trains anything -- at this test's cluster placement (background near
    angle pi, structurally easy per _synthetic_split's docstring), an
    UNTRAINED ansatz already sits at lower background infidelity than
    anomaly infidelity, so a hypothetical `fit` that returned `x0` unchanged
    would still pass the AUC check. This test closes that gap directly: it
    recomputes mean background infidelity at the SAME initial point `fit`
    itself draws (same seed, same U(-pi, pi) convention -- see module
    docstring's DETERMINISM note) BEFORE training, then compares it against
    the trained weights `fit` actually produces. `fit` must make this
    number go DOWN on the data it was trained on, independent of any
    cluster-placement geometry.
    """
    arm = QuantumAutoencoderArm(ansatz_reps=1, maxiter=20, seed=0)
    split = _synthetic_split(seed=2, n_bg=6, n_an=6)
    X_bg = split.X_train[split.y_train == 0]

    x0 = np.random.default_rng(arm.seed).uniform(-np.pi, np.pi, size=len(arm._ansatz_params))
    before = 1.0 - _fidelity_batch(arm._circuit, arm._fmap_params, arm._ansatz_params,
                                    arm._ancilla, X_bg, x0).mean()

    arm.fit(split)
    after = 1.0 - _fidelity_batch(arm._circuit, arm._fmap_params, arm._ansatz_params,
                                   arm._ancilla, X_bg, arm._weights).mean()

    assert after < before, (
        f"fit did not reduce mean background infidelity: {before:.4f} -> {after:.4f}")


def test_fidelity_batch_matches_manual_single_sample_computation():
    """_fidelity_batch's parameter-binding loop, cross-checked against
    binding both fmap and ansatz parameters directly (no batching, no
    _fidelity_batch code at all) for one row. Pins that _fidelity_batch
    assigns X's row to the FEATURE MAP parameters and `weights` to the
    ANSATZ parameters -- not the two swapped or interleaved, a mistake that
    would silently produce a different but still in-[0,1]-range number.
    """
    qc, fmap_params, ansatz_params, ancilla = _build_circuit(
        n_features=8, n_latent=4, reps=2, ansatz_reps=1, kind="zz")
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, np.pi, size=8)
    weights = rng.uniform(-np.pi, np.pi, size=len(ansatz_params))

    bind = dict(zip(ansatz_params, weights))
    bind.update(zip(fmap_params, x))
    bound = qc.assign_parameters(bind)
    sv = Statevector.from_instruction(bound)
    p0, p1 = sv.probabilities([ancilla])
    expected = float(np.clip(1.0 - 2.0 * p1, 0.0, 1.0))

    fidelities = _fidelity_batch(qc, fmap_params, ansatz_params, ancilla, x[None, :], weights)
    assert fidelities[0] == pytest.approx(expected, abs=1e-12)


def test_fit_and_score_deterministic_at_fixed_seed():
    """Same seed, twice, from scratch (two independent QuantumAutoencoderArm
    instances) -> identical scores, asserted exactly. Pins the DETERMINISM
    note in quantum_autoencoder.py's module docstring: the only randomness in
    `fit` is the initial point, drawn explicitly from
    `np.random.default_rng(seed)` and always passed as COBYLA's `x0`, and
    `score` is exact (shot-free) statevector evaluation -- so nothing here
    should depend on run-to-run entropy the way a `NeuralNetworkClassifier`
    with an implicit initial point, or a finite-shot Sampler, would.
    """
    split = _synthetic_split(seed=2, n_bg=4, n_an=4)

    arm_a = QuantumAutoencoderArm(ansatz_reps=1, maxiter=10, seed=7)
    arm_a.fit(split)
    scores_a = arm_a.score(split.X_train)

    arm_b = QuantumAutoencoderArm(ansatz_reps=1, maxiter=10, seed=7)
    arm_b.fit(split)
    scores_b = arm_b.score(split.X_train)

    assert np.array_equal(scores_a, scores_b), "same seed must reproduce identical scores exactly"
    assert np.array_equal(arm_a._weights, arm_b._weights)
