"""PLAN.md 3E.3 (see D27 -- this file follows the correction, not the original
spec text) -- the supervised VQC arm of the classical-vs-quantum comparison.

D27.3: VQC is the ONLY supervised arm among 3E.6's five (classical AE,
OneClassSVM-RBF, QAE, quantum kernel are all unsupervised); ranking it beside
those in one column measures supervision, not quantumness. That comparison is
`classical_vs_quantum.py`'s problem to solve (a `supervision` column + an
`SVC(rbf)` partner), not this file's -- this file only has to be honest about
what it is: `supervision = "supervised"` on the class, so the mistake can't
happen by omission.

WHY V2 PRIMITIVES, WHY FUNCTION-FORM CIRCUITS (D27.0 findings 2-3, measured
2026-08-22 against the pinned qiskit 2.5.2 / qiskit-aer 0.17.2 /
qiskit-machine-learning 0.9.1 trio):
  - `from qiskit.primitives import Sampler` (V1) raises ImportError on qiskit
    2.x. `qiskit_algorithms` is not installed at all -- COBYLA comes from
    qiskit_machine_learning.optimizers, not qiskit_algorithms.optimizers.
    Only `StatevectorSampler` (V2) is used here.
  - `RealAmplitudes` as a class is a deprecated NLocal subclass (qiskit
    2.1+); the function form `real_amplitudes()` builds the identical circuit
    with no warning. `quantum/feature_map.py::build_feature_map` already made
    this switch for the feature map side; this file makes the matching switch
    for the ansatz.

THE MAXFUN CLAMP TRAP (D27.0 finding 6) IS WHY n_objective_evals EXISTS AS A
FIELD, NOT A LOG LINE. qiskit's `COBYLA(maxiter=k)` is passed straight through
to scipy as `maxfun`, and scipy silently clamps `maxfun` up to a minimum of
`n_params + 2` regardless of what was asked for -- measured: `maxiter=1` with
a 16-parameter ansatz actually ran 18 evaluations (scipy emits a UserWarning
naming the clamped value, but a caller that doesn't capture warnings would
never see it). A caller who trusts `self.maxiter` for wall-clock accounting
would silently under-report cost by however far `maxiter` sits below
`n_params + 2`. This module counts the REAL number via the VQC `callback`
hook (invoked once per objective evaluation, confirmed by comparing
`len(evals)` against scipy's own clamped-MAXFUN warning at N=20 -- they
matched exactly), not by reading back `self.maxiter` or recomputing
`n_params + 2` by hand.

COST (D27.0 finding 6, measured at N=600 / 8 qubits / 32 ansatz params, i.e.
this class's own frozen defaults `n_features=8, ansatz_reps=3`): 10.35 s per
COBYLA objective evaluation (3.64 s at N=200), so a real `maxiter=200` fit is
~35 min. Never run that from a test -- see tests/test_quantum_vqc.py, which
uses a SMALL ansatz (ansatz_reps=0, i.e. 8 params on 8 qubits, clamping
maxfun to 10 regardless of the maxiter passed) and a SMALL N (build_split's
`limit_scenes` or a synthetic split) to stay well under a minute.

DIRECTION: predict_proba COLUMN ORDER IS DERIVED, NOT ASSUMED. VQC one-hot-
encodes labels internally via `sklearn.preprocessing.OneHotEncoder
(sparse_output=False)` (qiskit_machine_learning's NeuralNetworkClassifier),
whose default `categories='auto'` sorts the fitted unique labels ascending.
For this branch's {0, 1} uint8 labels (QuantumSplit: 1 = anomaly) that DOES
put column 1 = P(anomaly) -- verified empirically with a well-separated
synthetic split (see tests/test_quantum_vqc.py::test_score_direction_...).
But `_anomaly_column` below reads the fitted encoder's actual categories
rather than hardcoding index 1, so a future label convention change (or a
qiskit_machine_learning internals change) fails loudly with a RuntimeError
instead of silently mirroring every score -- exactly the sign-flip failure
mode the frozen interface's "Direction matters" note warns about: a flipped
score produces a mirrored AUC that still looks like a plausible bad result,
so the guard has to be structural, not a one-off eyeball check.

SEEDING: VQC's own default initial_point is NOT reproducible on its own --
`NeuralNetworkClassifier._choose_initial_point` falls back to
`qiskit_machine_learning.utils.algorithm_globals.random`, a module-level RNG
whose seed this class does not touch. This is a THIRD nondeterminism source
beyond the two the brief already names (numpy RNG, StatevectorSampler seed) --
worth checking for in `quantum_autoencoder.py` and `quantum_kernel.py` too,
if either trains a `qiskit_machine_learning` model the same way. An explicit
`initial_point`, drawn from `np.random.default_rng(seed)`, is therefore
passed on every fit -- confirmed by running the same seed twice and
comparing `predict_proba` bit-for-bit
(tests/test_quantum_vqc.py::test_seed_reproducible). Drawn from U(-pi, pi)
(rotation-angle-scaled), NOT qiskit's own unseeded default of U(0, 1) --
a deliberate choice, not a neutral one: init range affects where a
variational circuit's optimizer converges, so the real `maxiter=200`
background run inherits THIS range, not qiskit's default.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

from quantum.data import QuantumSplit
from quantum.feature_map import build_feature_map, transpiled_depth


def _anomaly_column(vqc: VQC) -> int:
    """Index into `vqc.predict_proba(X)`'s columns that is P(class == 1).

    Reads it off the fitted `OneHotEncoder`'s `categories_` rather than
    assuming column 1, and raises loudly (RuntimeError, not a silent
    fallback) if that read comes back anything other than exactly one label
    == 1 -- see module docstring's DIRECTION note for why guessing here is
    the one mistake this function exists to make impossible.
    """
    try:
        categories = vqc._target_encoder.categories_[0]  # noqa: SLF001 -- no public equivalent
    except AttributeError as exc:
        raise RuntimeError(
            "VQCArm._anomaly_column: could not read VQC's fitted "
            "_target_encoder.categories_ (qiskit_machine_learning internals may "
            "have changed) -- refusing to guess the anomaly column and risk a "
            "silent sign flip") from exc
    hits = np.flatnonzero(categories == 1)
    if hits.size != 1:
        raise RuntimeError(
            "VQCArm._anomaly_column: expected exactly one fitted label == 1 "
            f"(anomaly) in target_encoder categories, found {categories!r} -- "
            "QuantumSplit.y_train is contracted to be uint8 {0, 1}, so this "
            "means a non-conforming split was passed in")
    return int(hits[0])


class VQCArm:
    """Supervised binary anomaly/background classifier: angle-encoded feature
    map + `real_amplitudes` ansatz + COBYLA, trained by `qiskit_machine_learning`'s
    `VQC` on a `StatevectorSampler`. See module docstring for why every one
    of these choices is what it is (V1 primitives gone, RealAmplitudes class
    deprecated, maxfun clamp trap, direction, seeding).

    `n_features` is directly the qubit count (`build_feature_map`'s own hard
    constraint, 8..16 -- PLAN.md 3E.2), `reps` is the feature map's repetition
    count, `ansatz_reps` is `real_amplitudes`'s (32 parameters at the frozen
    default `n_features=8, ansatz_reps=3` -- matches D27.0 finding 6's
    measurement exactly). `kind` selects the feature map family
    ("zz" | "z" | "pauli", `quantum/feature_map.py::build_feature_map`).
    """

    name = "vqc"
    supervision = "supervised"

    def __init__(self, *, n_features: int = 8, reps: int = 2, ansatz_reps: int = 3,
                 maxiter: int = 200, seed: int = 0, kind: str = "zz") -> None:
        self.n_features = n_features
        self.reps = reps
        self.ansatz_reps = ansatz_reps
        self.maxiter = maxiter
        self.seed = seed
        self.kind = kind

        self._vqc: VQC | None = None
        self._feature_map: QuantumCircuit | None = None
        self._ansatz: QuantumCircuit | None = None
        self._anomaly_col: int | None = None

        self.fit_seconds: float | None = None
        self.n_objective_evals: int | None = None

    def fit(self, split: QuantumSplit) -> None:
        """Trains on `split.X_train` / `split.y_train` ONLY -- val/test are
        never touched here (a caller wanting a val-selected hyperparameter
        sweep does that outside this class, on top of the frozen split;
        `fit` itself is single-shot, matching every other arm's interface).

        `initial_point` is drawn from a fresh `np.random.default_rng(seed)`
        (see module docstring's SEEDING note -- VQC's own default is NOT
        reproducible). `self.n_objective_evals` is the callback-counted
        REAL number of COBYLA evaluations, which the maxfun clamp trap can
        make several times larger than `self.maxiter` for a small ansatz
        (see module docstring).
        """
        self._feature_map = build_feature_map(self.n_features, kind=self.kind, reps=self.reps)
        self._ansatz = real_amplitudes(self.n_features, reps=self.ansatz_reps)
        sampler = StatevectorSampler(seed=self.seed)

        rng = np.random.default_rng(self.seed)
        initial_point = rng.uniform(-np.pi, np.pi, size=self._ansatz.num_parameters)

        n_evals = 0

        def _count_eval(_weights: np.ndarray, _value: float) -> None:
            nonlocal n_evals
            n_evals += 1

        optimizer = COBYLA(maxiter=self.maxiter)
        vqc = VQC(feature_map=self._feature_map, ansatz=self._ansatz, optimizer=optimizer,
                  sampler=sampler, initial_point=initial_point, callback=_count_eval)

        X_train = np.asarray(split.X_train, dtype=np.float64)
        y_train = np.asarray(split.y_train)

        t0 = time.perf_counter()
        vqc.fit(X_train, y_train)
        self.fit_seconds = time.perf_counter() - t0
        self.n_objective_evals = n_evals

        self._vqc = vqc
        self._anomaly_col = _anomaly_column(vqc)

    def score(self, X: np.ndarray) -> np.ndarray:
        """[N, n_features] angle-encoded pixels -> [N] float, HIGHER = MORE
        ANOMALOUS (P(class == 1), column picked by `_anomaly_column`, never
        hardcoded -- see module docstring's DIRECTION note).

        SHOT-NOISE FLOOR: `StatevectorSampler` defaults to `default_shots=1024`
        (not overridden here -- the frozen interface has no `shots` param and
        must stay identical across arms), so every returned probability is
        quantized to k/1024 with binomial std ~sqrt(p(1-p)/1024), ~1.6% at
        p=0.5. On `score_scene_natural`'s few-hundred-pixel populations this
        is enough to produce real ties in the score ranking -- not a bug,
        a property of sampling-based inference the caller's ROC/PR-AUC
        computation should expect.
        """
        if self._vqc is None:
            raise RuntimeError("VQCArm.score called before fit()")
        proba = self._vqc.predict_proba(np.asarray(X, dtype=np.float64))
        return np.asarray(proba[:, self._anomaly_col], dtype=np.float64)

    def circuit_info(self) -> dict | None:
        """`quantum.feature_map.transpiled_depth()` of the FULL trained
        circuit (feature map composed with ansatz, on `n_features` qubits --
        the same construction `VQC.__init__` itself uses internally), plus
        `n_qubits`, `n_parameters` (the ansatz's trainable weight count --
        the number the maxfun clamp trap is keyed on, NOT the feature map's
        input-parameter count), `optimizer`, `maxiter`.
        """
        if self._vqc is None or self._feature_map is None or self._ansatz is None:
            raise RuntimeError("VQCArm.circuit_info called before fit()")
        full = QuantumCircuit(self.n_features)
        full.compose(self._feature_map, inplace=True)
        full.compose(self._ansatz, inplace=True)
        info = transpiled_depth(full)
        info.update({
            "n_qubits": self.n_features,
            "n_parameters": self._ansatz.num_parameters,
            "optimizer": "COBYLA",
            "maxiter": self.maxiter,
        })
        return info
