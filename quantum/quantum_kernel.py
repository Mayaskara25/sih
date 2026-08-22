"""PLAN.md 3E.5 -- fidelity quantum kernel -> precomputed Gram -> OneClassSVM,
the second established QML paradigm alongside the VQC (quantum/vqc_encoder.py).
Never a dependency of anomaly/, segmentation/, preprocessing/, pipeline/
(Roadmap §1.5, §9.10) -- PC + simulator only.

THE CENTRAL FINDING OF THIS MODULE (D27.0 finding 5, D27.4): the spec's path,
`FidelityQuantumKernel`, and the exact statevector-overlap Gram it is
mathematically equal to, are NOT interchangeable in cost. Measured on this
machine, 8 qubits / reps=2 / linear entanglement:

    FidelityQuantumKernel().evaluate   28 ms/pair, O(N^2) circuit evaluations
        N=50  -> 35.3 s  (1275 pairs)
        N=100 -> 142.8 s (5050 pairs)
    |<phi(x)|phi(y)>|^2 via Statevector  O(N) simulations + ONE matrix product
        N=40  -> 0.122 s   (FidelityQuantumKernel: 22.4 s on the same X)
        N=600 -> 1.82 s (full Gram)

Both compute the SAME quantity: FidelityQuantumKernel's `ComputeUncompute`
fidelity between two feature-map states is by definition |<phi(x)|phi(y)>|^2,
and its default fidelity primitive here evaluates that EXACTLY (no shot
sampling -- see gram_fidelity's docstring), not approximately. Building the
statevector |phi(x)> once per sample and taking |Psi_X @ Psi_Y^H|^2 as a single
matrix product is that same object, not an approximation of it -- measured
max|delta G| = 1.13e-11 at N=40 (this module's equivalence test re-measures
that at atol=1e-8, generously above the observed 1e-11, so a real regression
trips it while normal LAPACK/BLAS jitter does not).

This is the branch's most interesting result and it belongs here, not just in
plan.md: a "quantum kernel" that a laptop reproduces exactly, ~180x faster, by
multiplying two matrices. At 8 qubits, AerSimulator/Statevector time is a
statement about simulator overhead, not about a QPU (D27.4) -- the O(N^2) cost
FidelityQuantumKernel pays is the cost of re-deriving, circuit by circuit, an
inner product that a single matrix multiplication already gives exactly.

So this module ships BOTH gram_fidelity (the spec path) and gram_statevector
(the fast path), with the equivalence between them as a tested, first-class
property (test_quantum_kernel.py's equivalence test), not an implementation
detail hidden behind one function. QuantumKernelArm defaults to
backend="statevector" because a 600-sample train Gram is 1.82 s that way and
~38 min the FidelityQuantumKernel way (D27.0 finding 4) -- backend="fidelity"
stays selectable for anyone who wants the spec's literal object at a size that
can afford it.

ONE-CLASS SVM SIGN CONVENTION -- THE TRAP THIS MODULE'S HOUSE STYLE FLAGS.
`sklearn.svm.OneClassSVM.decision_function` is POSITIVE for inliers (the
signed distance to the separating hyperplane, background-side positive). This
branch's score contract, like every other detector in this repo, is HIGHER =
MORE ANOMALOUS. Returning decision_function unchanged silently mirrors every
ranking: an unusually good AUC and an unusually bad one look equally
plausible from the number alone, and only a sign check catches it (D27's
'pattern, again' -- a check that passes for the wrong reason is worse than one
that fails). `score_ocsvm` therefore negates. The determinism/sign test in
test_quantum_kernel.py asserts AUC > 0.5 on the training split specifically to
catch a reintroduced sign flip, which a bare "scores are finite" test would
not.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import OneClassSVM

from quantum.data import QuantumSplit
from quantum.feature_map import build_feature_map, transpiled_depth

_BACKENDS = ("statevector", "fidelity")


# ------------------------------------------------------------------- Gram matrices --

def _statevectors(X: np.ndarray, feature_map: QuantumCircuit) -> np.ndarray:
    """One `Statevector` per row of X, bound through feature_map's own
    `.parameters` order. X's columns must already be in that order -- true for
    every caller in this module, since QuantumSplit.X_* and QuantumFeatureTransformer
    both produce angles in qubit-index order (quantum/feature_map.py). Returns
    [N, 2**n_qubits] complex128, each row L2-normalized (a Statevector always
    is), which is exactly what makes gram_statevector's single matrix product
    equal to the fidelity kernel rather than an unnormalized inner product.
    """
    return np.array([Statevector(feature_map.assign_parameters(row)).data for row in X])


def gram_fidelity(X: np.ndarray, Y: np.ndarray | None = None, *,
                   feature_map: QuantumCircuit) -> np.ndarray:
    """Spec path (PLAN.md 3E.5): `qiskit_machine_learning.kernels
    .FidelityQuantumKernel`, no explicit `fidelity=` argument (verified to
    construct on this pinned stack -- see module docstring's measured facts).
    With no `fidelity=` given, FidelityQuantumKernel builds
    `ComputeUncompute(sampler=QMLSampler())`, and QMLSampler's own default is
    `shots=None` -> analytic/exact mode (read from
    `qiskit_machine_learning.primitives.QMLSampler.__init__`, not assumed) --
    so this evaluates state overlap EXACTLY, not via shot sampling. Confirmed
    empirically too: this module's own equivalence test holds to atol=1e-8
    against gram_statevector's exact linear algebra, which shot noise at any
    reasonable shot count would not survive.

    Y=None -> symmetric train Gram (diag forced to 1 by FidelityQuantumKernel's
    `evaluate_duplicates="off_diagonal"` default). Y given -> rectangular
    [len(X), len(Y)] cross-Gram, used for scoring test/val points against the
    fitted background set.

    COST (D27.0 finding 4, re-measured, not re-derived here): 28 ms/pair,
    O(N^2) circuit evaluations -- N=50 -> 35.3 s, N=100 -> 142.8 s. A 600-sample
    train Gram is on the order of half an hour. Reserved for small N or for
    someone deliberately re-verifying gram_statevector against the spec's
    literal object; QuantumKernelArm's default backend is "statevector".
    """
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return np.asarray(kernel.evaluate(np.asarray(X), None if Y is None else np.asarray(Y)))


def gram_statevector(X: np.ndarray, Y: np.ndarray | None = None, *,
                      feature_map: QuantumCircuit) -> np.ndarray:
    """Fast path: |<phi(x)|phi(y)>|^2 formed directly, one `Statevector` per
    sample (O(N) simulations, not O(N^2)) then a SINGLE matrix product. This is
    the same mathematical object gram_fidelity computes -- not an
    approximation of it (module docstring; measured max|delta G| = 1.13e-11 at
    N=40, D27.0 finding 5) -- ~180x faster because it never re-simulates a
    circuit per pair when the O(N) set of per-sample states already determines
    every pairwise overlap.

    Y=None -> symmetric Gram via Psi_X @ Psi_X^H (diag == 1 exactly, since each
    Statevector is unit-norm by construction -- no evaluate_duplicates special
    case needed here, unlike gram_fidelity). Y given -> rectangular
    [len(X), len(Y)] cross-Gram from Psi_X @ Psi_Y^H.

    PSD-ish, not exactly PSD in finite precision: the raw overlap matrix
    M = Psi_X @ Psi_X^H is Hermitian PSD (a Gram matrix of vectors), and the
    elementwise-squared matrix |M|^2 = M ∘ conj(M) is PSD by the Schur product
    theorem (conj(M) = M^T for Hermitian M, and the Hadamard product of two PSD
    matrices is PSD) -- but that is a statement about the exact object, and
    float64 rounding can leave a returned Gram with eigenvalues a hair below
    zero. `fit_ocsvm` doesn't need exact PSD (OneClassSVM.fit tolerates it),
    so no projection is applied here.
    """
    psi_x = _statevectors(np.asarray(X), feature_map)
    psi_y = psi_x if Y is None else _statevectors(np.asarray(Y), feature_map)
    return np.abs(psi_x @ psi_y.conj().T) ** 2


# ------------------------------------------------------------------------ OneClassSVM --

def fit_ocsvm(gram_train: np.ndarray, *, nu: float = 0.1) -> OneClassSVM:
    """Fit `sklearn.svm.OneClassSVM(kernel="precomputed")` on a background-only
    Gram matrix. `nu` is OneClassSVM's usual upper bound on the training
    background fraction treated as outliers (and lower bound on the support
    vector fraction) -- unrelated to this branch's `n_qubits`/`reps`
    hyperparameters, just sklearn's own nu-SVM knob.
    """
    model = OneClassSVM(kernel="precomputed", nu=nu)
    model.fit(gram_train)
    return model


def score_ocsvm(model: OneClassSVM, gram_test_train: np.ndarray) -> np.ndarray:
    """[N_test, N_train] cross-Gram (same N_train columns/order the model was
    fit on) -> [N_test] anomaly scores, HIGHER = MORE ANOMALOUS.

    NEGATES `decision_function`, which sklearn defines POSITIVE for inliers
    (module docstring's sign-flip trap) -- this is the one line in the module
    that exists purely to keep this arm's score contract the same direction as
    every other detector in the repo.
    """
    return -np.asarray(model.decision_function(gram_test_train))


# --------------------------------------------------------------------- frozen arm --

@dataclass
class QuantumKernelArm:
    """PLAN.md 3E.5/3E.6 frozen arm wrapper. Unsupervised: `fit` sees only
    background-labelled (y==0) rows of `split.X_train`, matching every other
    unsupervised arm in the branch's supervision-matched comparison (D27.3) --
    OneClassSVM is a one-class method and would be misused by feeding it
    anomaly rows to fit against.

    backend in {"statevector", "fidelity"} selects gram_statevector /
    gram_fidelity (see their docstrings for the ~180x cost gap, D27.0 finding
    4-5) -- both compute the identical Gram, so switching backend changes wall-
    clock, not the fitted model, up to the two functions' measured 1e-11
    agreement.

    `seed` is accepted for interface symmetry with the other frozen arms (VQC,
    QAE) and threads to nothing internal: both Gram backends are deterministic
    given X (gram_fidelity's default fidelity primitive evaluates exactly, not
    via shots -- see its docstring) and OneClassSVM's QP solve is itself
    deterministic, so there is no RNG in this arm's fit/score path to seed.
    Recorded on the instance anyway so a caller logging `arm.seed` gets a real
    value rather than an absent attribute.
    """
    n_features: int = 8
    reps: int = 2
    kind: str = "zz"
    nu: float = 0.1
    backend: str = "statevector"
    seed: int = 0

    name: str = field(default="quantum_kernel", init=False)
    supervision: str = field(default="unsupervised", init=False)

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {self.backend!r}")
        self.feature_map = build_feature_map(self.n_features, kind=self.kind, reps=self.reps)
        self._gram_fn = gram_statevector if self.backend == "statevector" else gram_fidelity
        self.model: OneClassSVM | None = None
        self.X_train: np.ndarray | None = None
        self.fit_seconds: float | None = None
        self.score_seconds: float = 0.0
        self.score_calls: int = 0

    def fit(self, split: QuantumSplit) -> None:
        """BACKGROUND ROWS ONLY (y_train == 0) -- see class docstring. Stores
        the background pool as `self.X_train` because OneClassSVM's
        precomputed-kernel `decision_function` needs the SAME set of training
        columns at score time (sklearn's precomputed-kernel convention: pass
        the full train-relative Gram, not just support-vector columns; sklearn
        indexes into `support_` internally).
        """
        X_bg = np.asarray(split.X_train)[np.asarray(split.y_train) == 0]
        t0 = time.perf_counter()
        gram_train = self._gram_fn(X_bg, feature_map=self.feature_map)
        self.model = fit_ocsvm(gram_train, nu=self.nu)
        self.X_train = X_bg
        self.fit_seconds = time.perf_counter() - t0

    def score(self, X: np.ndarray) -> np.ndarray:
        """[N, n_features] -> [N] anomaly scores, HIGHER = MORE ANOMALOUS
        (score_ocsvm's sign-flip fix). Raises RuntimeError if called before
        `fit` -- a None `self.model`/`self.X_train` would otherwise fail
        inside `_gram_fn`/`decision_function` with a less legible error.

        `self.score_seconds` ACCUMULATES across calls (and `self.score_calls`
        counts them) rather than being overwritten -- D27.7's reporting path
        calls `score_scene_natural` once per test scene, up to 28 times per
        arm, and a runner reading `arm.score_seconds` after that loop wants
        the arm's total scoring wall-clock, not whichever scene happened to
        be scored last. A caller wanting one call's cost can still time it
        externally; overwriting here would silently make that the ONLY
        thing `score_seconds` could ever mean.
        """
        if self.model is None or self.X_train is None:
            raise RuntimeError("QuantumKernelArm.score called before fit")
        t0 = time.perf_counter()
        gram_test = self._gram_fn(np.asarray(X), self.X_train, feature_map=self.feature_map)
        scores = score_ocsvm(self.model, gram_test)
        self.score_seconds += time.perf_counter() - t0
        self.score_calls += 1
        return scores

    def circuit_info(self) -> dict | None:
        """`transpiled_depth`'s dict (depth, num_qubits, transpiled `ops`,
        basis_gates, optimization_level -- see quantum/feature_map.py; depth
        without a named basis is not comparable between arms, D27.4) plus an
        explicit `n_qubits` key for a runner that wants the qubit count
        without reaching into `ops`/`num_qubits` naming. Never None for this
        arm -- the feature map always exists once constructed; the `| None`
        return type matches the frozen interface for arms where circuit
        introspection can legitimately be unavailable.
        """
        info = transpiled_depth(self.feature_map)
        info["n_qubits"] = self.feature_map.num_qubits
        return info
