"""PLAN.md 3E.4 -- SWAP-test quantum autoencoder (QAE). D27.2 supersedes the
original spec's "mirrors 3A.6" line: 3A.6 (a spectral-spatial conv AE) does not
exist, and comparing it to this 8-dim vector QAE would compare input
representations, not architectures. The actual classical partner is a dense
MLP AE on the SAME 8 reduced features, built separately in
quantum/classical_baselines.py -- this module only has to keep the frozen
interface (fit/score/circuit_info) so the two are interchangeable.

Never a dependency of the operational pipeline (Roadmap Section 1.5, Section 9.10).
Never runs on the edge device (Roadmap Section 6). PC + simulator only.

REGISTER LAYOUT (13 qubits for the frozen defaults n_features=8, n_latent=4):
    data[0 .. n_features-1]        the encoded pixel, n_latent of it is "kept"
                                    (never touched by the SWAP test) and the
                                    remaining n_trash = n_features - n_latent
                                    qubits are "trash" -- what the ansatz is
                                    trained to compress away.
    ref[n_features .. n_features+n_trash-1]   n_trash reference qubits, always
                                    |0...0> (no gate ever touches them).
    ancilla                        1 qubit, the SWAP-test control.
n_features=8, n_latent=4 -> n_trash=4 -> total = 8 + 4 + 1 = 13, matching the
register layout the brief states.

WHY EXACT STATEVECTOR, NOT SamplerQNN. D27.0 finding 5 already established,
for the quantum-kernel arm, that an exact `Statevector` per sample is ~180x
faster than a shot-based primitive AND agrees to 1.1e-11 -- same object,
O(N) instead of O(N^2) primitive calls. The same argument applies here at the
per-SAMPLE level: this module never needs a shot distribution, only the
scalar P(ancilla=0), which `Statevector.probabilities([ancilla])` returns
exactly and noiselessly. Building a SamplerQNN (which qiskit_machine_learning
0.9.1 confirms works with a V2 sampler, per the brief) would add shot noise
AND primitive-call overhead for no benefit: the identical-states and
orthogonal-states correctness checks below need EXACT 1.0 / 0.0 fidelities,
which a finite-shot Sampler cannot give without either huge shot counts or a
tolerance loose enough to also pass a broken SWAP test. `StatevectorSampler`
is therefore not used anywhere in this module -- a deliberate deviation from
the brief's seeding note, which describes how to seed a Sampler IF one is
used, not a requirement to use one. Logged here because the brief asked to be
told about anything that contradicts it: this doesn't contradict a measured
fact, it just takes the faster of two equally-valid implementations the
branch's own D27.0 already demonstrated. It also sidesteps a whole class of
problem the VQC arm has to manage explicitly: `StatevectorSampler`'s default
1024 shots quantize every probability to k/1024, which quantizes scores and
creates real ties on `score_scene_natural`'s few-hundred-pixel populations.
`score()` here has no shots at all -- `Statevector.probabilities` is an exact
marginal, not a shot estimate -- so there is no quantization and no `shots`
parameter to expose; this is a genuine difference from VQC's score(), not an
oversight, and is called out here so the comparison table doesn't assume the
two arms share a noise model.

DETERMINISM. `fit`'s only randomness is the ansatz's initial point, drawn
explicitly from `np.random.default_rng(seed)` -- U(-pi, pi) over
n_parameters, matching the VQC arm's own convention (there is no principled
reason to diverge, and matching makes the two arms' "does seed X reproduce"
checks comparable). `COBYLA.minimize` is always called with this explicit
`x0`, never `None`, so `qiskit_machine_learning`'s
`NeuralNetworkClassifier._choose_initial_point` fallback to the module-level
`algorithm_globals.random` RNG -- the third nondeterminism source the VQC
agent found -- cannot fire here: this module never constructs a
`NeuralNetworkClassifier` (or any qiskit-ml training wrapper) at all; fits
are driven directly by `COBYLA(...).minimize(fun=objective, x0=x0)`. Combined
with exact (shot-free) statevector evaluation, `fit` then `score` at a fixed
`seed` is exactly reproducible -- see
test_fit_and_score_deterministic_at_fixed_seed.

THE SWAP TEST'S OWN FIDELITY FLOOR IS 0.5, NOT 0. Textbook SWAP-test algebra:
measuring the ancilla in the computational basis after H - controlled-SWAP - H
gives P(ancilla=0) = 1/2 + 1/2*|<a|b>|^2, so two ORTHOGONAL states still give
P(0)=0.5, not 0. `fidelity := 1 - 2*P(ancilla=1)` (equivalently `2*P(0) - 1`)
undoes that floor, giving fidelity=1 for identical states and fidelity=0 for
orthogonal ones -- this is the quantity _fidelity_batch returns, and it is
what test_quantum_qae.py's identical/orthogonal checks pin. Getting this
formula wrong (e.g. reporting raw P(ancilla=0) as "fidelity") would silently
compress the reported range into [0.5, 1.0] without raising anything.

DIRECTION: the SWAP test measures FIDELITY (trash state close to |0...0> ->
well compressed -> the ansatz has seen this kind of input before -> NORMAL).
The anomaly score this module reports is INFIDELITY = 1 - fidelity, so
HIGHER = MORE ANOMALOUS, per the frozen interface. This is the easiest arm in
the branch to get backwards (a flipped score gives a mirrored-but-plausible
AUC, not an obvious failure) -- test_quantum_qae.py's AUC-on-training-split
check exists specifically to catch that flip, on top of the sign already
being pinned by the SWAP-test formula above.

TRAP CARRIED FROM D27.0 (VQC's, but the same optimizer): qiskit_machine_learning's
COBYLA(maxiter=k) maps to scipy's `maxfun`, silently clamped to a MINIMUM of
n_params + 2. At the frozen default ansatz_reps=3, real_amplitudes(8, reps=3)
has 32 parameters (verified: `real_amplitudes(8, reps=3).num_parameters == 32`,
matching D27.0's own count for the same ansatz shape), so any maxiter < 34
actually performs 34 evaluations. `fit` reports the optimizer's own `nfev`
in `self.n_objective_evals`, not the requested `maxiter`, for exactly this
reason.

MEASURED (this module, 2026-08-22, CPU, statevector backend), DIRECTLY, not
extrapolated: one `_fidelity_batch` call over the REAL 300-row background-
only training split (`build_split()`'s cached default, n_features=8/
n_latent=4/reps=2/ansatz_reps=3, 13 qubits total) -- i.e. exactly one COBYLA
objective evaluation at the frozen defaults -- costs 5.99 s (19.95 ms/sample,
consistent with the per-sample estimate this replaces). At maxiter=150 (no
clamping: 150 > 34) a real fit is 150 * 5.99s = ~898 s = ~15.0 min --
well under the brief's ~1h budget estimate, and reported here as the number
the brief asked for rather than left unverified.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA

from quantum.data import QuantumSplit
from quantum.feature_map import build_feature_map, transpiled_depth


def _swap_test_block(n_trash: int) -> tuple[QuantumCircuit, list[int], list[int], int]:
    """The bare SWAP-test sub-circuit, on its OWN register of 2*n_trash + 1
    qubits: [trash(0..n_trash-1), ref(n_trash..2n_trash-1), ancilla(last)].
    H - controlled-SWAP(ancilla; trash_i, ref_i) for each i - H on the
    ancilla, no measurement (the caller reads P(ancilla) off the exact
    statevector, see module docstring).

    Pulled out as its own function -- rather than inlined once inside
    _build_circuit -- for one reason: test_quantum_qae.py's identical/
    orthogonal correctness checks compose THIS SAME block (via
    QuantumCircuit.compose) rather than reimplementing the gate sequence, so
    the SWAP test is pinned as the exact code path fit/score run through, not
    a parallel hand-copy that could silently drift from it.

    Returns (block, trash_qubits, ref_qubits, ancilla_idx) -- local qubit
    indices within `block`'s own register, for the caller to map onto a
    larger register via `compose(block, qubits=...)`.
    """
    if n_trash < 1:
        raise ValueError(f"_swap_test_block: n_trash must be >= 1, got {n_trash}")
    total = 2 * n_trash + 1
    trash = list(range(n_trash))
    ref = list(range(n_trash, 2 * n_trash))
    ancilla = total - 1

    block = QuantumCircuit(total, name="swap_test")
    block.h(ancilla)
    for t, r in zip(trash, ref):
        block.cswap(ancilla, t, r)
    block.h(ancilla)
    return block, trash, ref, ancilla


def _build_circuit(n_features: int, n_latent: int, reps: int, ansatz_reps: int, kind: str
                    ) -> tuple[QuantumCircuit, list[Parameter], list[Parameter], int]:
    """Feature map (input-encoding) + real_amplitudes ansatz (trainable) on
    the n_features data qubits, then _swap_test_block composed onto [last
    n_trash data qubits, n_trash fresh reference qubits, 1 fresh ancilla].

    The "trash" qubits are, by convention, the LAST n_trash of the n_features
    data qubits (data[n_latent:]) -- an arbitrary but fixed choice; any subset
    would do, since the trained ansatz is free to rotate whichever subspace it
    likes into that slot, and real_amplitudes' full entanglement between all
    n_features qubits means no qubit is structurally privileged as "latent"
    before training.

    Returns (circuit, feature_map_parameters, ansatz_parameters, ancilla_index).
    The two Parameter lists are kept SEPARATE (not circuit.parameters, which
    would mix and sort them together) because fit/score bind them from two
    different sources every call: feature_map_parameters <- one data row,
    ansatz_parameters <- the (fixed, once trained) weight vector.
    """
    n_trash = n_features - n_latent
    if n_trash < 1:
        raise ValueError(
            f"_build_circuit: n_latent ({n_latent}) must be < n_features ({n_features}) "
            f"-- at least one trash qubit is required for a SWAP-test cost")

    data_qubits = list(range(n_features))
    trash_qubits = data_qubits[n_latent:]
    ref_qubits = list(range(n_features, n_features + n_trash))

    fmap = build_feature_map(n_features, kind=kind, reps=reps)
    ansatz = real_amplitudes(n_features, reps=ansatz_reps)
    swap_block, block_trash, block_ref, block_anc = _swap_test_block(n_trash)

    total_qubits = n_features + n_trash + 1
    ancilla = total_qubits - 1
    qc = QuantumCircuit(total_qubits, name="qae")
    qc.compose(fmap, qubits=data_qubits, inplace=True)
    qc.compose(ansatz, qubits=data_qubits, inplace=True)
    qc.compose(swap_block, qubits=trash_qubits + ref_qubits + [ancilla], inplace=True)

    return qc, list(fmap.parameters), list(ansatz.parameters), ancilla


def _fidelity_batch(circuit: QuantumCircuit, fmap_params: list[Parameter],
                     ansatz_params: list[Parameter], ancilla: int,
                     X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """[N, n_features] angle-encoded rows + a fixed ansatz weight vector ->
    [N] float64 SWAP-test fidelity per row (1.0 = identical to |0...0> trash
    reference, 0.0 = orthogonal -- see module docstring for the *2-1 formula
    that removes the SWAP test's own 0.5 floor).

    Binds ansatz_params ONCE per call (they're fixed across the whole batch --
    only fmap_params vary per row) and re-binds only the input parameters
    per row, via two separate `dict`s merged per-row rather than
    `assign_parameters` on the full parameter vector at once, so the (fixed)
    weight values aren't re-packed into a new dict on every row.

    O(N) exact statevector simulations (Statevector.from_instruction), never
    O(N^2) and never shot-based -- see module docstring's "why exact
    statevector" note and its measured ~20ms/sample cost.
    """
    X = np.asarray(X, dtype=np.float64)
    weight_bind = dict(zip(ansatz_params, np.asarray(weights, dtype=np.float64)))
    fidelities = np.empty(X.shape[0], dtype=np.float64)
    for i, row in enumerate(X):
        bind = dict(weight_bind)
        bind.update(zip(fmap_params, row))
        bound = circuit.assign_parameters(bind)
        sv = Statevector.from_instruction(bound)
        p0, p1 = sv.probabilities([ancilla])
        fidelities[i] = np.clip(1.0 - 2.0 * p1, 0.0, 1.0)
    return fidelities


class QuantumAutoencoderArm:
    """PLAN.md 3E.4's frozen interface. Compresses n_features qubits to
    n_latent via a SWAP-test trash-state fidelity cost, trained on
    BACKGROUND-ONLY rows (unsupervised -- an autoencoder trained on the
    anomalies it must later flag would learn to reconstruct them, which is
    self-defeating, D27.2 / the brief's restatement of Section 3A.6's original
    reasoning).
    """
    name: str = "qae"
    supervision: str = "unsupervised"

    def __init__(self, *, n_features: int = 8, n_latent: int = 4, reps: int = 2,
                 ansatz_reps: int = 3, maxiter: int = 150, seed: int = 0, kind: str = "zz"):
        self.n_features = n_features
        self.n_latent = n_latent
        self.n_trash = n_features - n_latent
        self.reps = reps
        self.ansatz_reps = ansatz_reps
        self.maxiter = maxiter
        self.seed = seed
        self.kind = kind

        self._circuit, self._fmap_params, self._ansatz_params, self._ancilla = _build_circuit(
            n_features, n_latent, reps, ansatz_reps, kind)
        self._weights: np.ndarray | None = None

        self.fit_seconds: float = 0.0
        self.n_objective_evals: int = 0

    def fit(self, split: QuantumSplit) -> None:
        """Trains the ansatz's `n_parameters` weights via COBYLA to MINIMIZE
        mean infidelity (1 - fidelity) over BACKGROUND-ONLY rows
        (split.X_train[split.y_train == 0]) -- never the anomaly rows, per
        the brief and D27.2.

        Records self.fit_seconds (wall clock) and self.n_objective_evals
        (COBYLA's own `result.nfev`, the ACTUAL evaluation count after
        scipy's maxfun-minimum clamp -- see module docstring's TRAP note --
        never the raw `maxiter` argument).

        Initial point: U(-pi, pi) over the ansatz's n_parameters, drawn from
        `np.random.default_rng(self.seed)` and passed to `COBYLA.minimize` as
        an explicit `x0` (never left `None`) -- see module docstring's
        DETERMINISM note for why this is what makes `seed` alone reproduce a
        fit, and why the VQC-side `algorithm_globals.random` fallback risk
        does not apply to this module.
        """
        X_bg = np.asarray(split.X_train, dtype=np.float64)[np.asarray(split.y_train) == 0]
        if X_bg.shape[0] == 0:
            raise ValueError(
                "QuantumAutoencoderArm.fit: split.X_train has zero background "
                "(y_train == 0) rows -- nothing to train on")

        rng = np.random.default_rng(self.seed)
        x0 = rng.uniform(-np.pi, np.pi, size=len(self._ansatz_params))  # matches vqc_encoder's convention

        def objective(weights: np.ndarray) -> float:
            fidelities = _fidelity_batch(self._circuit, self._fmap_params, self._ansatz_params,
                                          self._ancilla, X_bg, weights)
            return float(np.mean(1.0 - fidelities))

        optimizer = COBYLA(maxiter=self.maxiter)
        t0 = time.perf_counter()
        result = optimizer.minimize(fun=objective, x0=x0)
        self.fit_seconds = time.perf_counter() - t0

        self._weights = np.asarray(result.x, dtype=np.float64)
        self.n_objective_evals = int(result.nfev)

    def score(self, X: np.ndarray) -> np.ndarray:
        """[N, n_features] angle-encoded rows -> [N] float64 INFIDELITY
        (1 - SWAP-test fidelity against the trash reference), HIGHER = MORE
        ANOMALOUS. Raises RuntimeError if called before fit().

        EXACT, not shot-sampled: every score comes from
        `Statevector.probabilities`, not a finite-shot `Sampler`, so unlike
        the VQC arm there is no k/1024-style score quantization and no ties
        this introduces on `score_scene_natural`'s small populations -- see
        module docstring's "WHY EXACT STATEVECTOR" and DETERMINISM notes.
        """
        if self._weights is None:
            raise RuntimeError(
                "QuantumAutoencoderArm.score: call fit() first -- no trained ansatz weights")
        fidelities = _fidelity_batch(self._circuit, self._fmap_params, self._ansatz_params,
                                      self._ancilla, X, self._weights)
        return 1.0 - fidelities

    def circuit_info(self) -> dict | None:
        """quantum.feature_map.transpiled_depth() of the FULL circuit (feature
        map + ansatz + SWAP test, all n_qubits of it -- not just the ansatz),
        plus n_qubits/n_latent/n_trash/n_parameters/optimizer/maxiter, per the
        frozen interface. transpiled_depth binds all free parameters to 0.0
        before transpiling (its own documented convention, not this module's)
        -- see quantum/feature_map.py::transpiled_depth for why that's a real
        measurement and not a zero-gate artifact at optimization_level=1.
        """
        if self._circuit is None:
            return None
        info = transpiled_depth(self._circuit)
        info.update(
            n_qubits=self._circuit.num_qubits,
            n_latent=self.n_latent,
            n_trash=self.n_trash,
            n_parameters=len(self._ansatz_params),
            optimizer="COBYLA",
            maxiter=self.maxiter,
        )
        return info
