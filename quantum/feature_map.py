"""PLAN.md 3E.2 -- angle-encoding feature maps and the classical PCA-reduction path
that feeds them. Never a dependency of anomaly/, segmentation/, preprocessing/,
pipeline/ (Roadmap §1.5, §9.10) -- PC + simulator only.

QUBIT COUNT IS THE HARD CONSTRAINT on this whole branch: every feature map here
is an n_features-qubit circuit with one angle-encoded parameter per qubit, so
n_features IS the qubit count. 8..16 is enforced because a Bell-sized (2-qubit)
sanity circuit gives no branch-relevant signal and 16+ qubits is past what
AerSimulator's statevector backend, and any near-term hardware, can do in the
time budget this project has (§0.2).

DEPRECATED CLASSES, NOT FUNCTIONS: `ZZFeatureMap`/`RealAmplitudes`/`NLocal` as
classes emit DeprecationWarning as of qiskit 2.1+. The function forms used here
(`zz_feature_map`, `z_feature_map`, `pauli_feature_map`) are verified (by this
module's own test suite, warnings.catch_warnings) to emit none.

FIT ORDER MATTERS (D15): `reduce_bands` (preprocessing/harmonize.py) must be fit
on TRAIN-split pixels only and then REUSED, never refit, on val/test/eval. This
module's `fit_transformer` is the ONE place a PCA + angle-range pair gets fit for
the quantum branch -- classical_reduce and quantum.data.build_split both route
through it, so "refit at inference is a bug" is enforced structurally (one fit
path to audit) rather than by convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map

from core.contracts import SceneMeta
from preprocessing.harmonize import harmonize, reduce_bands

BASIS_GATES = ["rz", "sx", "x", "cx"]

_KIND_FNS = {"zz": zz_feature_map, "z": z_feature_map, "pauli": pauli_feature_map}


def build_feature_map(n_features: int, *, kind: str = "zz", reps: int = 2,
                       entanglement: str = "linear") -> QuantumCircuit:
    """Angle-encoding feature map on n_features qubits (one free Parameter per
    qubit, bound later to a [0, pi]-scaled reduced-band pixel by the caller).

    kind: 'zz' -> zz_feature_map (ZZFeatureMap's function form), 'z' ->
    z_feature_map, 'pauli' -> pauli_feature_map. All three are the FUNCTION
    forms (verified to emit no DeprecationWarning at qiskit 2.5.2) -- never the
    deprecated NLocal-subclass forms.

    n_features in 8..16 -- qubit count is this branch's hard constraint (see
    module docstring). Raised here, at construction, rather than left to fail
    inside a downstream transpile or simulator call where the qubit-count
    origin of the failure would be less obvious.
    """
    if not (8 <= n_features <= 16):
        raise ValueError(
            f"n_features must be in 8..16 (qubit count is the branch's hard "
            f"constraint, PLAN.md 3E.2), got {n_features}")
    try:
        fn = _KIND_FNS[kind]
    except KeyError:
        raise ValueError(f"kind must be one of {sorted(_KIND_FNS)}, got {kind!r}") from None
    return fn(n_features, reps=reps, entanglement=entanglement)


def transpiled_depth(circuit: QuantumCircuit, *, basis_gates: list[str] = BASIS_GATES,
                      optimization_level: int = 1) -> dict:
    """Transpile `circuit` to `basis_gates` and report depth + gate counts.

    Depth without a NAMED basis is not comparable -- the same logical circuit
    transpiles to wildly different depths depending on the target gate set, so
    basis_gates and optimization_level are both returned alongside depth rather
    than left implicit.

    Free parameters are bound to 0.0 BEFORE transpiling. transpile() on an
    unbound parametric circuit either fails outright or (depending on pass)
    silently reports a depth for a circuit shape that may not match the bound
    one, because parameter-dependent gate cancellation (e.g. an rz(0) folding
    into a neighbour at optimization_level>=2) cannot be evaluated on a free
    symbol. Binding to zero is a fixed, reproducible choice -- NOT a no-op:
    at optimization_level=1 (used here) a zero-angle rz is NOT folded away
    (measured: zz_feature_map(8, reps=2, 'linear') -> depth 33, {rz:46, cx:28,
    sx:16}), so this measurement is meaningful evidence about circuit cost, not
    an artifact of optimizing away trivial gates. A higher optimization_level
    WOULD fold zero-angle rz gates and make the depth incomparable across
    optimization_level settings -- another reason optimization_level is a
    returned field, not just an argument.
    """
    if circuit.num_parameters:
        bound = circuit.assign_parameters([0.0] * circuit.num_parameters)
    else:
        bound = circuit

    basis_gates = list(basis_gates)
    t = transpile(bound, basis_gates=basis_gates, optimization_level=optimization_level)
    return {
        "depth": t.depth(),
        "num_qubits": t.num_qubits,
        "ops": dict(t.count_ops()),
        "basis_gates": basis_gates,
        "optimization_level": optimization_level,
    }


# --------------------------------------------------------------- classical reduction --

@dataclass
class QuantumFeatureTransformer:
    """Bundles reduce_bands' fitted PCA/kPCA transformer together with the
    MinMax-to-[0, pi] angle-encoding range, as ONE object with a single
    .transform -- reduce_bands' transformer alone carries no angle range, and a
    bare MinMaxScaler alone carries no band reduction, so a caller reusing a
    fitted transformer across train -> val/test (D15) needs both fitted pieces
    glued together in fit order, in one object, or "reuse without refitting"
    has no single thing to hold onto and pass around.

    pca: anything with .transform (sklearn PCA/KernelPCA, or reduce_bands'
         return either way).
    lo, hi: [n_features] float64, the TRAIN-fitted per-component min/max used
         to scale to [0, pi]. Fixed after fitting -- see .transform.
    """
    pca: object
    lo: np.ndarray
    hi: np.ndarray
    n_features: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: [N, B] raw harmonized-band pixels (NO NaN rows -- caller filters,
        same convention as reduce_bands) -> [N, n_features] angles in [0, pi].

        CLIPS to [lo, hi] before scaling. A val/test pixel whose reduced value
        falls outside the TRAIN-fitted [lo, hi] would otherwise map outside
        [0, pi] -- an angle encoding has no defined meaning past pi (RealAmplitudes-
        style ansatzes and rotation gates are 2*pi-periodic, so an out-of-range
        angle silently aliases to a DIFFERENT valid angle rather than raising).
        Clipping is the deliberate, visible failure mode instead: an out-of-range
        pixel saturates at the nearest encoded extreme rather than aliasing.
        """
        reduced = np.asarray(self.pca.transform(X))
        clipped = np.clip(reduced, self.lo, self.hi)
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return (clipped - self.lo) / span * np.pi


def _lo_hi(reduced: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    if reduced.size == 0:
        return np.zeros(n_features, dtype=np.float64), np.ones(n_features, dtype=np.float64)
    return (reduced.min(axis=0).astype(np.float64), reduced.max(axis=0).astype(np.float64))


def fit_transformer(pool_raw: np.ndarray, *, n_features: int, method: str = "pca"
                     ) -> QuantumFeatureTransformer:
    """Fit reduce_bands' PCA/kPCA AND the [0, pi] angle range on the same raw
    (already-harmonized-band) pixel pool, in one call. THE single fit entry
    point for the quantum branch (classical_reduce and quantum.data.build_split
    both route through this) -- see module docstring.

    pool_raw: [N, B] float, harmonized bands, no NaN rows (caller filters valid
    pixels first -- same convention reduce_bands itself uses).

    Implementation note: reduce_bands expects a [H, W, B] cube, not a flat
    pixel pool, and its ndarray `fit_on` path fits ON `fit_on` -- not on the
    cube argument. Feeding pool_raw reshaped as a [N, 1, B] pseudo-cube (with
    fit_on=pool_raw) is therefore EXACT, not an approximation: the fitted PCA
    and the reduced values used to compute [lo, hi] both come from literally
    the same N pixels, just routed through reduce_bands' cube-shaped API.
    """
    pool_raw = np.asarray(pool_raw, dtype=np.float64)
    if pool_raw.shape[0] < n_features:
        raise ValueError(
            f"fit_transformer: pool has {pool_raw.shape[0]} pixel rows, need >= "
            f"n_features ({n_features}) for PCA to be well-posed")

    pseudo_cube = pool_raw.reshape(pool_raw.shape[0], 1, pool_raw.shape[1]).astype(np.float32)
    reduced_cube, pca = reduce_bands(pseudo_cube, n_components=n_features, method=method,
                                      fit_on=pool_raw)
    reduced_pool = reduced_cube.reshape(-1, n_features)
    lo, hi = _lo_hi(reduced_pool, n_features)
    return QuantumFeatureTransformer(pca=pca, lo=lo, hi=hi, n_features=n_features)


def classical_reduce(cube: np.ndarray, meta: SceneMeta, *, n_features: int = 8,
                      fit_on: object | np.ndarray | None = None
                      ) -> tuple[np.ndarray, QuantumFeatureTransformer]:
    """harmonize(cube, meta) -> reduce_bands(n_components=n_features) -> MinMax
    to [0, pi]. Returns ([H, W, n_features] float64 with NaN preserved
    positionally at input-NaN pixels, QuantumFeatureTransformer).

    Compares a quantum model on the SAME reduced feature basis as the classical
    AE (both start from reduce_bands' output) -- comparing a quantum model on
    one basis against a classical model on another would measure the basis,
    not the model (PLAN.md 3E.2).

    fit_on, threaded straight into reduce_bands / fit_transformer so an
    already-fitted transformer is reused and NEVER refit (D15):
      - a QuantumFeatureTransformer (this function's own return type): reused
        WHOLE, unchanged -- no .fit call happens at all. This is the val/test/
        scoring path.
      - an ndarray [N, B] raw-band pixel pool: fit FRESH on that external pool
        (fit_transformer), then transform `cube`. This is the "fit on a
        sampled train pool, not on this one scene" path.
      - any other object with .transform (e.g. a bare already-fitted PCA with
        no angle range attached yet): reused for reduction; the angle range is
        still fit fresh, from THIS cube's own valid pixels (no pool was given
        to fit a range from).
      - None: fit fresh on this cube's own valid pixels (fit_transformer).
    """
    cube_h, _ = harmonize(cube, meta)
    h, w, b = cube_h.shape
    flat = cube_h.reshape(-1, b).astype(np.float64)
    valid = ~np.any(np.isnan(flat), axis=-1)

    if isinstance(fit_on, QuantumFeatureTransformer):
        transformer = fit_on                                       # D15: reuse, never refit
    elif fit_on is not None and hasattr(fit_on, "transform"):
        reduced_valid = fit_on.transform(flat[valid]) if valid.any() else np.zeros((0, n_features))
        lo, hi = _lo_hi(np.asarray(reduced_valid), n_features)
        transformer = QuantumFeatureTransformer(pca=fit_on, lo=lo, hi=hi, n_features=n_features)
    elif fit_on is not None:
        transformer = fit_transformer(np.asarray(fit_on, dtype=np.float64), n_features=n_features)
    else:
        transformer = fit_transformer(flat[valid], n_features=n_features)

    angles = np.full((flat.shape[0], n_features), np.nan, dtype=np.float64)
    if valid.any():
        angles[valid] = transformer.transform(flat[valid])
    return angles.reshape(h, w, n_features), transformer
