"""PLAN.md 3E.1 -- quantum branch foundation. Proves the qiskit / qiskit-aer /
qiskit-machine-learning stack is installed and composes, and that a Bell circuit
run on AerSimulator produces a sane 50/50 distribution, before anything downstream
(feature_map.py, vqc_encoder.py, ...) is built on top of it.

Never a dependency of the operational pipeline (Roadmap §1.5, §9.10) and never
runs on the edge device (Roadmap §6) -- PC + simulator only, checked here and
nowhere else in the repo.

V1 PRIMITIVES ARE GONE in qiskit 2.5.2: `from qiskit.primitives import Sampler`
raises ImportError (measured). This module doesn't need a Sampler at all -- it
wants exact shot COUNTS for a fixed number of shots, which is what
AerSimulator.run(...).result().get_counts() gives directly; a Sampler would
wrap that in a quasi-probability distribution this check doesn't need. Later
modules that DO want a Sampler-mediated primitive (VQC, SamplerQNN) should use
StatevectorSampler (V2, exact) or qiskit_aer.primitives.SamplerV2 (V2, noisy/
shot-based) -- never the removed V1 `Sampler`.
"""
from __future__ import annotations

import sys

REQUIRED_VERSIONS = {
    "qiskit": "2.5.2",
    "qiskit_aer": "0.17.2",
    "qiskit_machine_learning": "0.9.1",
}

SHOTS = 4096
SEED = 7            # matches the measured fact this module was built against:
                     # {'00': 2039, '11': 2057} at (AerSimulator(seed_simulator=7), 4096 shots)


def check_environment() -> dict[str, str]:
    """Import the three packages and report their __version__. Does not assert
    equality against REQUIRED_VERSIONS -- an environment that has moved on is a
    fact to report, not a reason to crash a foundation module -- but main()
    prints both so a drift is visible immediately.
    """
    import qiskit
    import qiskit_aer
    import qiskit_machine_learning

    return {
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "qiskit_machine_learning": qiskit_machine_learning.__version__,
    }


def run_bell(shots: int = SHOTS, seed: int = SEED) -> dict:
    """Bell circuit (H(0); CX(0,1); measure both) on AerSimulator(seed_simulator=seed).

    Returns {'counts', 'shots', 'seed', 'sigma', 'within_3sigma'}.

    ACCEPT criterion (PLAN.md 3E.1): the measured distribution is within 3 sigma
    of 50/50, sigma = sqrt(shots * 0.25) -- 32.0 at 4096 shots, so each of '00'
    and '11' must land in 2048 +/- 96. within_3sigma additionally requires NO
    other outcome appears at all: a Bell state has exactly two computational-
    basis outcomes, so any '01'/'10' count (even one, from an entangling-gate
    or measurement-order bug) is a correctness signal a pure count-window check
    on '00'/'11' alone would silently pass through.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    sim = AerSimulator(seed_simulator=seed)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    sigma = (shots * 0.25) ** 0.5
    lo, hi = shots / 2 - 3 * sigma, shots / 2 + 3 * sigma
    only_bell_outcomes = set(counts) <= {"00", "11"}
    in_window = all(lo <= counts.get(k, 0) <= hi for k in ("00", "11"))

    return {
        "counts": dict(counts),
        "shots": shots,
        "seed": seed,
        "sigma": sigma,
        "within_3sigma": bool(only_bell_outcomes and in_window),
    }


def main() -> int:
    versions = check_environment()
    print("environment:")
    for name, version in versions.items():
        expected = REQUIRED_VERSIONS[name]
        flag = "" if version == expected else f"  <-- expected {expected}"
        print(f"  {name} = {version}{flag}")

    bell = run_bell()
    print(f"\nbell circuit: H(0); CX(0,1); measure  (seed={bell['seed']}, shots={bell['shots']})")
    print(f"  counts = {bell['counts']}")
    print(f"  3-sigma window around 50/50 = [{bell['shots']/2 - 3*bell['sigma']:.0f}, "
          f"{bell['shots']/2 + 3*bell['sigma']:.0f}]")
    print(f"  within_3sigma = {bell['within_3sigma']}")

    if not bell["within_3sigma"]:
        print("FAIL: Bell distribution outside 3 sigma of 50/50, or a non-Bell "
              "outcome was observed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
