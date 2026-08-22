# Branch 3E — Quantum research arm

Status: **in progress** (2026-08-22). The literature search below is complete; the
comparison table is added when `quantum/classical_vs_quantum.py` has run.

This branch's deliverable is **a comparison table, not an advantage claim** (plan.md §13
rule 4). The word "advantage" appears in this document only to disclaim it.

---

## 1. Novelty framing — the scoped claim, and how the search changed it

### 1.1 What plan.md §3E.8 originally proposed to claim

> *no existing published work applies VQC/QAE feature encoding directly to hyperspectral
> anomaly detection.*

**This claim is false as written.** The search below found it. Recording the correction
here rather than quietly weakening the wording, because a novelty claim that dissolves
under a reviewer's first search is worse than no claim at all.

### 1.2 Search record

**Date run:** 2026-08-22.
**Access route:** a general web search index reaching arXiv, IEEE Xplore, ResearchGate,
IOPscience, MDPI, Taylor & Francis and OSTI, plus direct `arxiv.org/abs/` fetches for the
three closest hits. This is **not** the same as querying each database's native interface,
and a formal review should redo it against arXiv's own API and IEEE Xplore directly. The
queries are recorded verbatim so that is possible.

**Queries, verbatim:**

1. `variational quantum classifier hyperspectral anomaly detection`
2. `quantum autoencoder hyperspectral anomaly detection arXiv`
3. `quantum kernel support vector machine hyperspectral remote sensing classification 2025`
4. `"hyperspectral" "anomaly detection" quantum machine learning variational circuit qubit encoding survey`
5. `quantum computing hyperspectral anomaly detection RX detector comparison simulator qiskit`

### 1.3 What the search found

**Directly overlapping — quantum + hyperspectral + anomaly detection:**

| ref | what it is | overlap with 3E |
|---|---|---|
| **arXiv 2605.04388** — *Hyperspectral Anomaly Detection Using Einstein Fuzzy Computing and Quantum Neural Network* (Lin, Young, Langari; 6 May 2026) | "HyFuHAD", fuses a quantum detector with classical detectors for hyperspectral AD | **Direct.** A quantum neural network applied to hyperspectral anomaly detection. This alone falsifies §3E.8's original wording. |

**Adjacent — quantum kernels on hyperspectral, but classification not anomaly detection:**

| ref | what it is | overlap |
|---|---|---|
| **arXiv 2605.17587** — *Large-Scale Quantum Kernels for Hyperspectral Data Classification* (Delilbasic, Miroszewski, Wijata, Nalepa, Mielczarek, Riedel, Cavallaro; 17 May 2026) | self-described "first large-scale study of fidelity-quantum-kernel SVMs for hyperspectral data classification"; Indian Pines + methane detection; tensor-network contraction + GPU; explicitly **without** prior dimensionality reduction | Overlaps §3E.5's method (fidelity quantum kernel) and the data domain, but the **task is classification**, and it deliberately avoids the PCA reduction that §3E.2 makes central. |

**Adjacent — QAE / VQC anomaly detection in other domains:**

| ref | domain |
|---|---|
| arXiv 2510.21837 — *Quantum Autoencoders for Anomaly Detection in Cybersecurity* (notably: 8-feature QAE, dense-angle encoding, `RealAmplitudes` ansatz — the same configuration 3E arrives at independently) | cybersecurity |
| arXiv 2404.17613 — *Quantum Patch-Based Autoencoder for Anomaly Segmentation* (SWAP-test anomaly map) | general images |
| arXiv 2607.02135 — *Quantum Convolutional Autoencoders for Reconstruction-Based Anomaly Detection* | general |
| arXiv 2606.27411 — *Compression-Driven Anomaly Detection in Brain MRI Using an Interpretable Quantum Autoencoder* | medical imaging |
| arXiv 2410.04154 · arXiv 2504.13113 | time series |
| *Quantum support vector data description for anomaly detection* (IOPscience, `10.1088/2632-2153/ad6be8`) | general one-class |

**Reviews checked for a hyperspectral mention:**

| ref | result |
|---|---|
| **arXiv 2408.11047** — *Quantum machine learning algorithms for anomaly detection: A review* (Corli, Moro, Dragoni, Dispenza, Prati; Aug 2024, rev. Mar 2025) | Application domains covered are **"cybersecurity, fraud detection, particle physics."** **No hyperspectral or remote-sensing mention found.** |

### 1.4 The revised scoped claim

Everything below is a claim about a **protocol**, not about a territory being empty:

> As of 2026-08-22 we found no published work that evaluates a variational quantum
> classifier, a SWAP-test quantum autoencoder, and a fidelity quantum kernel **side by side
> on hyperspectral anomaly detection**, on a **shared PCA feature basis**, against
> supervision-matched classical baselines, under a **leakage-controlled split**.

And what is **not** claimed, explicitly:

- **Not** that quantum ML on hyperspectral anomaly detection is unexplored. arXiv 2605.04388 explores it.
- **Not** that fidelity quantum kernels on hyperspectral data are new. arXiv 2605.17587 is larger-scale than anything here.
- **Not** that the QAE configuration is novel. arXiv 2510.21837 reaches an 8-feature / dense-angle / `RealAmplitudes` QAE independently.
- **Not** any advantage, of any kind, at any scale. See §3 below.

The residual contribution is the **comparison discipline**: identical features, identical
split, supervision declared per arm. That is a smaller claim than §3E.8 imagined, and it
is one this branch can actually support with its own numbers.

---

## 2. Method — what is being compared

Six arms, on **identical** 8-dimensional PCA features derived through
`preprocessing.harmonize.reduce_bands`, scored on the **same** flightline-disjoint HAD100
test split. Read as **two families of three**, never as one ranking of six:

| family | arms |
|---|---|
| unsupervised | dense classical AE · `OneClassSVM(rbf)` · quantum autoencoder · quantum kernel + `OneClassSVM(precomputed)` |
| supervised | `SVC(rbf)` · VQC |

Comparing a supervised VQC against an unsupervised classical AE measures **supervision**,
which is a larger effect than anything quantum — the same error §3E.2 warns about one level
up, where comparing across feature bases measures the basis. Hence the supervised classical
partner, and hence the `supervision` column.

**Dataset: HAD100 only, and this is forced rather than chosen.** `classical_reduce` reuses
`reduce_bands`, which consumes `harmonize` output, which requires per-band wavelengths. ABU
and HYDICE ship none (O8 / D13.4). plan.md §13 rule 6 therefore binds 3E exactly as it
binds 3B: no ABU or HYDICE quantum number may be produced, implied, or interpolated.

---

## 3. What the numbers in this document do NOT mean

**Wall-clock is `AerSimulator` CPU time and carries no implication about QPU runtime.**
At 8 qubits every circuit here is classically simulable in milliseconds. The proof is in
this branch's own code: `quantum/quantum_kernel.py` ships two Gram implementations, the
specified `FidelityQuantumKernel` and an exact statevector overlap, and they agree to
`1.1e-11` while the statevector path runs **~180× faster** (measured 2026-08-22: N=40 →
0.122 s vs 22.4 s). A "quantum kernel" that a laptop reproduces exactly, faster, by
multiplying two matrices is a useful pedagogical object and not evidence of anything else.

**Circuit depth is reported transpiled**, to `basis_gates = ["rz","sx","x","cx"]` at
`optimization_level = 1`, both recorded in the results JSON. Depth without a named basis
and coupling map is not comparable between arms.

**No hardware result appears anywhere.** plan.md O1: no IBM Quantum account is held, so
§3E.7 is a prerequisite task, not a deliverable. Nothing in this document was run on a QPU.

---

## 4. Results

*Pending — added when `quantum/classical_vs_quantum.py` completes its run.*
