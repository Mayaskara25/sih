# Branch 3E — Quantum research arm

Status: **complete** (2026-08-22). The literature search, method and disclaimers
(§1–§3) are binding; §4 holds the run's results.

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

Run completed 2026-08-22, `git_sha 4f47a40`, full manifest in
`experiments/quantum_results/run_manifest.json` (package versions: qiskit 2.5.2,
qiskit-aer 0.17.2, qiskit-machine-learning 0.9.1; seed 0; 8 features). All 8 rows scored all
28 test scenes with zero failures (224/224 scene rows `status=ok`). Total wall-clock was
53 min, under the ~2 h budgeted: VQC fit took 1671 s (~28 min) and QAE 796 s (~13 min),
against the 35/15 min estimates in D27.0 finding 6.

### 4.1 Primary table — scene-macro pooled, natural prevalence

ROC-AUC and PR-AUC are macro means over the 28 test scenes, computed per scene by
`score_scene_natural` with D27.7's background re-weighting (`sample_weight`) applied — never
on the balanced split. **Wall-clock is `AerSimulator` CPU time and carries no QPU
implication (D27.4 / §3).** Depth is transpiled to `basis_gates=["rz","sx","x","cx"]`,
`optimization_level=1`.

| row_id | supervision | swept | roc_auc_MACRO | pr_auc_MACRO | n_qubits | depth (transpiled) | fit_wall_clock_s (AerSimulator/CPU) |
|---|---|---|---|---|---|---|---|
| classical_ae | unsupervised | yes | 0.5819 | 0.2054 | — | — | 0.72 |
| classical_ocsvm | unsupervised | yes | 0.6293 | 0.1416 | — | — | 0.01 |
| classical_svc | supervised | yes | **0.8130** | 0.4078 | — | — | 0.01 |
| rx_8feat | unsupervised | yes | 0.7316 | **0.5132** | — | — | <0.01 |
| vqc | supervised | no | 0.4736 | 0.0723 | 8 | 41 | 1671.0 |
| qae | unsupervised | no | 0.4186 | 0.0184 | 13 | 79 | 796.1 |
| quantum_kernel_spec | unsupervised | no | 0.4017 | 0.0206 | 8 | 33 | 1.09 |
| quantum_kernel_valscale | unsupervised | yes | 0.6815 | 0.1455 | 8 | 33 | 1.29 |

Read as two families of three (§2), never one ranking of eight. The headline result is a
negative one, stated plainly: **every quantum arm loses to `rx_8feat`, an unsupervised
classical baseline on the same 8 features** (0.4736 / 0.4186 / 0.4017 / 0.6815 vs 0.7316
scene-macro ROC-AUC). This is a legitimate §3E.6 outcome, not a defect to fix. The only arm
that beats `rx_8feat` anywhere is the supervised `classical_svc` (0.8130), which is the
supervision effect §2 predicted, not a quantum effect.

### 4.2 Pixel-micro pooled figures — SECONDARY, not comparable to any primary number

Micro-pooled over all scored pixels of all 28 scenes. Labelled explicitly because an
unlabelled pooled figure is banned in this repo (D27.7): these weight large scenes more than
small ones and are shown only for completeness beside the macro table above.

| row_id | roc_auc_micro | pr_auc_micro |
|---|---|---|
| classical_ae | 0.6580 | 0.3453 |
| classical_ocsvm | 0.6802 | 0.2284 |
| classical_svc | 0.7858 | 0.3880 |
| rx_8feat | 0.7798 | 0.5914 |
| vqc | 0.4633 | 0.0795 |
| qae | 0.4518 | 0.0352 |
| quantum_kernel_spec | 0.3763 | 0.0280 |
| quantum_kernel_valscale | 0.6483 | 0.1290 |

The ranking agrees with the macro table on every conclusion drawn here; nothing in this
document rests on a micro figure alone.

### 4.3 Kernel concentration diagnostic — the run reproduced D28

Mean off-diagonal of the fitted background Gram (300 train-background rows, exact statevector
path), from the run manifest beside D28's measured values:

| configuration | mean off-diag (this run) | mean off-diag (D28) |
|---|---|---|
| zz reps=1, scale 1.0 (diagnostic-only, never scored) | 0.0484 | 0.0484 |
| zz reps=2, scale 1.0 (`quantum_kernel_spec`) | 0.0384 | 0.0384 |
| zz reps=2, scale 0.5 (`quantum_kernel_valscale`) | 0.0706 | 0.0706 |

Reproduced to the reported precision. The pattern D28 documented also reproduces behaviourally:
the spec-scale kernel scores below chance on held-out flightlines (test ROC-AUC 0.4017 against
val 0.6872), while rescaling the angles to `[0, π/2]` lifts test ROC-AUC to 0.6815 with the same
circuit, the same feature map and the same 300 training rows. One constant moves the kernel arm
from inverted to competitive-with-nothing-better-than-RX — still below `rx_8feat`.

### 4.4 `quantum_kernel_valscale` differs from `quantum_kernel_spec` — both are reported

Validation selected angle_scale 0.5 (val AUC 0.7044) over the specified 1.0 (val 0.6872), so
the two rows are **not identical** and both appear in the table. Had val selected 1.0 the two
rows would have been byte-identical, and that would be stated rather than presenting one result
twice. The full sweep `{1.0 → 0.5 → 0.25 → 0.125}` shows val AUC falling monotonically past
0.5 while the Gram's mean off-diagonal rises monotonically — the concentration/scale trade-off
D28 measured, recomputed inside the runner rather than copied from the note.

### 4.5 Who was tuned, and who was not

`swept=True` on five rows: the four cheap arms (classical AE latent size, OCSVM γ/ν grid,
SVC C/γ grid, RX regularizer) and the quantum kernel's angle scale. **`vqc` and `qae` are
`swept=False`: a single COBYLA fit costs ~28 min (VQC) and ~13 min (QAE) at the frozen
defaults, so their hyperparameters (`ansatz_reps=3`, `maxiter` 200/150) ran once, unswept.**
A swept arm beside an unswept one is not a like-for-like comparison — the quantum kernel rows
are directly comparable to each other and to the sweep-matched classical arms; the VQC/QAE
rows measure single-point performance at their frozen configurations, and their low scores must
be read with that asymmetry in mind.

### 4.6 F1 is present and uninformative

Every arm lands at F1_MACRO ≈ 0.019–0.044 with precision ≈ 0.010 (≈ natural prevalence) and
recall ≈ 1.0. The thresholds were calibrated per-arm on val for 0.98 target recall on a far
more balanced split; scored at ~0.4 % natural prevalence they admit essentially every pixel.
This is recall-first calibration under prevalence mismatch, not a bug, and it hits all eight
arms identically — F1 discriminates nothing here. ROC-AUC (prevalence-invariant) and PR-AUC
(weight-corrected per D27.7) carry the entire comparison.

### 4.7 Shot noise is negligible — closed

Measured previously at the real test-split size: scoring with 1024 shots costs 0.0001 ROC-AUC
against exact statevector probabilities despite 173/600 duplicate score values; even 256 shots
costs 0.0024. All quantum arms in this run use exact probabilities; nobody needs to reopen the
shots question for this branch.

### 4.8 What these results are not comparable to

Not compared to `experiments/rx_vs_ae/results_pooled.csv`'s HAD100 row (D27.9): those
detectors are unsupervised and per-scene and never generalize across flightlines, whereas every
arm here trains and is tested on held-out flights — two tasks sharing a label. Not extended to
ABU or HYDICE (§13 rule 6 / D27.6). No hardware number appears anywhere (§3, O1). Nothing here
is a quantum-advantage claim (§13 rule 4); the table is the deliverable, and the table says the
quantum arms lose to a classical RX on the same features under the same split.
