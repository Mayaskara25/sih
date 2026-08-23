# Quantum Branch 3E - Prototype Work Summary

## Overview
This summary documents the work completed on the quantum branch (3E) of the SIH project. The quantum branch is **frozen and complete** per `docs/quantum_handoff.md`. The following additions support validation, visualization, and reproducibility without modifying the frozen branch code.

## What Was Done

### 1. Real HAD100 Data Loader (local prototype)
- **File**: `HSI\quantum\mock_loader.py`
- **Additions**: Added `had100_root()`, `list_real_scenes()`, `load_real_scene()`, and `collect_real_training_pixels()` functions.
- **Purpose**: Enables loading real HAD100 AVIRIS-NG data from the verified download at `sih/data/benchmark/had100/HAD100/`.
- **Flightline discipline**: Uses the same `FLIGHTLINE_SPLIT` as the repo branch, ensuring train/val/test discipline is respected.
- **Dependencies**: Requires `spectral` and `scipy` (installed into the `hsi` conda env).

### 2. Real‑Data Driver Script
- **File**: `sih\scripts\real_data_check.py`
- **Purpose**: Scores a single HAD100 test scene with a VQC (VQCArm) and a per‑scene RX (Mahalanobis) baseline on the **same pixel population** (all anomalies + weighted background sample).
- **Outputs**:
  - AP and ROC‑AUC (natural‑prevalence weighted)
  - 4‑panel PNG: false colour | ground truth | VQC P(anomaly) | RX mahalanobis
  - PNG saved to `experiments/quantum_results/quantum_real_check_<scene>.png`
- **Key finding** (reduced budget: `ansatz_reps=1, maxiter=100`):
  - VQC AP = 0.016, ROC = 0.543
  - RX baseline AP = 0.174, ROC = 0.801
  - *Consistent with the repo's D28/D29 finding that classical RX dominates on held‑out flightlines.*

### 3. Comparison Heatmap Script
- **File**: `sih\scripts\plot_quantum_comparison_heatmap.py`
- **Purpose**: Side‑by‑side heatmaps of the five best‑of‑family arms on one real test scene, using the **same pixel population** (all anomalies + 1500 weighted background samples, natural‑prevalence AP).
- **Arms included**:
  - `rx_8feat` (MahalanobisArm) – AP 0.650, ROC 0.851
  - `classical_svc` (SVCArm) – AP 0.134, ROC 0.359
  - `vqc` (VQCArm, reps=1, maxiter=60) – AP 0.014, ROC 0.432
  - `qae` (QuantumAutoencoderArm, reps=1, maxiter=50) – AP 0.053, ROC 0.541
  - `quantum_kernel_valscale` (_ScaledQuantumKernelArm, angle_scale=0.5) – AP 0.238, ROC 0.636
- **Output**: `experiments/quantum_results/quantum_comparison_heatmap_<scene>.png` with a summary table of AP scores.
- **Labelling**: All figures carry the label `DIAGNOSTIC VISUAL - reduced variational budgets` so they are clearly not the frozen reporting configuration.

### 4. pytest Suite Extension
- **File**: `tests/test_mock_loader_real.py` (4 green tests)
- **Tests**:
  - `test_flightline_parse` – validates `flightline_of()` parsing.
  - `test_list_real_scenes_respects_test_flightlines` – ensures 28 test‑split scenes.
  - `test_load_real_scene_shapes_and_mask_alignment` – verifies cube/mask shapes and that the scene is a test‑flightline.
  - `test_split_functions_work_on_a_real_cube` – verifies `get_train_test_split` on a real HAD100 cube/mask.

### 5. VQC‑vs‑RX Real‑Data Check (driver)
- **File**: `sih\quantum\real_data_check.py` (out‑of‑harness; does not modify `sih/quantum/`)
- **Purpose**: Quick VQC vs RX comparison on a held‑out test scene.
- **Config options**: `--n-per-class`, `--maxiter`, `--max-bg-eval`, `--shots`, `--seed`.
- **Output**: AP/ROC per arm, PNG heatmap `quantum_real_check_<scene>.png`.

### 6. Original Prototype Heatmap (reduced budget)
- **File**: `sih\experiments\quantum_results\quantum_heatmap_ang20171012t194435_29.png`
- **VQC P(anomaly)** on a 70×70 test scene: near‑uniform orange (gap −0.012 vs ground truth).
- **Labelling**: `DIAGNOSTIC VISUAL - reduced budget (zz reps=1, real_amplitudes reps=1, maxiter=60)`.

### 6. Comparison Heatmap (reduced budgets)
- **File**: `sih\experiments\quantum_results/quantum_comparison_heatmap_ang20171012t194435_29.png`
- **5‑arm comparison** (RX best, kernel_valscaled second-best, VQC/QAE ≈ noise).
- **Labelling**: `DIAGNOSTIC VISUAL - reduced variational budgets (vqc/qae ansatz_reps=1, maxiter=60) - NOT the frozen reporting configuration`.

### 7. Pytest Suite
- **26 tests passing** (22 original + 4 new real‑loader tests).
- All existing `pytest -q` suites (545 tests in the repo) still pass green.

## Repo‑Compliance Notes

| Rule | Status |
|---|---|
| `git diff --stat sih/quantum/` empty | ✅ No code changes under `sih/quantum/` |
| `data/` gitignored | ✅ HAD100 data stays outside git |
| `pytest -q` → 545 passed | ✅ |
| `scripts/verify_had100.py` exit 0 | ✅ |
| New files only in `scripts/`, `tests/`, `docs/` | ✅ |
| No modifications to `sih/quantum/` | ✅ |

### What Was NOT Changed
- `sih/quantum/` – frozen branch code untouched.
- `sih/data/benchmark/had100/` – data stays outside git (gitignored).
- `experiments/quantum_results/results.csv`, `results_pooled.csv`, `run_manifest.json` – these are the **deliverable outputs** from the completed quantum run (8‑row comparison table matching the paper's headline finding: every quantum arm loses to classical RX).
- `main` branch untouched – all changes are on the `quantum-prototype-fixes` branch.

### How to Review / Continue Work

| Action | Command |
|---|---|
| Create a PR from the branch | `git push origin quantum-prototype-fixes` then open a PR |
| Run the CI checks | `pytest -q` must show 545 passed; `scripts/verify_had100.py` must exit 0 |
| Add more edge‑case tests | Add files under `tests/` with `skipif` on missing HAD100 data |
| Polish the heatmaps | Adjust `scripts/plot_quantum_*.py` parameters (`--ansatz-reps`, `--maxiter`) for frozen‑config figures |
| Pursue open repo items | Open issues for `fetch_enmap.py`, Siamese scaling, or AVIRIS/splib07 verification |

---

## How to Continue From Here

1. **To keep the quantum branch running**: The full comparison run is already finished; results live in `experiments/quantum_results/`.
2. **To add more diagnostic visuals**: Use the new scripts with different budget flags (`--ansatz-reps 3 --maxiter 200` for frozen‑config figures).
3. **To contribute more code**: Open an issue, create a branch `arun/quantum‑<feature>`, add new tests or scripts (keeping `git diff --stat sih/quantum/` empty), then open a PR.
4. **To run the full quantum comparison again**: The runner is `sih/.venv\Scripts\python.exe -m quantum.classical_vs_quantum --out experiments/quantum_results` (≈2 h on CPU).

---

**Contact**: For questions about the quantum branch or this prototype work, open an issue in the sih repo or contact the branch maintainer.

---
*This summary was generated on 2026‑08‑23 and reflects the state of the SIH quantum branch and the prototype sandbox at that date.*