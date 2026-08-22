# Branch 3E — Quantum: handoff brief to finish the last ~15%

> **COMPLETED 2026-08-22.** The run finished (53 min, 8/8 rows, zero failures) and §4 is
> written. Results: `experiments/quantum_results/`, write-up: `docs/experiments.md` §4,
> dated decision: `plan.md` D29. Headline, per the brief's own rule: every quantum arm lost
> to `rx_8feat`, and it is reported plainly. This brief is retained for provenance only.

**All code is written, tested and committed.** What remains is one long run and one document.
No new modules are needed. If you find yourself writing a new module, stop — something has
been misread.

Spec: `plan.md` §6.5. **The decisions that override it are `plan.md` D27 and D28 — read both
in full before touching anything.** They are corrections to §6.5, not commentary.

## State on arrival

| item | state |
|---|---|
| `quantum/qiskit_basics.py` · `feature_map.py` · `data.py` | built, 22 tests |
| `quantum/vqc_encoder.py` | built, 13 tests |
| `quantum/quantum_autoencoder.py` | built, 14 tests |
| `quantum/quantum_kernel.py` | built, 11 tests |
| `quantum/classical_baselines.py` · `classical_vs_quantum.py` | built, 35 tests |
| `docs/experiments.md` §1–§3 (novelty, method, caveats) | **written — do not rewrite** |
| `docs/experiments.md` §4 Results | **placeholder — this is your deliverable** |
| the full comparison run | **not started — this is your other deliverable** |

Environment: `.venv/bin/python` (3.12.13). `qiskit 2.5.2`, `qiskit-aer 0.17.2`,
`qiskit-machine-learning 0.9.1`, all installed and verified to compose.

## Task 1 — run the comparison

```bash
cd /home/mayaskara/projects/sih
.venv/bin/python -m quantum.classical_vs_quantum --smoke        # ~25 s, sanity first
.venv/bin/python -m quantum.classical_vs_quantum --out experiments/quantum_results
```

**Run the second command in the background and let it finish.** Expect **~2 hours**, dominated
by two unswept fits: VQC ~35 min, QAE ~15 min (both measured, not estimated). Everything else is
seconds. Do not reduce `maxiter` to make it faster — those values are the reported configuration.

Writes `results.csv`, `results_pooled.csv`, `run_manifest.json` to `experiments/quantum_results/`.

**Do not modify any file under `quantum/` to make the run work.** If the run fails, that is a
finding: report the traceback rather than patching around it. Six defects in this project were
found exactly this way and all six were invisible to a green test suite.

### 8 rows you should get back

7 arms; the quantum kernel is scored twice, at the specified angle scale and at a val-selected one.

| row_id | supervision | swept |
|---|---|---|
| `classical_ae` · `classical_ocsvm` · `classical_svc` · `rx_8feat` | mixed | yes |
| `vqc` | supervised | **no** |
| `qae` | unsupervised | **no** |
| `quantum_kernel_spec` (angle scale 1.0, `zz reps=2`, exactly as §3E.5 specifies) | unsupervised | no |
| `quantum_kernel_valscale` (val-selected angle scale) | unsupervised | yes |

## Task 2 — write `docs/experiments.md` §4

Sections 1–3 are finished and binding: the dated literature search, the method, and an explicit
"what these numbers do NOT mean". **Append §4 only.** Do not soften §1's narrowed novelty claim
or §3's disclaimers to fit the results.

§4 must contain:

1. **The table**, scene-macro pooled, primary. Columns: `row_id`, `supervision`, `swept`,
   `roc_auc_MACRO`, `pr_auc_MACRO`, `n_qubits`, transpiled `depth`, `fit_wall_clock_s`.
   Pixel-micro goes in a secondary table, **explicitly labelled** — an unlabelled pooled figure
   is banned in this repo.
2. **The kernel-concentration diagnostic** from the manifest, beside D28's measured values, with
   a sentence on whether the run reproduced them.
3. **Whether `quantum_kernel_valscale` actually differs from `quantum_kernel_spec`.** On the real
   split, val preferred angle scale 0.5 (val 0.7044 vs 0.6872). If the run selects 1.0 instead,
   the two rows are identical and **you must say so** rather than presenting one result twice.
4. **A statement of who was tuned.** `vqc` and `qae` are `swept=False` because a fit costs 35 and
   15 minutes. Say it in the prose, not only in a column. A swept arm beside an unswept one is
   not a like-for-like comparison.
5. **Two known non-results, stated as such:**
   - **F1 will not discriminate.** At ~0.4 % natural prevalence with a threshold calibrated for
     0.98 recall on a far more balanced val split, every arm lands near F1 ≈ 0.017 with FP rate
     ≈ 1.0. That is recall-first calibration under prevalence mismatch, not a bug. ROC-AUC and
     PR-AUC carry the comparison; say why F1 is present and uninformative.
   - **Shot noise is negligible.** Measured at the real test-split size: 1024 shots costs 0.0001
     ROC-AUC against exact probabilities despite 173/600 duplicate score values; even 256 shots
     costs 0.0024. A closed question, worth one line so nobody reopens it.

### Hard reporting rules — these are not stylistic

- **`plan.md` §13 rule 4:** scoped novelty ≠ quantum advantage. The word "advantage" appears only
  to disclaim it. **The table is the deliverable.**
- **§13 rule 6:** HAD100 only. No ABU or HYDICE quantum number may be produced, implied or
  interpolated (D27.6 — it is forced by `harmonize` needing wavelengths, not chosen).
- **D27.4:** wall-clock is `AerSimulator` CPU time and carries no QPU implication. Label the
  column so. Depth is transpiled to a named basis or it is not comparable.
- **O1:** no IBM Quantum account. §3E.7 does not happen. **No hardware number anywhere.**
- **D27.9 / D28:** do not compare these numbers to `experiments/rx_vs_ae/results_pooled.csv`'s
  HAD100 row. Those detectors are unsupervised and per-scene and never generalize across
  flightlines; these arms train and are tested on held-out flights. Two rows labelled "HAD100",
  two different tasks. If you reference that file at all, it is to say why it is not comparable.
- If a quantum arm loses to `rx_8feat`, **report that plainly.** It is a legitimate §3E.6 result
  and the whole point of building the comparison this way.

## Traps

1. **The prevalence trap (D27.7).** Metrics come from `score_scene_natural`, which returns
   `(scores, labels, sample_weight)`. The runner already passes `sample_weight`. **Do not
   recompute any metric without it** — unweighted reads PR-AUC 0.87 where the truth is 0.34.
2. **Never select on test.** Every sweep reads val. If you find yourself comparing test numbers to
   choose anything, stop.
3. **A passing test suite proves little here.** The kernel arm passes 11 tests including a
   sign-orientation check, and scored 0.378 — *below chance* — on held-out flightlines. Its tests
   asserted against the training split.
4. **Do not "fix" a bad number.** D28 is a finding, not a bug.

## Definition of done — what will be verified

- [ ] `experiments/quantum_results/{results.csv,results_pooled.csv,run_manifest.json}` exist, 8 rows.
- [ ] Every row has a non-empty `supervision` and a `swept` flag.
- [ ] Manifest carries `git_sha`, package versions, and `kernel_concentration`.
- [ ] `docs/experiments.md` §4 written; §1–§3 unchanged (`git diff` will be checked).
- [ ] Scene-macro labelled primary; any pixel-micro figure labelled.
- [ ] No ABU/HYDICE number. No hardware number. No "advantage" claim.
- [ ] `.venv/bin/python -m pytest -q` still green (~380 tests).
- [ ] `.venv/bin/python scripts/verify_had100.py` and `scripts/verify_benchmarks.py` both exit 0.
- [ ] Nothing under `quantum/` modified — `git diff --stat quantum/` empty.

Report the 8-row pooled table in your summary, plus anything that contradicted this brief.
