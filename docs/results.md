# Results — what works, what doesn't, and where the evidence is

One page, honestly. Every number here comes from a file in `experiments/`, and every
row links to the artifact that produced it so you can check it yourself.

**Read this before quoting any number from this project.** Several results are
negative, several are qualified, and a few look better than they are if you read
them without the caveat attached. The caveats are not fine print — in this project
they are the finding.

Last updated 2026-08-23. Suite: **545 tests passing**.

---

## 1. What works

### Classical anomaly detection — the core of the system

Scene-macro pooled ROC-AUC (primary metric). Regenerate with
`scripts/run_benchmark.py --datasets abu,hydice,had100`; raw numbers in
`experiments/rx_vs_ae/results_pooled.csv`.

| dataset | scenes | best detector | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| HAD100 | 94 | `kernel_rx` | **0.9713** | 0.6529 |
| ABU | 13 | `crd` | **0.9674** | 0.4299 |
| HYDICE | 1 | `local_rx` | **0.9976** | 0.5237 |

No single detector wins everywhere — `kernel_rx` leads on HAD100 and is *fourth* on
ABU. That is why the detector is a config choice (§4.1) rather than a hardcode.

**Read the PR-AUC column too.** ROC-AUC near 0.97 sounds close to solved; PR-AUC of
0.43 on ABU says otherwise. At the ~0.4–2% anomaly prevalence these scenes carry,
PR-AUC is the honest number and ROC-AUC flatters.

### Georeferencing — verified against the real world

Phase 5 **Level 2** (EnMAP, 30 m) is **closed**: ROI polygons land within 2 pixels of
their true position, checked in QGIS against an OpenStreetMap basemap. Corner
round-trip error is exactly 0.0 m; centroid error ≤0.707 GSD, which is the half-pixel
diagonal — i.e. pixel-centre quantisation, not drift. Evidence:
`experiments/phase5_level2/level2_metrics.json`, project
`qgis/projects/phase5_level2_verify.qgz`.

### The cascade saves what it claims to save

The demo reports a **99.3% pixel saving** — segmentation runs on 27 ROI pixels of a
4096-pixel scene — at stage-1 recall 1.0. `experiments/demo/demo_summary.json`.
(Note this is the *demo* scene. The edge branch's separate cascade criterion fails;
see §2.2.)

### Offline operation — proven, not asserted

`pipeline/demo.py --assert-offline` replaces `socket.socket` for the whole inference
stage, so any connection attempt raises rather than being counted afterwards.

---

## 2. What doesn't work — four documented failures

These are real, they are ours, and they are reported rather than buried. Each names
the criterion it missed.

### 2.1 Fusion does not beat its own best input (plan.md D25)

§3A.9 requires fused AUC ≥ best single component on ≥10 of 13 scenes.
**Measured: 5 of 8 held-out scenes.** Fusion beats `local_rx` alone by **+0.0018**
macro AUC, which is noise.

The grid search the plan prescribed was run — 728 weightings, selected on 5 tuning
scenes, evaluated on 8 scenes never touched during selection — and **it did not
rescue the result**. The optimizer assigned `ace` a weight of **exactly 0.0**, and so
did the next four best weightings.

*Why:* ACE's target signature is bootstrapped from the top 0.1% of pixels *by the base
detector*. When those pixels aren't a coherent material, the "signature" averages
unrelated spectra and ACE measures nothing. On `abu-urban-4` it scores 0.5029 —
indistinguishable from chance.

On HAD100 the 4-component `fused` scores **0.9359** against `kernel_rx`'s **0.9713**.
Evidence: `experiments/rx_vs_ae/fusion_weights.json`.

**Do not claim fusion beats its components.** The defensible claim is that it is
*comparable* to the best single detector while requiring no per-scene detector choice.

### 2.2 The ROI cascade misses its pixel-fraction target (plan.md D31)

Criterion: <10% of scene pixels at ≥0.98 recall. **Recall met at 0.9861; pixel
fraction 3.69× the scene — 37× over target.**

*Why:* ABU-Airport-1 is 100×100 and the patch is 64, so the full grid is 2×2 windows.
Calibrating to 0.98 recall flags 87.8% of background pixels, and each flagged
component expands to its own 64×64 window. **On a scene this small, scattered false
positives cost more windows than running the full grid.** This is scene-size versus
patch-size, not a defect in the cascade.

*Not retestable on a larger scene:* `roi_vs_full_comparison` requires ground truth by
design, and EnMAP has none. Evidence: `docs/edge.md`,
`experiments/edge_benchmarks/*.jsonl`.

### 2.3 Quantization reaches 1.82×, not ~12× (plan.md D31)

Literature target ~12× size reduction. **Measured 1.82×** (7,770,837 → 4,277,782
bytes). **INT8 was never applied** — the run is FP16-only.

That was a design choice, not an oversight: uniform INT8 destroys covariance
conditioning silently, and no INT8-safe subgraph of this UNet has been identified.
The target is stated as not hit, not approached by another route.

### 2.4 Every quantum arm loses to classical RX (plan.md D28, D29)

Scene-macro ROC-AUC on held-out flightlines, identical 8 features, identical split:

| arm | supervision | ROC-AUC |
|---|---|---|
| `classical_svc` | supervised | **0.8130** |
| `rx_8feat` | unsupervised | **0.7316** |
| `quantum_kernel` (val-tuned) | unsupervised | 0.6815 |
| `vqc` | supervised | 0.4736 |
| `qae` | unsupervised | 0.4186 |
| `quantum_kernel` (as specified) | unsupervised | **0.4017** — below chance |

The specified quantum kernel scores **below chance** on held-out flightlines. It
scores 0.98 on training data and passes 11 unit tests including a sign-orientation
check — because those tests assert against data the model has already seen.

*Why:* the fidelity kernel **exponentially concentrates**. Changing one constant — the
angle encoding scale, `[0,π]` → `[0,π/2]` — moves test AUC from 0.378 to 0.693. The
cure is the encoding, not the regularizer.

The one arm above RX is *supervised*, which is a supervision effect, not a quantum
one. Evidence: `experiments/quantum_results/`, write-up `docs/experiments.md` §4.

**No quantum-advantage claim is made anywhere.** The comparison table is the
deliverable.

---

## 3. Results that are real but heavily qualified

### 3.1 Change detection numbers are all SYNTHETIC-PAIRS (plan.md D30)

There is **no real bi-temporal hyperspectral pair on disk**. All 20 EnMAP scenes were
checked: none overlap spatially at different dates. So t2 is *constructed* from a real
scene by known misregistration + implanted targets + a +12% illumination gain.

| arm | AUC | pseudo-change rate |
|---|---|---|
| SAM + physics fusion | **0.7651** | **0.6119** |
| classical difference | 0.6179 | 0.7797 |
| SiameseChangeNet | 0.5550 | 0.9242 |

Physics fusion wins on both. The learned arm at a modest budget is worst of the three.
**No number here may be quoted without the construction attached.**

### 3.2 The Level 3 case study cannot separate construction from season (plan.md D34)

Sentinel-2 over Noida International Airport (Jewar), bracketing the dated 25 Nov 2021
groundbreaking. The pipeline runs and the outputs are correctly dated. But:

- **The reported ROI areas are not a measure of how much changed.** Selection is a
  fixed top-5% percentile, so a comparable area is selected *whether or not anything
  changed*.
- **The NDVI progression cannot be separated from season.** Four single-year snapshots
  cannot distinguish land-use conversion from an agricultural cycle.
- **`TemporalBaseline` is unusable at n=2 epochs** — MAD over two points is `|a−b|/2`;
  68.1% of pixels exceed z>3.
- **The cloud mask was never exercised** — the AOI is clear on all four dates.

The dated groundbreaking *corroborates an observation*; it is not something the
imagery proves.

### 3.3 F1 is uninformative here, by construction

At ~0.4% prevalence under recall-first calibration, every arm lands at F1 ≈ 0.019–0.044
with precision ≈ prevalence. F1 discriminates nothing at this operating point. Use
ROC-AUC and PR-AUC.

---

## 4. What is simulated, and what does not exist

| claim | status |
|---|---|
| Edge latency / memory figures | **SIMULATED** on an x86 laptop. Not a Raspberry Pi. |
| Power / energy consumption | **Does not exist.** No wattmeter. No number may be quoted. |
| Quantum hardware results | **Do not exist.** Simulator only; no IBM Quantum account. |
| Quantum wall-clock | `AerSimulator` CPU time. Carries no QPU implication. |

At 8 qubits the simulator reproduces the "quantum" kernel exactly and **180× faster**,
which is itself part of the finding.

---

## 5. The pattern worth knowing

Five separate defects in this project shared one shape: **a check that passed for the
wrong reason.**

| # | the check | why it passed anyway |
|---|---|---|
| D22.2 | regularizer test | tested on data the model had seen |
| D28 | quantum kernel, 11 tests incl. a sign check | asserted against training data; scored 0.378 held out |
| D33 | GeoJSON timestamp well-formed | every source had `acquired=None`, so the wrong value *was* the only value |
| D35 | UI memory guard, own unit tests green | used a guessed constant; allowed the exact scene the kernel killed |
| D25 | fusion defaults "reasonable" | never compared against its own best component |

If you add a test here, ask what it would take for it to pass while the thing it
guards is broken.

---

## 6. Where to look next

| you want | go to |
|---|---|
| the full decision record | `plan.md` — D1 through D35, each with evidence |
| dataset provenance and verification tier | `docs/datasets.md` |
| the quantum comparison in full | `docs/experiments.md` |
| edge measurements and their limits | `docs/edge.md` |
| Level 2 / Level 3 validation | `docs/validation.md` |
| what's built and what isn't | `docs/buildable_now.md` |
