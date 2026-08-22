# PLAN.md — Software Execution Plan
### AI-Based Hyperspectral Anomaly Detection & Geospatial Semantic Mapping System

**Companion to** `roadmap.md` (strategy, phases, guardrails) and `blueprint_upgrades_changelog.md` (research justification).
**This document is** the agent-executable software specification: every file, every signature, every acceptance test.

**Status:** v1.0 — contracts frozen, ready to execute.
**Sequencing:** by dependency only. No calendar dates — there is no time constraint on this project.

---

## 0. Scope

### 0.1 In scope
Every Python module in the repository: preprocessing, anomaly detection, segmentation, change detection, geospatial vectorization, edge/streaming/quantization code, the quantum research branch, the profiling harness, and the test suite.

### 0.2 Explicitly excluded — prerequisites, not tasks
These are things a human must do. No task in this plan waits on them except where marked **CONDITIONAL**.

| Item | Status | Blocks |
|---|---|---|
| Raspberry Pi 5 (8GB) procurement | **Not owned** | Phase 6 real-hardware measurement |
| FPGA / NPU dev board | **Not owned** | Pi-vs-accelerator comparison arm (Roadmap 3D) |
| ESP32 telemetry node | **Not owned** | Optional; out of the AI pipeline entirely (Roadmap §1.4) |
| QGIS desktop install | Not installed (`which qgis` → not found) | Phase 2 exit criterion, Phase 7 demo |
| IBM Quantum account | **Not held** | 3E real-QPU demonstration run |
| Copernicus Data Space account | **Held** | — |
| EnMAP / AVIRIS access | **Held** | — |
| PRISMA access | **Not held** — needs an ASI proposal with lead time (**O7**) | Nothing. Level 2 proceeds on EnMAP + AVIRIS; PRISMA is a bonus arm, not a dependency. |

### 0.3 The no-hardware consequence — read this before writing any `edge/` code
There is no Raspberry Pi. Therefore:

1. All `edge/` code is written **Pi-ready** (ARM64-compatible, no CUDA at inference, ONNX Runtime CPU EP, strip-streaming memory ceiling enforced in code).
2. All Phase 6 numbers come from `edge/constrained_sim.py` — a `taskset` + cgroup v2 harness that pins the pipeline to 4 cores and caps RSS at 8 GB to approximate a Pi 5.
3. **Every simulated number is labelled `SIMULATED` in its filename, its JSON payload, and every table it appears in.** A simulated latency figure must never be quoted as measured. This is a hard rule — Roadmap §1.8 requires edge value be *measured*, and a constrained x86 laptop is not a Pi 5. It is a regression guard and a relative comparison, nothing more.

---

## 1. Resolved Decisions — deltas from `roadmap.md`

These were open or under-specified in the roadmap. They are now decided. Anything still open is in §14.

### D1 — Python is pinned to **3.12.13**. The binding constraint is `fiona`.
Verified empirically, not assumed:

```
Python 3.14.7 : uv pip install FAILS
                → fiona==1.10.1 has no cp314 wheel
                → falls back to sdist → needs system gdal-config → build error
Python 3.12.13: uv pip install SUCCEEDS, all imports clean
                rasterio 1.5.1 (GDAL 3.12.4) · geopandas 1.1.4 · fiona 1.10.1
                numpy 2.5.2 · scipy 1.18.0 · scikit-learn 1.9.0 · torch 2.13.0
                onnx 1.22.0 · onnxruntime 1.29.0 · opencv 5.0.0
                qiskit 2.5.2 · qiskit-aer 0.17.2 · qiskit-machine-learning 0.9.1
```

Write the *reason* down, not just the pin: **fiona is the only blocker**, and geopandas 1.x uses `pyogrio` by default anyway. If someone revisits this pin later, the correct question is "does fiona have a cp3XX wheel yet, and do we still need fiona at all" — not "is 3.14 supported."

### D2 — Indian Pines has no CRS and no affine transform. This is a landmine in Phase 2.
> **Amended by D11.5.** All 616 HAD100 ENVI scenes carry real `map info` + UTM/WGS-84 CRS strings (2.3 m NG, 17 m Classic). The synthetic-affine workaround below applies to **Indian Pines only**, and the first *real* georeference check moves forward from Phase 5 Level 2 into 3A, where genuine coordinates are available.

Roadmap §2 names Indian Pines as the Week-1 dev dataset. Roadmap Phase 2 step 6 requires opening the GeoJSON in QGIS and confirming polygons land in the right place. **These are incompatible as written** — Indian Pines is a `.mat` cube with no georeferencing at all, so there is no "right place."

**Decision:** synthesise a documented-fake affine.
- Indian Pines is a real AVIRIS scene over NW Indiana, ~20 m GSD.
- `preprocessing/raster_loader.py` assigns `CRS = EPSG:32616` (UTM 16N) and an affine anchored at a fixed origin, at 20 m pixel size.
- The GeoTIFF carries the tag `SYNTHETIC_GEOREF=true` and every export from it carries `"georef": "synthetic"`.
- **Phase 2 verifies the transform *plumbing* is correct — that polygon pixel coords round-trip through the affine and land in the right place relative to the raster.** It does not verify real-world geographic accuracy.
- Real georeferencing is verified for the first time in **3A.1**, on HAD100's genuine UTM/WGS-84 headers (D11.5), and independently re-verified in **Phase 5 Level 2** on an EnMAP/AVIRIS GeoTIFF. Geospatial accuracy metrics (Roadmap §8) are reported from those, never from Indian Pines.

No branch owner may assume Indian Pines has real georeferencing.

### D3 — Score normalization is now fully specified (closes a Roadmap §2 gap).
Roadmap §2 says score rasters are "normalized 0–1" without saying how. Phase 4 fuses anomaly + change + confidence; unspecified normalization means fusing incomparable scales. Two normalizations exist, used for different things:

**For storage / thresholding — percentile clip (invertible):**
```
v_lo = percentile(score[valid], 1.0)
v_hi = percentile(score[valid], 99.9)
norm = clip((score - v_lo) / (v_hi - v_lo), 0, 1)
```
`p_hi = 99.9`, not 98 — for an anomaly score the upper tail *is* the signal; clipping at 98 destroys it. 99.9 removes single-pixel sensor spikes and nothing else. `v_lo`, `v_hi`, `p_lo`, `p_hi` are written into the GeoTIFF tags so the transform is invertible and the raw score is recoverable.

**For fusion — rank normalization (scale-free):**
```
norm = (rankdata(score[valid], method='average') - 1) / (n_valid - 1)
```
Fusing an RX Mahalanobis distance, an ACE cosine ratio and an NDVI index by weighted sum is only meaningful if they are on a common scale. Rank normalization is the only transform that guarantees that. It is **within-scene only** — it is not comparable across scenes, so it is never stored as a product, only used inside `anomaly/fusion.py`.

Both live in `anomaly/scoring.py`. Nothing else may define a normalization.

### D4 — `confidence` is now defined (closes a Roadmap §2 gap).
The GeoJSON schema requires a `confidence` field; nothing in the roadmap said how to compute it. It is a weighted mean over *available* components, with weights renormalized when a component is missing:

| Component | Symbol | Weight | Source |
|---|---|---|---|
| Mean normalized anomaly score inside polygon | `c_anom` | 0.40 | 3A fused score |
| Mean normalized change score inside polygon | `c_change` | 0.20 | 3C change score |
| Mean segmentation probability inside polygon | `c_seg` | 0.25 | 3B U-Net sigmoid |
| Clear-sky fraction (`1 - cloud_shadow_fraction`) | `c_clear` | 0.10 | `preprocessing/cloud_mask.py` |
| Shape plausibility score | `c_shape` | 0.05 | `segmentation/postfilter.py` |

```
confidence = Σ(wᵢ · cᵢ over available i) / Σ(wᵢ over available i)
```

In Phase 2 only `c_anom` exists, so `confidence == c_anom`. The set of components actually used is recorded per-feature in `confidence_components` so a value is never ambiguous about what went into it.

### D5 — Approved amendment to the Section 2 GeoJSON contract: ROI provenance.
Roadmap §9.11 makes contract changes a team-sync event, so this is written as a decision with its rationale, not a drive-by addition.

**Rationale (from the target-profile decision, §D6):** running two target profiles means the ROI post-filter has different thresholds depending on which branch produced the candidate. The post-filter runs at **Stage 4 (ROI extraction)**, not at Stage 6 (vectorization). So the profile tag must be attached where the candidate is *born* and must survive Phase 4 fusion. If it is inferred later, it will be inferred wrong — which is exactly the silent integration mismatch the frozen contracts exist to prevent.

**Added fields, set at ROI creation, never inferred:**

| Field | Type | Values |
|---|---|---|
| `source_branch` | str | `"anomaly"` \| `"change"` \| `"fused"` |
| `target_profile` | str | `"object"` \| `"landcover"` |
| `roi_id` | str | `"{scene_id}:{branch}:{index:04d}"` |
| `linked_roi_ids` | list[str] | cross-profile spatial overlaps, references only |
| `confidence_components` | list[str] | which of D4's components were available |
| `georef` | str | `"real"` \| `"synthetic"` (see D2) |

**Fusion rule for conflicting parents — decided:**
> Phase-4 ROI-level fusion is permitted **only between parents carrying the same `target_profile`**. The result gets `source_branch="fused"` and inherits that shared profile. Two ROIs of *different* profiles that overlap spatially are **not merged** — each survives independently and each records the other in `linked_roi_ids`.

The alternative (merge, and take the higher-scoring parent's profile) is also defensible, but destroys provenance: a merged ROI would silently carry post-filter thresholds it was never screened against. Non-merge preserves the audit trail. Silence on this point was not an option.

### D6 — Target profile: **both `object` and `landcover`**, config-selected.
Two profiles maintained in `configs/target_profile.yaml`, selected by flag, driving post-filter thresholds and the spectral index set.

```yaml
object:
  postfilter: {min_area_px: 4, max_area_px: 2000, min_solidity: 0.15, max_elongation: 8.0}
  indices:    [ndbi, iron_oxide_ratio, clay_ratio, brightness]
  validated_on: [abu, hydice_urban_anomaly, had100]
landcover:
  postfilter: {min_area_px: 50, max_area_px: null, min_solidity: 0.05, max_elongation: 20.0}
  indices:    [ndvi, ndwi, nbr, bsi]
  validated_on: [sentinel2_timeseries]
```

**Routing:** the anomaly branch (3A) defaults to `object`; the change-detection branch (3C) defaults to `landcover`. The default is a config default, not a hardcode — either branch can be run under either profile, and whichever profile was active is stamped onto every ROI it produces per D5.

### D7 — Segmentation training data: extended synthetic + real-only scoring.
Full spec in **§6.2**. The governing rules:
- **Train on synthetic. Score on real. Never both on the same data.** Synthetic data provides training volume; every reported metric comes from real ground truth.
- Two training branches are built and compared as separate rows, never merged into one number: (a) **implanted** — real target spectra mixed into real backgrounds; (b) **pretext** — self-supervised, zero real target spectra.
- **Leakage is controlled at the spectrum level, not just the scene level.** Target spectra harvested from ABU/HYDICE ground truth carry that dataset's identity with them, so a model trained on ABU-derived spectra scored on ABU is leaking even though no ABU *scene* was in its training set. The leave-one-dataset-out matrix in §6.2 is what enforces this; a scene-ID check alone would pass while the leak is live.
- Background pool comes from **AVIRIS-NG and EnMAP L2A only**. Not Sentinel-2 — it is multispectral (~13 bands) and incompatible with the hyperspectral band count the rest of the pipeline assumes (Roadmap §9.1).

### D8 — GPU budget: GTX 1650, 4 GB VRAM. This constrains 3A and 3B.
| Model | Config that fits | Note |
|---|---|---|
| Lightweight U-Net | 64×64 patch, C=30, batch 16, AMP fp16 | comfortable |
| Compact transformer segmenter (SegFormer-B0 class) | 64×64 patch, C=30, batch 8, AMP + grad-accum 2 | tight but fits |
| "Stronger deep detector" (3A) | **scoped down** to a compact spectral-spatial transformer on 64×64 patches | a full graph-transformer detector as in the changelog **will OOM at 4 GB** |

The deep-detector arm is scoped down deliberately, and the report must say so. Do not claim a graph-transformer result that was not run. If a bigger GPU becomes available, `anomaly/deep_detector.py` is the only file that changes — and **D12** governs what happens if that bigger GPU is a free-tier cloud notebook rather than owned hardware.

### D9 — Band harmonization is mandatory and is a new preprocessing module.
Training on AVIRIS-NG (425 bands) and scoring on ABU (**seven distinct band counts** across 13 scenes: 205/204/193/191/188 AVIRIS and 102 ROSIS — D13.2) and HYDICE (**175**, not 162 — D13.3) means band counts differ per scene, and by more than the earlier draft assumed. A segmentation model with a fixed input channel count cannot consume them directly. **All model-facing code consumes a canonical grid**, produced by `preprocessing/harmonize.py`: interpolate to 400–2500 nm at 10 nm → **211** bands → drop water-absorption windows (1350–1450 and 1800–1950 nm, **endpoints inclusive**) → **184** bands → PCA/kPCA to `C=30`. Classical detectors (RX family, CRD) run on native bands; only learned models see the canonical grid.
**Interpolation is not the first step — sorting is.** AVIRIS-Classic wavelength arrays are non-monotonic in 262 of 262 files (D11.4), and `np.interp` neither requires nor checks that `xp` ascends. Sort, collapse duplicates, assert strictly increasing, *then* interpolate.

**Band arithmetic — verified by execution, not asserted.** The earlier draft of this
document said `189`. That number is unreachable under *any* endpoint convention, and the
reason it survived is that the convention itself was never written down. Both are now fixed.
Run on the pinned interpreter (Python 3.12.13, numpy 2.5.2):

```
len(np.arange(400, 2501, 10, dtype=np.float32))            = 211
all canonical wavelengths exactly representable in float32 = True   (400.0 … 2500.0)

dropped bands per window, endpoints inclusive: [1350,1450] = 11 ·  [1800,1950] = 16

  inclusive   lo <= w <= hi :  dropped 27  ->  retained 184   <-- ADOPTED
  half-open   lo <= w <  hi :  dropped 25  ->  retained 186
  exclusive   lo <  w <  hi :  dropped 23  ->  retained 188
```

**Adopted convention: endpoints inclusive → `RETAINED_BANDS = 184`.** Inclusive is the right
default because 1350 and 1450 nm *are* inside the absorption feature, not clean shoulders of
it; keeping them to reach a rounder number would be keeping the noisiest bands in the set.

Because every canonical wavelength is exactly representable in float32, `>=` / `<=` are safe
here and no tolerance is needed. That is a property of this grid, not a general one — if the
step ever changes to a non-integer nm spacing, the comparison must move to integer
tenths-of-nm or an explicit tolerance, or the endpoint bands will drop in or out at random.

### D10 — New modules added beyond Roadmap §3.
Additions, with rationale. None change a data contract; `core/contracts.py` exists specifically to *enforce* them.

| New file | Why |
|---|---|
| `core/contracts.py` | Single source of truth for `SceneMeta`, `ROIRecord`, `ScoreRaster` dataclasses + validators. Every branch imports from here. This is the mechanism that makes "frozen contracts" real rather than a document. |
| `preprocessing/harmonize.py` | D9 |
| `preprocessing/bands.py` | wavelength→index lookup, spectral index computation |
| `anomaly/streaming_rx.py` | Roadmap 3A calls for a streaming RX refactor but §3 lists no file for it |
| `segmentation/synth.py` | D7 synthetic generation |
| `segmentation/datasets.py` | torch Dataset wrappers over real + synthetic |
| `edge/constrained_sim.py` | §0.3 — no Pi |
| `geospatial/roi_fusion.py` | Phase 4 ROI-level fusion; implements the D5 same-profile-only merge rule. Roadmap §4 calls for ROI-level fusion but §3 lists no file for it |
| `pipeline/run_pipeline.py` | the Phase 2 spine; Roadmap §3 lists branch folders but no orchestrator |
| `configs/` | `target_profile.yaml`, `pipeline.yaml`, `paths.yaml` |
| `tests/` | contract + numerical tests |
| `scripts/` | dataset fetchers |

### D11 — HAD100 as it actually ships. Every number below was read off the downloaded archive.

The earlier draft carried "100 test scenes 64×64 · 260 AVIRIS-NG / 262 AVIRIS-Classic
background" — taken from the project page. The archive was downloaded
(`HAD100.zip`, 3 754 846 290 bytes,
sha256 `6f91035543b7bc7806ebc555cd5411d320f2810f0af8dfcf20fe4f331227c19f`,
`unzip -t` clean) and every ENVI header parsed. Raw counts and the page's counts are
**not the same numbers**, because the page describes the dataset *after* the repo's
`main.py` unpacking step and the archive contains the *inputs* to it.

```
data/aviris_ng_target   94 scenes   425 bands  BIL  float32   <- test SOURCE
data/aviris_ng_normal  260 scenes   425 bands  BIL  float32   <- NG background
data/aviris_normal     262 scenes   224 bands  BIL+BIP  int16+float32
gt/aviris_ng_gt         94 .mat masks
main.py                                                       <- the unpacker
```

**Counts — raw vs unpacked. Both are real; they answer different questions.**

| | raw files in archive | after `main.py` | why they differ |
|---|---|---|---|
| Test | **94** | **100** patches | 18 scenes are in `crop_dict`; 6 of those yield 2 crops each → 94 + 6 = 100 |
| Background NG | **260** | **1 040** patches | `for id in range(4)` — four corner 64×64 crops per scene |
| Background Classic | **262** | **1 048** patches | same |
| **Background total** | **522** | **2 088** patches | |

260 / 262 / 522 are confirmed exactly. "100 test scenes" is confirmed **as an unpacked-patch
count**, and is wrong as a file count. Say which one you mean every time.

**The background pool is 4× larger than the plan assumed** — 2 088 usable 64×64 patches, not
522 scenes. That is the single biggest input to 3B's training volume and it was undercounted.

#### D11.1 Spatial dimensions — the raw scenes are NOT uniformly 64×64

Full distribution, every shape seen, `lines × samples`:

```
aviris_ng_target (94)   64x64:76  70x70:2  75x75:1  80x80:1  86x81:1
                        100x100:11  101x101:1  120x120:1        -> 8 distinct shapes
aviris_ng_normal (260)  81x81:260                               -> uniform
aviris_normal   (262)   66x66:41  71x71:11  71x81:1  76x76:2
                        81x81:205  91x91:1  130x130:1           -> 7 distinct shapes
```

The 18 non-64×64 test scenes are exactly the 18 keys of `crop_dict` — that is what the crop
offsets exist for. Note `86x81` and `71x81` are **non-square**; any code that assumes `H == W`
breaks on them.

#### D11.2 Consequence for `patch=64` — it stays, but for a stated reason

`patch = 64` remains correct **for the unpacked HAD100 patches**, which are uniformly 64×64 by
construction. It was never correct as a claim about HAD100 *scenes*. The three sites that
hardcode it are therefore unchanged in value and changed in justification:

- `segmentation/infer.py` `segment_rois(patch=64)` — correct; ROI windows are patch-aligned crops,
  and the model's input size is a property of the model (D8), not of the dataset.
- `anomaly/local_rx.py` HAD100 note `outer=15, inner=3, n_components=12` — correct for the
  64×64 unpacked patches. **Do not apply it to raw scenes**: at 120×120 the default
  `outer=21` is fine, and 15 needlessly starves the annulus.
- `segmentation/postfilter.py` `max_area_px = min(2000, 0.5 * scene_px)` — already computed
  per scene, so it self-adjusts from 4 096 px (64×64) to 14 400 px (120×120). This is why it
  was written per-scene, and it is the only one of the three that needed no revisiting.

**Canonicalization rule — spatial only. See D11.6, which corrects this for the spectral axis.**
The pipeline takes its **spatial** crops from the repo's own `main.py` — we do not re-implement
the cropping. Reasons: the `crop_dict` offsets
encode where the annotated targets actually are, so a naive top-left 64×64 crop of a 100×100
scene silently drops targets and every recall number afterwards is wrong; and using the
official unpacker is what makes our numbers comparable to published HAD100 results.
A scene smaller than 64 in either axis would need padding — **there are none**; the minimum
observed is 64×64, so no pad path is required. Assert it rather than assume it.

The **spectral** half of `main.py` is a different matter: its `band_select` leaves holes of up
to 276 nm in the canonical grid, and we do not inherit it. See **D11.6**.

#### D11.3 The two background pools have different band counts and CANNOT be stacked

`main.py`'s own `band_select` drops water/noise bands per sensor (we do not use its output as
model input — D11.6 — but the counts below are what make the two pools unstackable either way):

```
aviris_ng : 425 bands -> np.r_[15:109, 118:145, 158:187, 227:274, 328:407] -> 276 bands
aviris    : 224 bands -> np.r_[7:57, 65:79, 85:104, 122:149, 172:224]      -> 162 bands
```

**276 ≠ 162.** Any code that pools all 2 088 background patches into one array in native
bands fails. `preprocessing/harmonize.py` (D9) must run **before** pooling, not after —
harmonize is the join, and it is now a hard ordering constraint in §3A.6, §3B.1 and the §11 DAG,
not an implicit hope. This was previously unstated, which is how "trained on the background
pool (522 scenes)" got written as if the pool were one homogeneous thing.

*(Coincidence worth naming so nobody trips on it: AVIRIS-Classic reduces to 162 bands, the same
count as HYDICE Urban. They are unrelated. Never key any logic on band count as a dataset ID.)*

#### D11.4 AVIRIS-Classic wavelengths are NON-MONOTONIC — this breaks `np.interp`

**In 262 of 262 Classic headers**, the wavelength array descends at three points:

```
... 657.7651, 667.5610, 655.2923, 665.0994 ...      <- back-step at the VNIR/SWIR1 seam
... 1253.480, 1262.964, 1253.373, 1263.346 ...      <- SWIR1/SWIR2 seam
... 1871.227, 1873.184, 1867.664, 1877.725 ...      <- SWIR2/SWIR3 seam
```

That is the AVIRIS four-spectrometer design: adjacent spectrometers overlap in wavelength, so
band index order is not wavelength order. **`np.interp` requires `xp` to be increasing and
does not check** — it returns silent garbage across the seams rather than raising. D9 as
originally written ("linear interpolation onto the canonical grid") would have produced a
harmonized cube that looks fine, passes a shape assertion, and is wrong in three spectral
regions on every Classic scene.

`harmonize()` therefore **must**, before interpolating:
1. `order = np.argsort(wl)`; apply to both `wl` and the band axis;
2. collapse duplicate/near-duplicate wavelengths (mean of the overlapping bands);
3. `assert np.all(np.diff(wl_sorted) > 0)` — fail loudly, never interpolate on unsorted input.

AVIRIS-NG (0 of 354 files non-monotonic) does not need this. Classic does, universally.
This is not a Classic-only defensive check to skip when convenient — it is the difference
between a correct cube and a plausible one.

#### D11.5 Four more things the headers say that the plan assumed otherwise

1. **Every scene is georeferenced.** All 616 ENVI headers carry `map info` and a full
   `coordinate system string` — UTM, WGS-84, real easting/northing, 2.3 m GSD (NG) and 17 m
   (Classic). **This partly retires D2's problem.** D2 says Indian Pines has no CRS so real
   georeferencing cannot be checked until Phase 5 Level 2. HAD100 gives real CRS + affine from
   Phase 3 onward, so the pixel→world→GeoJSON path can be validated against genuine coordinates
   far earlier. D2's synthetic-affine workaround still stands **for Indian Pines only**, and
   §7's Phase 5 Level 2 is no longer the first real-georeference test — it is the first
   *independent* one. Move the first real check into 3A.
2. **Mixed interleave and dtype inside one pool.** `aviris_normal` contains both BIL and BIP
   files, and both int16 and float32. `preprocessing/raster_loader.py` must read the header
   for each file — never assume a pool is homogeneous because its sibling is.
3. **Two different no-data sentinels.** `aviris_ng_target` uses `-9999.0`; `aviris_ng_normal`
   uses `-9999.0` **and** `1e-34`. Masking only `-9999` leaves `1e-34` values in the cube,
   where they behave as near-zero radiance and quietly bias every covariance estimate. Read
   `data ignore value` per file; do not hardcode either.
4. **The four corner crops of one scene overlap heavily.** 81×81 with four 64×64 corner crops
   means ~47 px of overlap per axis, so the 4 patches from one source scene share most of
   their pixels. **Train/val splitting must be at the source-scene level, not the patch
   level** — a patch-level split puts near-duplicate patches on both sides and inflates
   validation. This is the same leakage class as D7 and §3B.5b, arriving through a different
   door, and `test_data_hygiene.py` gains a third check for it.

#### D11.6 — model input harmonizes from the RAW ENVI cubes, NOT from `main.py`'s band-selected output

This corrects an error in D11.2 as first written. D11.2 said "the pipeline consumes the
unpacked dataset." For the **spatial** crops that is right. For the **spectral** axis it is
wrong, and wrong in the silent way.

`band_select` is not a contiguous range — it is five disjoint index slices per sensor, chosen
to drop AVIRIS's noisy and water-saturated bands. Projecting its output onto the canonical
400–2500 nm grid therefore leaves canonical bands with **no source band anywhere near them**.
Measured, not estimated:

```
AVIRIS-NG  post-band_select: 452.0-2410.4 nm, 276 bands, median step 5.01 nm
  interior holes:  917.8->967.9 (50 nm) · 1098.1->1168.2 (70 nm)
                  1308.5->1513.8 (205 nm) · 1744.2->2019.7 (276 nm)
  canonical retained bands with no source within one step:  43/184  (23 %)
      6 below sensor start · 9 above sensor end · 28 in interior holes

AVIRIS-Classic post-band_select: 433.7-2497.0 nm, 162 bands, median step 9.93 nm
  interior holes:  889.2->976.2 (87 nm) · 1101.0->1167.9 (67 nm)
                  1323.2->1512.6 (189 nm) · 1771.6->1988.3 (217 nm)
  canonical retained bands with no source within one step:  23/184  (12 %)

RAW ENVI, same canonical grid, same water windows:
  aviris_ng  376.9-2500.5 nm, 425 bands, step 5.01 -> interior gaps: 0, uncovered: 0/184
  aviris     365.9-2497.0 nm, 224 bands, step 9.94 -> interior gaps: 0, uncovered: 0/184
```

**Nearly a quarter of every harmonized NG spectrum would have been invented.** `np.interp`
does not raise on an interior gap — it draws a straight line across it — and outside the range
it clamps to the edge value. The result is smooth, correctly shaped, passes
`shape[-1] == RETAINED_BANDS`, and is fabricated across a 276 nm stretch of SWIR. This is the
same failure mode as D11.4 arriving through a different door: an operation that is defined on
bad input instead of rejecting it.

**Decision.** `harmonize()` reads the **raw ENVI cubes** (`HAD100/data/**.dat` + `.hdr`), which
cover all 184 retained canonical bands with zero gaps for both sensors. `main.py` is used for
its **crop geometry only** — the `crop_dict` offsets and the four-corner rule are imported from
it verbatim, so the spatial comparability argument of D11.2 is fully preserved. We do not
re-derive where the targets are; we simply decline to inherit a spectral subsetting that was
designed for a different pipeline than ours.

Consequences, all of which are simplifications:
- `RETAINED_BANDS = 184` stands unchanged, and is now genuinely *covered* rather than nominal.
- **No NaN reaches `reduce_bands`.** Neither PCA nor kPCA accepts NaN, so had we kept
  `band_select` we would have been forced to either drop NaN columns before fitting or narrow
  the canonical grid to the cross-sensor intersection — two different `C=30` bases, with the
  pickled transformer having to match whichever was chosen. Reading raw removes the choice.
- `HAD100Dataset/` (the `main.py` output) is still produced and still used — as the
  published-comparability scoring path, so our HAD100 numbers remain comparable to Li et al.
  It is **not** the model-input path. Two paths, stated, rather than one path silently doing
  both jobs.

**Assert it.** `harmonize` fails if any retained canonical band has no source wavelength within
one median sensor step, rather than interpolating across the hole. `test_harmonize.py` covers
both directions: raw NG/Classic pass with 0 uncovered bands; a synthetic `band_select`-style
gapped input raises.

**Re-verify with `scripts/verify_had100.py`.** It parses every ENVI header and re-derives all
of the above. Rerun it if the archive is ever re-fetched; the numbers here are only as good as
the file whose sha256 is recorded above.

### D12 — Cloud GPU: local-first for anything on the critical path; Colab/Kaggle is a labelled stretch arm only.

D8 fixes the local budget at a GTX 1650, 4 GB. The obvious temptation is to reach for a free
cloud notebook to escape it. That is allowed, but only under a rule, because the failure mode
is not "we run out of quota" — it is a table row that reads like a result on our hardware and
is not.

**Rule 1 — the critical path never leaves the local GPU.**
Every model that gates Phase 4 integration trains locally on the GTX 1650 at the scoped-down
sizes already specified in D8. That is `LightUNet`, `CompactSegFormer`, the compact
spectral-spatial `deep_detector`, the autoencoder, and the Siamese change net. The reason is
scheduling, not principle: Colab's free tier disconnects idle sessions, caps a session at
~12 h, and does not guarantee a GPU is available at all. A critical path that depends on that
is a critical path with a third party's queue in it. Everything in §11's DAG must be
reproducible on hardware the team owns.

**Rule 2 — Colab/Kaggle is a stretch arm for `anomaly/deep_detector.py`, and nothing else.**
The one experiment worth the extra platform is the arm D8 had to scope down: the
graph-transformer detector originally named in `blueprint_upgrades_changelog.md`. Training it
at full size on a 16 GB card is a genuine bonus comparison — "here is what the scoped-down
model costs us" is a stronger claim than silence. It is a *bonus row*, never a replacement for
the local result, and never a prerequisite for any downstream stage.

| Platform | Free-tier GPU | Session / quota (vendor-variable — re-check before relying on it) |
|---|---|---|
| Google Colab | T4, 16 GB VRAM | up to ~12 h/session, roughly 15–30 GPU-h/week, **not guaranteed available**; idle sessions are disconnected |
| Kaggle Notebooks | P100 16 GB, or T4 ×2 (2 × 16 GB) | ~30 GPU-h/week, 12 h/session, ~9 h with background execution (survives closing the tab) |

Both vendors state these limits fluctuate. Treat the table as orientation, not as a contract;
whoever schedules a run confirms the current limit that day.

**Rule 3 — the labelling discipline, identical to `SIMULATED` in §0.3.**
Any number produced on Colab or Kaggle is tagged
**`trained on Colab/Kaggle (T4/P100) — not reproducible on target local hardware`**
in its checkpoint filename, in its results JSON, and in **every table it appears in**. Same
reasoning as §0.3: a figure that did not come from the hardware the reader assumes is a
different kind of claim, and the tag is what keeps the two kinds from blending together in a
results table six weeks later. `test_profiling.py` already asserts the `SIMULATED` tag; it
gains a sibling assertion for this one (§12).

A row carrying this tag may **never** be the headline number for the deep-detector comparison.
The headline is the local result. The cloud row sits beside it as a ceiling.

**Rule 4 — operational notes, learned the hard way by everyone who has done this.**
- **Free-tier quota is per Google/Kaggle account.** Route different training jobs through
  different team members' accounts rather than burning one person's weekly hours. Six people
  is six quotas. Record which account ran which job in the results JSON, or a rerun becomes
  impossible to schedule.
- **Checkpoint to Google Drive every epoch, not at the end.** Free sessions disconnect on
  idle and without warning. A run that only writes its final weights is a run you will do
  twice.
- **Pin the environment.** Colab ships whatever CUDA/torch it ships that month. The notebook
  installs from the same `requirements.txt` lock as §4.2, or the comparison is against a
  different software stack as well as different hardware, and the row means nothing.
- **Never upload the benchmark data to a third-party notebook without checking its licence
  terms first.** HAD100, ABU and EnMAP each carry their own redistribution conditions, and
  "I uploaded it to my Drive to train faster" is redistribution.

### D13 — Indian Pines, ABU and HYDICE as they actually ship. One finding is a blocker.

Verified the same way as D11: files downloaded, every variable loaded, invariants asserted.
Re-derivable via `scripts/verify_benchmarks.py` → `docs/benchmarks_verified.json`.

#### D13.1 Indian Pines — **D2 confirmed**

```
Indian_pines_corrected.mat : indian_pines_corrected (145,145,200) uint16  [955, 9604]
Indian_pines_gt.mat        : indian_pines_gt        (145,145)     uint8   [0, 16]
georeference keys: NONE      wavelength keys: NONE      one variable per file
```

D2's premise was load-bearing and unverified; it now holds. There is no CRS, no affine, no
`map info`, nothing — the files contain exactly one array each. The synthetic-affine design in
D2 and the Phase 2 exit criterion stand as written. Note the cube is **uint16 raw DN, not
reflectance**, and is already the 200-band water-band-removed product (220 → 200 upstream).

#### D13.2 ABU — the plan's "205 bands" is wrong for 8 of 13 scenes

Counts and grouping are right: **13 scenes, 4 airport / 4 beach / 5 urban**. Everything the
plan said about the spectral axis is wrong.

```
scene            cube               dtype     anom px    % of scene
abu-airport-1    (100,100,205)      uint16       144      1.44
abu-airport-2    (100,100,205)      uint16        87      0.87
abu-airport-3    (100,100,205)      int16        170      1.70
abu-airport-4    (100,100,191)      uint16        60      0.60
abu-beach-1      (150,150,188)      int16         19      0.084
abu-beach-2      (100,100,193)      int16        202      2.02
abu-beach-3      (100,100,188)      int16         11      0.11
abu-beach-4      (150,150,102)      float64       68      0.302
abu-urban-1      (100,100,204)      int16         67      0.67
abu-urban-2      (100,100,207)      int16        155      1.55
abu-urban-3      (100,100,191)      uint16        52      0.52
abu-urban-4      (100,100,205)      int16        272      2.72
abu-urban-5      (100,100,205)      int16        232      2.32

band counts : {205:5, 191:2, 188:2, 193:1, 204:1, 207:1, 102:1}  -> SEVEN distinct
spatial     : 100x100 x11, 150x150 x2      ("mostly 100x100", now quantified)
dtypes      : int16 x8, uint16 x4, float64 x1  -> THREE distinct
keys        : ('data','map') in all 13      masks binary {0,1} in all 13
```

`abu-beach-4` at 102 bands is the ROSIS scene; the rest are AVIRIS variants. **Nothing in the
plan may key on "ABU = 205 bands"** — §3A.1's accept criterion said exactly that and is
corrected. Three dtypes across one dataset means the loader normalizes dtype explicitly;
`int16` vs `uint16` on raw DN is a sign-interpretation bug waiting to happen.

The anomaly fraction spans **0.084 % to 2.72 %, a 32× range**. Any single global threshold, or
any pooled metric that is not scene-weighted, is dominated by the two densest scenes. This is
why §3A.10 and §3B.8 define **pooled-macro (primary)** and **pooled-micro (secondary, labelled)**
rather than a bare "pooled", and it bears directly on §4.2's recall calibration.

#### D13.3 HYDICE — the plan named the wrong dataset

**Two different datasets are both called "HYDICE urban", and the plan described one while
pointing at the other.**

| | HYDICE *anomaly* scene (what we have) | HYDICE Urban *unmixing* scene |
|---|---|---|
| Site | Michigan, USA | Copperas Cove, TX |
| Size | **80 × 100 × 175** float64 in [0,1] | 307 × 307 × 210 → **162** bands |
| Ground truth | binary anomaly mask, **21 px in 10 components** | six-endmember abundance maps |
| Use | anomaly detection ✔ | spectral unmixing ✘ |

The plan said "HYDICE (162)" — that is the *unmixing* scene's band count, and its ground truth
is endmember abundances, not an anomaly mask. It cannot be scored against `map`-style pixel
masks at all. **Our file is the 175-band Michigan anomaly scene**, which is the correct one for
this pipeline; only the description was wrong.

**The directory is therefore named `hydice_urban_anomaly/`, not `hydice_urban/`** — the
distinction is made visible in every path that mentions it, and `SceneMeta.source` (C1) rejects
the bare `hydice_urban` spelling outright. Its `README.md` ships alongside the `.mat` and states
the Michigan provenance — keep both.

`scripts/fetch_hydice.py` is **pinned by committed SHA256**
(`a998766a…93f4b`, 2 544 597 B) and additionally asserts `shape == (80,100,175)` and
`21` anomaly pixels after download. The checksum is the only reliable discriminator between the
two datasets, because the file we want is itself named `HYDICE-urban.mat`. §1.6 previously said
"public mirrors", which is not an executable instruction and would have let a contributor fetch
the 162-band unmixing scene and never notice. `docs/datasets.md` states the prohibition
explicitly for anyone who searches "HYDICE" later.

#### D13.4 BLOCKER — none of the three ships wavelengths, and D9 needs them

```
Indian Pines  wavelength keys: NONE
ABU (13/13)   wavelength keys: NONE
HYDICE        wavelength keys: NONE
```

Every one of these is a bare `.mat` holding a data cube and a mask. `harmonize()` (D9)
interpolates **from source wavelengths onto the canonical grid** — and there is no source
wavelength array to interpolate from. HAD100 was fine because ENVI headers carry
`wavelength = {...}`; these carry nothing.

This is not a formality. ABU has **seven different band counts**, and there is no published
mapping from "191 bands" or "204 bands" to *which* AVIRIS bands survived in that particular
scene. So the wavelengths cannot be reconstructed by assuming a standard subset either — a
guess would be wrong per-scene and would corrupt every harmonized ABU cube silently, in exactly
the way D11.6 describes for interpolation across holes.

**What this blocks:** D7 requires 3B models to be *scored on ABU and HYDICE*, and 3B models
consume the canonical grid. No wavelengths → no harmonize → no canonical grid → **no ABU or
HYDICE score for any learned model.** The classical RX/CRD detectors are unaffected; they run on
native bands (D9) and need no wavelengths at all.

**Do not paper over this.** The three honest options, in preference order:

1. **Source a wavelength table per scene from the ABU authors' original AVIRIS flightlines.**
   Correct if obtainable; requires identifying the parent flightline per scene. Unresolved.
2. **Score learned models on HAD100 only** (which has real wavelengths), and report ABU and
   HYDICE for the classical detectors only. Costs the cross-dataset generalization claim in
   §3B.8 and shrinks the LODO matrix in §3B.5b — say so in the report rather than quietly
   dropping rows.
3. **Resample by band index rather than wavelength** for these datasets, labelling every
   resulting number **not wavelength-registered**. Cheapest, and the most dangerous: a
   plausible-looking number with no physical basis, which must never be pooled with a
   wavelength-registered one.

**This was `O8`. It is now DECIDED — option 2 is adopted** and written into §3B.8, §13 rule 6
and the C1 `source` enum: learned models score on HAD100 only, classical detectors keep all
three. **It no longer blocks 3B.** The residual research question — can the true wavelengths be
recovered from NASA's per-flight calibration archive — is **O9**, which blocks nothing.

---

### D14 — Phase 1–2 walking skeleton built and run (2026-08-21). Two spec errors found by execution, not review.

Phase 1 is now actually bootstrapped, not just specified: `.venv` (Python 3.12.13) holds the
full §1.2 package list, locked via `uv pip compile --generate-hashes`; `.tooling/venv` is
deleted per §1.1's instruction, once `scripts/verify_had100.py` and `scripts/verify_benchmarks.py`
passed from `.venv`. `core/contracts.py` implements C1–C6.

Phase 2's spine — `raster_loader → normalize → global_rx → scoring → postfilter → polygonize
→ projections → geojson → run_pipeline` — is built and green: 54 tests pass, and
`python -m pipeline.run_pipeline` runs end to end on `Indian_pines_corrected.mat`, producing 3
ROIs, `validate_geojson` green, export CRS confirmed `EPSG:4326`.

Two things in this document were wrong and were caught only by running real files through the
code, not by reading the spec more carefully — the same lesson D11/D13 already taught about
documentation:

**D14.1 — §2.2's water-band indices target the wrong band count.** The comment
`[104-108,150-163,220]` indexes into the *original* 220-band AVIRIS layout. D13.1 already
established that `Indian_pines_corrected.mat` is *itself* the 200-band water-band-removed
product (220 → 200 upstream, one variable, no record of which 20 bands were dropped) — but
D13.1 never connected that fact back to §2.2's `drop_bad_bands` accept criterion, so the
mismatch survived. Applying the 220-band indices to the shipped 200-band array drops the wrong
20 bands outright, and index `220` doesn't exist in a 200-band array at all.
**Fix:** `drop_bad_bands` is a documented no-op for `source == "indian_pines"`; the removal is
recorded as already-applied upstream, not re-applied.

**D14.2 — ENVI `map info` rotation is a live trap, not an edge case, and it bites in 3A.1.**
A first implementation of the `.hdr` branch of `raster_loader.py` hand-parsed `map info`'s
tie-point + pixel-size fields into an axis-aligned affine, ignoring the optional trailing
`rotation=` field. **Every real HAD100 header checked carries nonzero rotation** — verified 33°
on an AVIRIS-NG scene (`ang20191004t185054_13.hdr`) and 90° on an AVIRIS-Classic scene
(`f170507t01p00r10_1.hdr`) — so this was not a theoretical gap: diffing the hand-rolled
transform against GDAL's own ENVI driver (`rasterio.open()` on the sibling data file) showed
they disagreed. On the 90° scene the correct transform has `a == 0`, `b == pixel_size`
(rotation swaps scale into the off-diagonal entirely); the hand-rolled version had `a ==
pixel_size`, `b == 0` — a plausible-looking, wrong affine, exactly the class of bug D2/D11 exist
to catch.
**Fix:** `raster_loader._load_envi` now delegates CRS/transform/nodata to `rasterio.open()` on
the sibling data file (located via `spectral`'s own `img.filename`) instead of re-deriving
rotation math by hand; `spectral` is still used for the cube array and wavelength parsing, per
§2.1's original dispatch. Pinned in `tests/test_loader.py` by asserting the loader's transform
equals GDAL's on both a rotated NG and a rotated Classic scene.
**Why this belongs in the decision log and not just the commit:** D2 and this section's own
text say real georeferencing is verified for the first time in **3A.1**, on these same HAD100
headers. Anyone hand-rolling ENVI georeferencing there — rather than reusing `raster_loader.py`
— inherits this exact trap. `gsd_m` is derived as `hypot(transform.a, transform.b)`, which
assumes square pixels (`px == py`, true of every header checked so far but not asserted).

**§2.10 QGIS verification is NOT done.** No GUI was available to build this Phase 2 walking
skeleton, so the interactive confirmation in `qgis/projects/phase2_verify.qgz` does not exist.
As a partial substitute, §2.6's accept criterion (polygon bounds vs. `meta.transform`-derived
pixel bounds, exact to 1e-6) was run against the **real** Indian Pines pipeline output, not just
a synthetic fixture: all 3 real ROIs matched. This is evidence the affine plumbing is right; it
is not the written exit criterion. **Phase 2 exit is not formally signed off — see O4.**

---

### D18 — Local-GPU training and preprocessing budgets, measured. Neither Colab nor CuPy/cuML is needed for the scheduled work.

Two recurring proposals — "train on Colab, the 4 GB card is too small" and "accelerate
preprocessing with CuPy + RAPIDS cuML" — are both answered by measurement rather than judgement.
Measured 2026-08-21 on the actual target hardware (GTX 1650, 4096 MiB, torch 2.13.0+cu130).

**Training (§3B `LightUNet`, representative 0.29 M-param U-Net, C=30, 64×64, batch 16, AMP fp16):**

```
per iteration   73.7 ms         per epoch    9.65 s  (131 steps, 2088 patches)
100 epochs      16.1 min        300 epochs   48.3 min
peak VRAM       113 MiB of 4096            ~3% of the card
```

The 4 GB limit is **not binding** for the P0 model — there is ~36× headroom. The reason is
`reduce_bands`: training consumes **C=30**, not 184 bands. D12 Rule 1 therefore stands on
scheduling grounds alone, and now also on the simple ground that the local card is sufficient.
If `train_unet.py` lands materially larger, re-measure — but it would need ~30× more model
before VRAM matters.

**Preprocessing (§3E path: PCA C=30 → `n_features=8`, then MinMax to [0, π]):**

| workload | PCA fit+transform | MinMax |
|---|---|---|
| one patch (4 096 px) | 100.6 ms | 1.3 ms |
| 100 k px subsample | 26.5 ms | 14.8 ms |
| 1 M px subsample | 154.6 ms | 134.7 ms |

**Under 0.3 s on CPU for a million pixels.** A CuPy/cuML port would spend comparable time on the
host↔device round trip alone, before cuML's multi-second import. There is no CPU bottleneck here
to remove: §3E's cost is **AerSimulator circuit execution**, which is unaffected by how the
features were produced.

**Consequence for the quantum branch.** §3E.2 already requires `classical_reduce` to reuse
`preprocessing/harmonize.reduce_bands`, because *"comparing a quantum model on one feature basis
against a classical model on another measures the basis, not the model."* A separate GPU
preprocessing pipeline with its own PCA and its own band selection would produce a different
basis and **invalidate §3E.6, which is the branch's actual deliverable**. It would also
re-implement band selection outside `harmonize()`/`coverage_ok` (the D11.6 trap) and re-open the
train-split-only fitting constraint that D15 closed.

**If GPU preprocessing is still wanted**, it is an *optional backend behind the existing
interface*, never a parallel path. Conditions, all of them: it consumes `harmonize()` output and
never selects bands itself; it applies the **already-fitted** transformer from §3B.3 and never
re-fits; `cupy`/`cuml` are optional imports that fall back to NumPy, with tests `skipif`-guarded
so a fresh clone stays green (CONTRIBUTING); and a numerical-equivalence test asserts the GPU path
matches the CPU path within tolerance. A CPU-vs-GPU preprocessing benchmark is then a legitimate
engineering result, reported separately from the quantum comparison — which is what the proposal
was actually after.

> Note also §0.3: `edge/` code must run with **no CUDA at inference**. A CUDA-only preprocessing
> path cannot appear anywhere the edge arm depends on.

### D16 — EnMAP L2A does NOT fully cover the canonical grid. `harmonize()` raises on it, correctly.

Verified 2026-08-21 against a real product metadata file
(`ENMAP01-____L2A-DT0000207400_20260802T060817Z_021…-METADATA.XML`, 4.2 MB, saved to
`data/raw/enmap/`). This is the first **file-opened** EnMAP fact in the project; everything
previously in §15 was a project-page claim.

```
bands                 224   (91 VNIR + 133 SWIR)
range                 418.42 – 2445.30 nm      strictly increasing: True
median step           7.98 nm                  FWHM 5.80 – 11.43 nm
bandInterpolation     No                       terrainCorrection: Yes
backgroundValue       -32768                   (int16 sentinel, matches STAC data_type)
```

**Two interior gaps**, and where they fall is the whole story:

| gap | width | verdict |
|---|---|---|
| 1390.48 → 1449.43 | 58.95 nm | **harmless** — sits inside the 1350–1450 water window, dropped anyway |
| 1780.22 → 1967.66 | 187.44 nm | **straddles** the 1800–1950 water window; 1790 nm falls outside it and is genuinely uncovered |

**`coverage_ok(enmap_wl, retained) == False`. 8 of 184 canonical bands are uncovered:**

```
400, 410           below EnMAP's first band (418.42 nm)
1790               inside the SWIR detector gap, just short of the water window
2460 … 2500  (5)   above EnMAP's last band (2445.30 nm)
```

So `harmonize()` **raises** on EnMAP L2A as §3A.1 is currently written. That is the D11.6
self-defence firing correctly, not a defect: `np.interp` would otherwise clamp at both edges and
bridge the 1790 nm hole, fabricating reflectance across 187 nm of missing SWIR.

**Do not widen `tol` to make this pass.** Widening the tolerance is fabrication with extra steps —
the same failure D11.6 exists to prevent, arrived at deliberately instead of by accident.

**Resolution — per-sensor bad-band mask.** EnMAP contributes **176/184** valid canonical bands with
those 8 marked invalid in `meta.bad_bands`. The pool still stacks to one `[N, 64, 64, 184]` tensor
(D11.3 holds), carrying a validity mask rather than invented values. Detectors must already honour
`bad_bands`; confirm they do before EnMAP enters the pool.

**Grid stability — checked across all 8 downloaded scenes, 2026-08-21.** The caution above was
that a per-acquisition SWIR-detector choice might change the wavelength array. Measured:

| checked | result |
|---|---|
| 8 metadata files parsed | all **224 bands, 418.42–2445.30 nm, 91 VNIR + 133 SWIR** |
| SHA-256 of the wavelength array | **one distinct grid — byte-identical across all 8** |
| `SWIRAOrSWIRBSelected` | **SWIRA × 8** |

So the grid is stable, and the 8 uncovered canonical bands above are a **constant** for SWIRA
acquisitions — the per-sensor bad-band mask can be computed once, not per scene.

> **Still open: SWIRB.** All eight scenes in hand are SWIRA, so this verifies stability *within*
> SWIRA and says nothing about SWIRB. Keep reading the wavelength array from each product's
> METADATA.XML rather than hardcoding; treat a SWIRB acquisition as unverified until one is
> opened. §8.0 stands for SWIRB.

### D15 — `preprocessing/harmonize.py` built (2026-08-21); the C=30 criterion moved to §3B.3 for a leakage reason, not an assembly-order one; the first real georeference check (§3A.1) is closed.

**`reduce_bands` deferred to 3B — corrected reasoning.** D14 originally deferred `reduce_bands`
because the background pool isn't assembled until 3B. That is true but not the binding reason:
the PCA/kPCA transformer must be **fit on the train split only** — fitting on the whole pool (or
on anything before a split exists) leaks scoring-scene spectral statistics into the
representation every model is built on, the same class of leak §3B.5b's LODO matrix exists to
catch, one level deeper (not a scene in a training manifest — the scene's *statistics* baked into
the basis every scene, including held-out ones, gets projected through). §3A.1's Accept line and
`reduce_bands`'s docstring are amended accordingly: **3A's accept criterion is now `shape[-1] ==
184` and the two sensors stack** — the D11.3 join, verified, nothing about `C=30`. The `C=30`
claim now lives in **§3B.3**, tied to `RealSegDataset`'s existing train/eval boundary, with the
fit-on-train-only constraint stated explicitly.

**`harmonize()` is self-defending, verified by execution.** `coverage_ok` is checked *before*
interpolating and raises rather than fabricating; fed a reconstructed `band_select`-style gapped
axis (the exact NG index slices from `main.py`), it reproduces the plan's own measured figure —
**43/184 uncovered** — exactly, and `harmonize()` raises. Fed real raw-ENVI NG (425) and Classic
(224) headers, both harmonize to `shape[-1] == 184` and stack into one array. The `band_select`
fixture has four holes and truncated endpoints at once, which can't distinguish "caught the hole"
from "caught the truncation" — closed with an isolated case: a source axis dense everywhere
except one clean 200 nm interior gap in a retained region, confirmed `False`; and the inverse, a
uniformly coarse (40 nm step) but gapless axis, confirmed `True` — coarse-but-gapless is
genuinely interpolable and `coverage_ok`'s median-step tolerance must not reject it.

**A real numerical bug, found by a test before it shipped.** The first implementation built the
184-target interpolation as one weight matrix ([184, n_src]) applied via a single matmul over the
full source band axis — mathematically equivalent to `np.interp` on clean input, verified so
directly. But `0.0 * NaN == NaN` in IEEE 754: a matmul sums every source column, including
zero-weighted ones, so **one NaN band anywhere in a pixel's spectrum poisoned every one of the
184 output bands for that pixel**, regardless of that band's actual interpolation weight —
silently collapsing per-band nodata into whole-pixel nodata. A synthetic test poisoning a single
mid-spectrum band caught this immediately (all 184 outputs went NaN, not the 1-2 that
legitimately reference it). **Fixed** by gathering each target band from only its (at most two)
bracketing source indices rather than multiplying through the full matrix — the matrix-based path
is kept only as a `np.interp` cross-check in tests, never in the production path.

**Verified, not assumed: HAD100 currently has zero nodata pixels of either shape.** The plan
(§D11.5) records two nodata sentinels in use (`-9999.0`, `1e-34`) and warns against masking only
one. Checked directly: across **all 616** raw ENVI scenes, at float32 precision (matching
`raster_loader`'s actual cast), **zero pixels exactly equal their declared sentinel** — 354/616
headers declare one, none of the 616 scenes' data actually contains it. Whether nodata would be
whole-pixel or per-band is therefore currently unverifiable from the files — there is no nodata
to inspect. `harmonize()`'s NaN handling (above) is correct for both shapes regardless, which is
what the synthetic test proves; nothing here should be read as "HAD100 has no nodata," only as
"not in the archive as currently downloaded."

**First real georeference check closed (§3A.1, D2, D11.5).** Independently re-derived the
pixel→world affine straight from raw `map info` text (tie point, pixel size, **rotation** —
33° on the NG header, 90° on the Classic one) using plain trigonometry, with no
rasterio/GDAL/spectral in the derivation path, and it matches `raster_loader`'s GDAL-delegated
transform (D14.2) component-wise on both headers. This is a stronger check than D14.2 alone,
which only established GDAL was being used correctly (self-consistency); this establishes GDAL's
own output is correct against the header's stated rotation (independent correctness). Both real
headers have `tie_x == tie_y == 1.0`, which multiplies the formula's constant-term (tie-point
offset) arithmetic by zero and leaves it untested by that comparison alone — closed with a
separate property test: for a synthetic tie point at 5 different (non-unity, mixed-rotation)
values, the derived affine applied to the zero-based tie pixel recovers the tie point's map
coordinate exactly, by construction, independent of any real file. Then ran the actual
`pixel → world → EPSG:4326` path (§2.6–2.7) on a known pixel block of each real scene: the
exported centroid falls within one pixel of the independently-computed coordinate on both.
Indian Pines could not do this (D2, synthetic affine); this is the first time real coordinates
have been exercised anywhere in the plan, and Phase 5 Level 2 is now the first *independent*
recheck rather than the first check, as D11.5 anticipated.

---

### D17 — HAD100 background pool built and cached (2026-08-21): 522 scenes → 2088 harmonized patches. The binding resource constraint was RAM, not disk, and the first attempt was OOM-killed finding that out.

**Note on numbering:** D16 (above) was written by other work in this project between sessions —
this entry did not exist when D16 was inserted, and D16 landed positioned before D15 in the file.
Left as found rather than reordered, to avoid conflicting with work in progress elsewhere. **D16
is also directly relevant here**: it establishes that `harmonize()`'s hard-raise-on-any-gap
behaviour is too strict for EnMAP (176/184 covered, 8 raise) and proposes a per-sensor bad-band
mask instead of a raise. That resolution is **not implemented** — `harmonize()` still only
raises or fully succeeds, no partial mode. Out of scope for this entry (HAD100's own raw ENVI
has 0/184 uncovered on both sensors, D15), but the next thing to touch `harmonize()` should read
D16 first.

**Built:** `preprocessing/background_pool.py` (`four_corner_offsets`, `harmonize_and_crop_scene`,
`build_background_pool`, `build_background_pool_to_disk`, `scene_groups`, `save_pool` /
`save_pool_manifest`) and `scripts/build_background_pool.py`. Crop geometry is HAD100/main.py's
own four-corner rule (lines 103–111), reproduced as offset arithmetic and cross-checked against
main.py's literal slicing on a real cube — not re-derived as a judgement call (D11.2). Source is
the raw ENVI cube per scene (D11.6); each full scene is harmonized **once**, then four 64×64
corner crops are taken from the harmonized result — verified numerically identical to
harmonizing each crop separately (harmonize() is per-pixel, order-independent), at roughly a
quarter of the interpolation cost.

**Result, verified against the built artifact, not just the code:** `data/processed/`
`had100_background_pool.npy` — `[2088, 64, 64, 184]` float32, 6.29 GB, sha256 recorded in
`had100_background_pool_summary.json`. 522 unique source scenes (260 NG + 262 Classic), each
with exactly 4 crops; sensor counts exact (1040 NG, 1048 Classic, matching D11 exactly);
`array_index` contiguous 0–2087. `had100_background_manifest.csv` carries `scene_id` per patch —
`scene_groups()` reads it into a `GroupKFold`/`GroupShuffleSplit`-ready array, and is written up
as **the one sanctioned way** to derive train/val indices from this pool, so a later ad hoc
patch-index split has to actively bypass the provided function to leak (D11.5's ~47–62 px
per-axis crop overlap, worse for HAD100's smaller background scenes than the 81×81 case the plan
originally quoted).

**A real resource-constraint bug, found by the first real run, not by review.** The build was
planned around "39 GB free disk, 6.29 GB tensor" — true, and irrelevant to what actually failed.
The first implementation assembled the pool as a Python list of 522 per-scene `[4,64,64,184]`
arrays, then `np.concatenate`d them: **two live copies of the ~6.3 GB tensor simultaneously**
(the list, until GC'd, and concatenate's output). This machine has **13 GB RAM total, 8.3 GB
available, and no swap** (`free -h`, checked directly after the failure, not assumed) — the run
was killed by the OOM killer (exit 137) partway through, silently, with no Python traceback.
**Fixed** with `build_background_pool_to_disk`: a `numpy.lib.format.open_memmap` preallocated at
the final shape, written directly slice-by-slice as each scene is processed, so the tensor never
exists fully in RAM — peak RSS on the real run was ~6.3 GB of mostly-reclaimable memmap page
cache, not two irreducible live copies. Verified byte-identical to the in-memory path on a small
real subset before trusting it at scale. The general lesson, not just this run: a stated resource
budget ("N GB free") answers "does it fit on disk," not "does it fit in the memory model actually
used to build it" — those are different questions, and this project has no swap to blur the
difference.

**Leakage constraint is enforced by a test, not just documented.** `tests/test_data_hygiene.py`
runs against this real 522-scene manifest: a naive `train_test_split` on patch index is shown to
actually leak at least one scene across train/val (the failure mode, demonstrated, not asserted
away), and `GroupShuffleSplit(groups=scene_groups(records))` is shown to leak none. This is §12's
`test_data_hygiene.py` third check (crop-level leakage) — the file didn't exist until this pool
did, and it's now the thing standing between a future `pool[:1700]`/`pool[1700:]` split and a
green suite.

**Forward-compat gap, not a bug today.** D16 resolves EnMAP's 8 uncovered canonical bands with a
per-sensor bad-band mask rather than a raise, and says the pool "still stacks to one
`[N, 64, 64, 184]` tensor." The cached format built here cannot carry that: the tensor is bare
float32 with no per-patch validity mask, and `BackgroundPatch` has no `bad_bands` field. Nothing
is wrong for HAD100 alone — both raw sensors are 0/184 uncovered (D15) — but D16's resolution is
a *format* change to this cache, not an append, and whoever wires EnMAP into the pool will hit
that the first time they try.

### D19 — §3B built (2026-08-21): `synth.py` → `datasets.py` → `train_unet.py`. `unet_all_real` in §3B.8's table has the same missing-wavelength problem as the two rows already marked suspended — it just wasn't marked.

**Built:** `segmentation/synth.py` (`implant_targets` — linear mixing `m = a*t + (1-a)*s`,
abundance sweep, exact/free masks, provenance metadata; `pseudo_anomaly_patch` — 4-kind
self-supervised pretext task with no `target_spectra` parameter at all, so the zero-prior
property is structural, not just documented), `segmentation/datasets.py` (`train_val_scene_split`
— `GroupShuffleSplit` keyed on `scene_groups()`, never patch index, per D11.5;
`fit_reduce_bands_transformer` — streams the TRAIN split only from the memmap, bounded pixel
sample, never touches val/eval, closing the D15 leakage gap `reduce_bands` was deferred out of
§3A.1 for; `SyntheticSegDataset` / `RealSegDataset`, the latter asserting `split == "eval"` so
training on real GT is a structural error, not a discipline problem, per D7), and
`segmentation/train_unet.py` (`LightUNet`, ~1.9M params, `combined_loss` = 0.5 BCE + 0.5 Dice per
§3B.4). 94→126 tests, all passing; `pytest` and both `verify_had100.py`/`verify_benchmarks.py`
clean before and after.

**Two real findings from execution, not from re-reading the plan:**

**1. PCA output scale.** `reduce_bands`'s PCA on raw radiance-scale HAD100 patches (not
reflectance) produces components in the thousands (observed range roughly −6500 to +8 across a
real patch) — numerically unsafe for fp16 and borderline even at fp32. Fixed by standardizing
(per-band zero-mean/unit-variance, `preprocessing/normalize.py`) the reduced cube before it
reaches the model, in both `SyntheticSegDataset` and `RealSegDataset`. Not previously stated
anywhere in §3B — `reduce_bands` itself does no scale normalization by design (D15 only specifies
*where* it's fit, not output scale), and nothing upstream of it happens to produce reflectance-
scale data, so this bites in `datasets.py`, not `harmonize.py`.

**2. AMP is currently unsafe on this machine's GPU — `train_unet`'s `amp` default is now
`False`, not §3B.4's stated "AMP fp16."** Under `torch.autocast`, `LightUNet`'s `dec2` Conv2d
produces NaN from a fresh forward pass on fp32-clean input, on this exact stack (GTX 1650, cuDNN
9.2.0, torch 2.13.0+cu130). Isolated by layer-by-layer forward inspection, then confirmed with
`torch.backends.cudnn.enabled = False`, which makes the NaN disappear — consistent with a cuDNN
fp16 kernel issue on this GPU/driver stack, though a genuine fp16-range/accumulation limitation in
that same kernel path reads the evidence equally well; the fix is identical either way.
`cudnn.benchmark = True` only masks it for the first few calls (0/20 clean iterations in a real
optimizer loop once the algorithm cache settles). fp32 training is fully stable (0/30 NaN) and
uses ~244 MB VRAM at batch=16 — nowhere near the 4 GB budget (D8), so disabling AMP costs nothing
here. Pinned as a permanent regression test, `test_amp_is_currently_unsafe_on_this_gpu` in
`tests/test_train_unet_real_gpu.py` — if a future driver/cuDNN update makes it start failing
(no more NaN), that's the signal to reconsider the default.

**This is a live concern for §3B.5 (SegFormer, not yet built).** D8 sizes that arm as "batch 8 +
grad-accum 2 + AMP at 4 GB" — i.e. it assumes AMP works to fit the budget. That assumption is now
unverified on this hardware, not confirmed-and-forgotten. Whoever builds §3B.5 should re-run this
same isolation (autocast on → layer-by-layer NaN check → `cudnn.enabled = False` control) against
SegFormer specifically before trusting AMP there, and if it reproduces, re-derive whether
fp32 SegFormer actually fits in 4 GB before committing to that arm's batch size.

**A plan gap, found while implementing `synth.py`'s `load_target_spectra`, not previously
written down anywhere.** §3B.8's table (line ~1779) lists `unet_all_real`'s spectra provenance as
`lib`+`abu_real`+`hyd_real` and marks it **scoreable on had100/test only** — with no suspension
note, unlike `unet_lodo_abu`/`unet_lodo_hyd` immediately above it. But `abu_real` and `hyd_real`
mean *real target pixel spectra pulled from the ABU/HYDICE GT masks* (§3B.7, line ~1637), and
implanting a target spectrum into a canonically-harmonized 184-band background patch requires
that spectrum to be on the same 184-band grid — which requires a wavelength array to harmonize
it. ABU and HYDICE ship no wavelength arrays at all (D13.4/O8). This is the *same* underlying
constraint that suspended the two LODO rows, just biting on the training-input side instead of
the scoring side — O8 being "decided" (score learned models on HAD100 only) did not also resolve
the training-input question, and O9 (wavelength recovery) is what would. `synth.py`'s
`load_target_spectra("abu_real")` / `("hyd_real")` therefore raise `NotImplementedError` right
now, pending O9, exactly like the LODO rows' scoring path — `unet_all_real` is equally suspended
and the table should say so. This was decided without stopping for user sign-off, on the
reasoning that it's mechanically forced by O8/D13.4, already-accepted project constraints, not a
new judgement call; flagged here for the same reason D-entries exist at all, so a later reader
doesn't have to re-derive it. §3B.8's table (line 1779) and its "suspended, not deleted" paragraph
(lines 1783–1787) should be updated to include `unet_all_real` alongside the two LODO rows —
not done as part of this entry, since editing that section is a §3B.8 change and this entry only
records the finding; see the table edit alongside this one.

---

### D20 — `fuse_scores` cannot run its 4-component form on ABU, and §3A.9's accept criterion is written against a scoring set that can only support 3. Same root cause as O8/D13.4, third surface.

**The collision, in four lines already in this plan:**

1. §3A.8 — `spectral_index_score` selects bands "by nearest wavelength via `preprocessing/bands.py` — **never by band index**, which differs per sensor."
2. D13.4 / O8 — ABU, HYDICE and Indian Pines ship **no wavelength array**. HAD100 is the only benchmark that does.
3. §3A.9 — default weights are `{rx: 0.40, ace: 0.25, index: 0.15, spatial: 0.20}`.
4. §3A.9 accept — "fused AUC ≥ best single component on **≥ 10 of 13 ABU scenes**."

So the accept criterion is evaluated on the one dataset where one of its four components is
structurally unavailable. An agent handed §3A.9 unamended has exactly two moves, and **both are
wrong**: hardcode ABU band indices (violating §3A.8's explicit prohibition, and silently, because
a wrong-index NDBI still returns a plausible-looking float array), or stall on a P0 task.

**Decision — fusion is component-adaptive, and the component set is reported beside every number.**

`fuse_scores` takes whatever components it is given, renormalizes the weights of the components
actually present to sum to 1.0, and records the active component set in its output metadata. It
does **not** silently substitute a zero raster for a missing component: a zero-filled `index`
channel after rank-normalization is not "no information", it is a constant that drags every
fused score toward the same value and would quietly damage the ranking the AUC is computed from.

| scoring set | wavelengths | components | weights |
|---|---|---|---|
| **HAD100** (100 test patches) | real, verified (D11) | `rx` · `ace` · `index` · `spatial` | the §3A.9 defaults, unchanged |
| **ABU** (13), **HYDICE** (1), **Indian Pines** | none (D13.4) | `rx` · `ace` · `spatial` | renormalized: `rx 0.4706 · ace 0.2941 · spatial 0.2353` |

**§3A.9's accept criterion is amended** to: fused AUC ≥ best single component on ≥ 10 of 13 ABU
scenes **for the 3-component fusion**, and the table says `fusion(rx+ace+spatial)` in the method
column — never bare "fusion", which would read as the 4-component detector and overstate what was
tested. The 4-component form is accepted separately on HAD100.

**Correction, 2026-08-22 — this entry's own worked example was wrong when first written.** It
gave the renormalized ABU weights as `rx 0.50 · ace 0.3125 · spatial 0.25`, which sum to
**1.0625**, not 1.0: the three active weights sum to 0.85 and I divided by 0.80. Caught by the
agent implementing `fuse_scores`, which asserted the arithmetic instead of transcribing the
table. It implemented the **general rule** — divide each active weight by the sum of the
active weights — which the prose above states unambiguously, and which is correct regardless
of which components are missing. That is the right reading: **the prose is normative and the
worked example is only an illustration.** Recorded rather than silently patched because a
hardcoded weight triple is exactly the kind of thing that gets copied out of a plan table
into a config file, and the wrong one would have skewed every fused ABU score by 6.25% while
still looking like a deliberate choice.

**Why not just score fusion on HAD100 only, like the learned models?** Because the RX family and
CRD are scored on all 13 ABU scenes (§3B.8's classical row), and fusion's whole claim is that it
beats its own best component. Dropping it to HAD100 would compare fusion against a *different*
scene set than the baselines it claims to beat, which is the §3E.2 error — measuring the basis,
not the model — in a different costume. Keeping ABU with a declared 3-component set preserves the
like-for-like comparison, and the missing component is disclosed rather than hidden.

**This is the third surface of one root cause.** O8 hit it on the learned-model *scoring* side.
D19 hit it on the learned-model *training-input* side. D20 hits it on the *classical fusion*
side. All three resolve automatically if **O9** recovers per-sensor wavelengths; none of them is
independently fixable. A fourth surface should be assumed to exist until someone looks: the rule
is that **any module that selects bands by wavelength is unavailable on ABU/HYDICE/Indian Pines**,
and that is a property of the datasets, not of the module.

---

### D21 — `SPECTRA_POOLS["lib"]` is human-gated at BOTH sources, so `unet_implanted_lib` is blocked by retrieval, not by any data limitation. Probed 2026-08-22. One of the two advertised URLs answers a Range request with **HTTP 206 `text/html`**.

D19 recorded that three of §3B.8's five learned arms are suspended pending O9. It did not check
the fourth. **`unet_implanted_lib` is blocked too** — and therefore `unet_pretext` is currently
the *only* trainable arm of five, which makes §3B.8's headline `implanted_lib` vs `pretext`
comparison a one-sided table rather than the comparison the plan is built around. That was
implicit in `synth.load_target_spectra`'s `FileNotFoundError` ("no `scripts/fetch_speclib.py`
exists") but nowhere stated as a §3B.8 consequence.

The cause is **not** the missing-wavelength problem behind O8/D19/D20. It is retrieval.

| source | status | evidence |
|---|---|---|
| `usgs_splib07` | human-gated | DOI `10.5066/F7RR1WDJ` → ScienceBase `5807a2a2e4b0841e59e3a18d`. `usgs_splib07.zip`, 5 479 324 354 B, `"pathOnDisk": "__s3__"`, `"published": false`, `checksum: null`, and an `s3DownloadRequestPageUri`. Both advertised URLs return HTML. |
| `ecostress_aster` | human-gated | `https://speclib.jpl.nasa.gov/download` is a "Request Download" form (6 139 files across 9 categories). No unauthenticated direct-download URL. |

**The trap, and it is a new one.** `https://sciencebase.usgs.gov/manager/download/<cuid>` answers
`curl -r 0-511` with **HTTP 206, `Content-Type: text/html`** — a partial-content response that a
downloader reads as "range requests work, resuming supported", while the payload is the
ScienceBase web UI. A naive resumable downloader writes 5 GB of HTML and exits 0. `assert_magic`
alone would not have saved us either, since it is `assert_not_html` that catches this;
**both guards in `core/http_guard.py` are load-bearing, and this is the first case found where
the status code actively argues for the wrong conclusion.** Note the contrast with the BigTIFF
bug, where our own guard was too strict; here the server is the one lying.

**Consequence for §3B.8, stated plainly.** Of five learned arms: three suspended pending O9
(D19), one blocked pending this retrieval (D21), one trainable (`unet_pretext`). The
implanted-vs-pretext comparison the plan calls "always the headline" cannot be run until a human
fetches an archive. Unlike O9 — which needs per-sensor wavelengths nobody has — **this one is
cheap to clear**: two browser downloads, no account, no payment. It belongs on the same list as
the EnMAP downloads and the QGIS eyeball, and it is the highest-value item on that list, because
it converts §3B.8 from one arm to two.

**What was deliberately NOT done.** Several third-party GitHub repos redistribute ECOSTRESS/ASTER
subsets, and pointing the fetcher at one would have made this D-entry unnecessary. They carry no
publisher-verifiable checksum, and §1.6 requires a provenance record (URL, date, size, license,
citation) per fetcher. Using a mirror satisfies the code path and fails the standard. This project
has already declined that trade twice — D11 (HAD100's own project page wrong five ways) and D13
(ABU and HYDICE wrong three more) — and the CLAUDE.md verification standard is explicit that
documentation is assumed wrong until checked against the files.

**`scripts/fetch_speclib.py` exists now and does the honest half.** `--check` re-resolves the DOI
and re-tests the gate, exiting **2** if `pathOnDisk` stops being `__s3__` — i.e. it is a live
regression test on this D-entry rather than a snapshot of it, and it tells the next reader to
re-probe with `assert_not_html` rather than trusting the 206. `--ingest` refuses to parse until an
archive is actually present, and carries the four checks the parser will need — ascending
wavelength axis (D11.4, silent failure), `coverage_ok` before `harmonize` (D16), **reflectance vs
radiance scale** (D19 measured HAD100 backgrounds at radiance scale, so a `[0,1]` reflectance
endmember mixed by `m = a*t + (1-a)*s` would be invisible at every abundance and the §3B.8
abundance sweep would flatline looking like a model failure), and the `[K, RETAINED_BANDS]` +
provenance-tag return shape. Writing that parser now, against format documentation nobody on this
project has checked against a real file, is precisely the D11/D13 mistake.

---

### D22 — `global_rx` raises `LinAlgError` on 3 of 13 ABU scenes. The ridge is an **absolute** constant against data whose covariance diagonals run 1e4–1e6, so it regularizes nothing. Found by integration, not by any unit test, and both tests and pipeline were structurally incapable of finding it.

Found 2026-08-22 while smoke-testing `fuse_scores` against a real detector on a real scene —
i.e. by wiring two finished components together, which is the first thing neither component's own
test suite does.

**The failure.** `global_rx(cube)` on the raw benchmark cube:

| | scenes |
|---|---|
| `LinAlgError` from `cho_factor` | `abu-beach-1` · `abu-urban-3` · `abu-urban-4` |
| ok | the other 10 |
| ok after `standardize(cube)` | **all 13** |

**The cause.** `sigma = sigma + reg * np.eye(b)` with `reg=1e-6`. ABU ships native radiance
(`max` 4 100–19 492 across the 13 scenes, D13.2), so covariance diagonals run ~1e4–1e6 and a 1e-6
absolute ridge sits **ten to twelve orders of magnitude** below the quantity it is supposed to
condition. It is not a weak regularizer; it is arithmetically absent. Cholesky then fails on
exactly the scenes whose covariance is closest to singular.

This is the *same root cause* the agent building §3A.2 reported independently and correctly as an
aside: `local_rx`'s pinned `reg=1e-4` is "numerically inert at ABU's native radiance scale." Two
observers, two modules, one bug — **every detector in this repo takes an absolute `reg` and every
one of them is scale-blind.** Treat that as the finding, not the three crashing scenes.

**The fix — make the ridge scale-relative:** `reg * (trace(sigma) / b) * I`, so `reg` means "a
fraction of mean band variance" rather than "this many units of whatever the sensor happened to
record in." Verified across all 13:

- all three `LinAlgError` scenes now score, at AUC **0.9800 / 0.9523 / 0.9880** — these were not
  marginal scenes with unstable numbers, they were three of the *strongest* results in the set,
  and the benchmark was silently unable to report any of them;
- the 10 already-working scenes move by **< 0.006 AUC** (largest single change 0.0049 on
  `abu-beach-4`), so this is not a numbers-changing intervention dressed as a bug fix.

**Why nothing caught it, which is the part worth keeping.** Three independent guards all missed
it for the same structural reason:

1. `test_rx.py` scores Indian Pines and synthetic cubes. Indian Pines happens to be conditioned
   well enough to pass. Synthetic data is generated at unit scale, where an absolute `reg=1e-6`
   is a perfectly sensible number — **the fixture's scale hid the bug the fixture existed to find.**
2. `run_pipeline` calls `standardize` *before* the detector, so the operational path never sees
   native scale. The pipeline is not wrong; it is immune, which is worse, because it means
   end-to-end green says nothing about detector robustness.
3. §3A.10 / Phase 5 L1 call detectors **directly** on native-scale benchmark scenes — that is the
   whole point of a detector benchmark. So the one consumer that would have hit this is the one
   that had not been built yet.

**Consequence for the Phase 5 L1 harness, which must be honoured when it is written:** the harness
decides, per detector, whether it standardizes first, and **records that choice beside every
number**. Two detectors compared across different input scalings are not comparable, for the same
reason §3E.2 gives about feature bases. Standardizing is not automatically the right default
either — RX on standardized data is a different estimator from RX on radiance, and the literature
AUCs this project will be compared against are mostly the latter.

**APPLIED 2026-08-22.** `global_rx` and `streaming_rx` now both use `reg * (trace(sigma)/b) * I`.
They had to change together: `streaming_rx` replicated the absolute ridge verbatim, and the
`rtol=1e-5` equivalence that justifies that module's existence would otherwise have been comparing
two different estimators. **`global_rx` now runs on 13/13 ABU scenes, previously 10/13.**

---

### D22.2 — the SAME `reg=1e-6` is also wrong in `kernel_rx`, for a different reason, and there it fails **silently** instead of crashing.

`reg=1e-6` was carried into §3A.3's signature from §2.3's linear RX. But the regularized operator
in kernel RX is a different object: the centered RBF Gram has **unit-scale** entries (`k(x,x)==1`),
so this is not D22's scale-blindness — RBF is scale-free. It is that `N*reg` is simply far too
small, leaving the N×N system effectively unregularized, so it **interpolates** the background
subsample exactly.

The consequence is worth stating precisely, because it is the most dangerous failure mode found so
far. The score stops measuring anomaly and starts measuring *"were you in the random background
subsample?"* — and it does so **without any error**:

| implanted spike | 1σ | 2σ | 3σ | 5σ | 8σ | 15σ | **50σ** |
|---|---|---|---|---|---|---|---|
| rank of the implanted pixel (of 900) | 513 | 481 | 481 | 481 | 481 | 481 | **481** |

Rank is **pinned at 481 regardless of magnitude** — a 50σ outlier and a 2σ one are ranked
identically, because rank had stopped depending on the data at all. On real data (`abu-beach-2`),
`reg=1e-6` scores **AUC 0.845** against **0.921** for anything in `[1e-3, 1]` — a flat plateau, so
`1e-2` is a mid-plateau default rather than a tuned one. Fixed; the §3A.3 unit test that caught it
asserts the implanted target ranks in the top 1%.

**The generalization, which is the point of recording all three of these.** D22, D22.1 and D22.2
are one mistake in three costumes: **a regularization constant transplanted between operators
whose scales have nothing to do with each other.** §2.3 → §3A.2 carried it onto radiance-scale
covariances (inert, D22 aside); §2.3 → §3A.5 carried it into the streaming accumulator (same);
§2.3 → §3A.3 carried it onto a unit-scale Gram (silent mis-ranking). **`crd`'s `lam=1e-2` is the
next one to check** — it regularizes a per-annulus Gram whose scale is again different, and the
§3A.4 accept criterion (`lam → ∞` collapses the score to `‖y‖₂`) tests the regularizer's *direction*
but says nothing about whether its default *magnitude* is in the useful range. Any future detector
that takes a `reg`/`lam` should state what the regularized operator's scale is and why the default
sits inside it.

---

### D22.1 — §3A.9's default fusion weights lose to the best single component on 3 of the first 4 ABU scenes. The grid search is load-bearing, not decorative.

Measured in the same session, `fuse_scores` with the §3A.9 defaults (renormalized 3-component per
D20, since ABU has no wavelengths):

| scene | local_rx | ace | spatial | **fused** | fused ≥ best? |
|---|---|---|---|---|---|
| `abu-airport-1` | 0.9647 | 0.7807 | 0.7881 | 0.9354 | **no** |
| `abu-airport-2` | 0.8959 | 0.7535 | 0.7843 | 0.8822 | **no** |
| `abu-airport-3` | 0.9142 | 0.6308 | 0.9180 | 0.9040 | **no** |
| `abu-airport-4` | 0.9271 | 0.8692 | 0.8230 | **0.9580** | yes |

This is not a bug in `fuse_scores` — the arithmetic is right and the components are correctly
rank-normalized before weighting. It is the defaults: `ace` and `spatial` score ~0.63–0.87 while
`local_rx` scores ~0.90–0.96, and the two weak components jointly carry **53%** of the weight,
so they dilute the strong one. §3A.9's accept criterion ("fused AUC ≥ best single component on
≥ 10 of 13") is **not met by the stated defaults** and would not be met by tuning them gently.

§3A.9 already says the weights are "tuned by grid search on an ABU validation split; the tuning
split is recorded and NEVER reused for reporting." That sentence now has teeth: the sweep is the
only thing standing between the plan's headline fusion claim and a table where fusion loses to
its own best input. **And the leakage constraint inside it is the sharp edge** — ABU is 13 scenes,
so a grid search over all 13 followed by a reported AUC over the same 13 is straightforward
train-on-test, and it would produce a fusion result that beats every baseline for entirely
illegitimate reasons. The tuning split must be carved out and named in
`experiments/rx_vs_ae/fusion_weights.json` before any fused number is reported.

---

### D23 — a scene with zero ROIs crashed the pipeline. Found by the Phase 4 registry test, on the most benign possible input.

`rois_to_geojson([])` raised `ValueError: Unknown column geometry`. `gpd.GeoDataFrame` cannot
infer a geometry column from an empty record list, so the empty case needed its own path and did
not have one.

**This is not a hypothetical edge case.** It surfaced while testing §4.1's registry with
`local_rx(outer=15, inner=3, n_components=12)` on Indian Pines at the 99th percentile: 211 pixels
survive thresholding and **zero** survive `morphological_cleanup`'s opening, because they are
scattered singletons. Zero ROIs is the correct answer there. The pipeline's response to the
correct answer was a traceback.

**Why it matters more than its size suggests.** Phase 5 Level 1 runs **13 ABU + 100 HAD100 + 1
HYDICE + Indian Pines** unattended, and Phase 7 runs a live scripted demo. A clean scene, a strict
threshold, or an aggressive post-filter all produce zero ROIs. The failure mode is therefore:
*the benchmark crashes partway through a 115-scene sweep, on whichever scene happened to be
cleanest.* And a demo that dies on "found nothing" fails in front of judges on the one input where
the system is behaving perfectly.

**Fixed** by writing a schema-correct empty `FeatureCollection` by hand rather than through
`GeoDataFrame`. `validate_geojson` already accepted it — it iterates `features`, so zero features
passes — and QGIS opens it as an empty layer instead of refusing the file. Regression-tested at
both levels: the writer directly, and the full pipeline on the detector config that first exposed
it.

**The pattern worth noting, since this is the fourth of its kind.** D22, D22.1, D22.2 and D23 were
all found by *integration*, not by unit tests, and all four were invisible to a green suite. The
suite had 200 passing tests when this bug was live. Unit tests check that a function does what it
does; they systematically miss the empty case, the extreme case, and the case where two correct
components disagree about a convention. **The Phase 5 harness should be treated as a bug-finding
instrument, not merely a reporting one** — it is the first thing in this project that runs every
detector against every scene, and on this evidence it will find more.

---

### D24 — `global_rx` accumulated its covariance in **float32**. §3A.5 wrote the warning about exactly this hazard into the *streaming* spec and never applied it to the reference implementation the streaming module is validated against.

§3A.5 requires `streaming_rx` to match `global_rx` to `rtol=1e-5`, and says why the accumulator
must be float64: *"float32 co-moment accumulation loses precision over 20 000+ pixels and quietly
biases the covariance."* The agent building `tests/test_streaming_rx.py` measured the actual gap
and found it nowhere near the spec:

| scene | measured max relative difference | vs spec `1e-5` |
|---|---|---|
| Indian Pines | 8.5e-4 | 85× looser |
| `abu-urban-3` | **4.3e-2** | **~4 300× looser** |

**The reference was the imprecise one.** `global_rx` did `flat = cube.reshape(-1, b)` on a float32
cube, so `centered.T @ centered` was a **float32** matmul over 10 000+ pixels. Verified directly:
max relative error **1.15e-4** against a float64 reference of `global_rx`'s own formula on
`abu-urban-3`. `streaming_rx` was already float64 — as §3A.5 demanded — so the two disagreed
precisely to the extent the reference was wrong, and the module being validated was the more
accurate of the pair.

**Fixed** by accumulating `global_rx` in float64. The equivalence then becomes essentially exact:

| scene | after |
|---|---|
| Indian Pines | **0.0** (bit-identical) |
| `abu-urban-3` | **0.0** |
| `abu-beach-1` | 1.2e-7 |

AUC impact on the 13 ABU scenes is ≤ 0.0007 — this corrects a bias, it does not manufacture
results.

**Why this one is worth recording even though the fix is one line.** The natural response to a
failing `rtol=1e-5` assertion is to loosen the tolerance until the test passes; the agent's first
instinct was to use `rtol=0.1` for ABU, which would have been green, defensible-looking, and
wrong. What made the difference was measuring *which side* was inaccurate before adjusting
anything. **A tolerance is a claim about two implementations, and when it fails the reference is a
suspect too** — here it was the culprit. The same reasoning applies to every remaining
cross-implementation check in this project, and there are several coming in Phase 5.

**Fifth consecutive finding from integration rather than unit tests** (D22, D22.1, D22.2, D23,
D24). `test_rx.py` passed throughout: it never compared `global_rx` against a higher-precision
reference, only against itself and synthetic fixtures.

---

### D22.3 — `crd`'s `lam=1e-2` is CORRECTLY sized. D22.2's suspicion was wrong, and the negative result closes the question rather than leaving it open.

D22.2 predicted `crd`'s `lam` would be the fourth instance of a transplanted regularization
constant, and said why the existing §3A.4 accept criterion could not settle it (`lam → ∞` tests
the regularizer's *direction*, not whether its default *magnitude* is usable). Measured on dense
30×30 crops of three real ABU scenes, chosen to retain enough positives for a meaningful AUC:

| scene | 1e-6 | 1e-4 | **1e-2** | 1e0 | 1e2 | 1e4 |
|---|---|---|---|---|---|---|
| `abu-airport-1` | 0.7485 | 0.8622 | **0.9574** | 0.9518 | 0.9650 | 0.9352 |
| `abu-beach-2` | 0.7981 | 0.9222 | **0.9762** | 0.9674 | 0.9679 | 0.9793 |
| `abu-beach-4` | 0.8350 | 0.9548 | **0.9949** | 0.9923 | 0.9976 | 0.9902 |

`lam=1e-2` sits **on the plateau**, not on a cliff: everything from 1e-2 to 1e4 is within ~0.03
AUC, while 1e-6 costs 0.15–0.21. The default is right.

It also passes the magnitude-response check that exposed `kernel_rx` — the diagnostic D22.2
recommended precisely because AUC alone would not have caught the kernel failure. Rank of an
implanted spike, real ABU crop, at the default `lam`:

| spike (σ) | 1 | 3 | 8 | 20 | 50 |
|---|---|---|---|---|---|
| rank (of 900) | 42 | 3 | **0** | **0** | **0** |

Rank **responds monotonically to magnitude and saturates at the top**, which is what a working
detector does. Contrast `kernel_rx` at `reg=1e-6`, where rank was pinned at 481 from 2σ to 50σ.

**Why `crd` escaped and the other three did not.** `lam` multiplies `Gamma^T Gamma`, where
`Gamma = diag(‖y - x_i‖₂)` is built from the data's own distances — so the regularizer **carries
the data's scale with it**, and `lam` is already dimensionless relative to the operator it
regularizes. `global_rx` and `kernel_rx` both added `reg * I`, an absolute term, to operators whose
scale they did not control. **The lesson generalizes better than "check every constant": a
regularizer built from the data is scale-safe by construction, one added as a bare identity is
not.** That is the property to look for in any future detector, and it is checkable by reading the
formula rather than by running a sweep.

---

### D25 — §3A.9's fusion accept criterion is NOT met on ABU, and the prescribed grid search does not rescue it. The criterion is also ambiguous in a way that matters. Measured 2026-08-22 on a held-out split.

D22.1 found the stated default weights losing to their best component. §3A.9's remedy is a grid
search, so it was run: **728 weightings**, selected on 5 recorded TUNE scenes, evaluated on 8
REPORT scenes never touched during selection (`experiments/rx_vs_ae/fusion_weights.json` holds
both lists — ABU is 13 scenes, so tuning and reporting on the same 13 would be train-on-test).

**Report split, scene-macro AUC:**

| `rx` (local) | `ace` | `spatial` | **fused (tuned)** | per-scene oracle |
|---|---|---|---|---|
| 0.9522 | 0.7740 | 0.9313 | **0.9540** | 0.9601 |

**The criterion is ambiguous.** "Fused AUC ≥ best single component on ≥ 10 of 13 ABU scenes" can
mean the best *fixed* component (one detector used everywhere) or the best component *per scene*.
The second uses the labels to pick a winner scene by scene — it is an **oracle, not an achievable
detector**, and requiring fusion to beat it is close to requiring fusion to be clairvoyant. The
fixed reading is the meaningful one and should be what the plan says.

**Under either reading, the criterion fails:**

| comparison | scenes won | required |
|---|---|---|
| vs best **fixed** component (`rx`) | 5/8 (62.5%) | 10/13 (76.9%) |
| vs per-scene **oracle** | 4/8 (50.0%) | 10/13 (76.9%) |

Fusion beats `rx` alone by **+0.0018 macro AUC**. That is noise, not a result.

**ACE contributes nothing, and the optimizer says so unprompted.** The best weighting assigns
`ace` a weight of **exactly 0.0**, and so do the next four. ACE's own macro is 0.7740 against
`rx`'s 0.9522, and on `abu-urban-4` it scores **0.5029** — indistinguishable from chance. The
likely cause is §3A.8's bootstrap: the target signature is the mean spectrum of the top 0.1% of
pixels *by the base detector*, so on a scene where those pixels are not a coherent material, the
"signature" is an average of unrelated spectra and ACE measures nothing. This is a property of
unsupervised signature estimation, not an implementation error — but it means the 4-component
design is really a 2-to-3-component one.

**What this changes about what may be claimed.** §13 rule 4 already forbids overclaiming for the
quantum branch; the same discipline applies here. **The report must not say fusion beats its
components.** The defensible statements are: fusion is *comparable* to the best single detector
(+0.002 macro, 5/8 scenes) while requiring no per-scene detector choice — which is a real
operational benefit, since the oracle column is unavailable at inference — and ACE as currently
bootstrapped does not earn its place. Reporting fusion as the headline detector on this evidence
would be exactly the kind of claim §13 exists to prevent.

**RESOLVED 2026-08-22 — the 4-component fusion was measured on HAD100 and it does not rescue the
result.** D25 originally caveated that everything above was the 3-component variant on ABU, where
`index` cannot run at all (D20), and that HAD100 — the one benchmark shipping real wavelengths —
was the only place the fusion the plan actually specifies could be evaluated. That run is done:
**94 HAD100 scenes, all 94 emitting `ace+index+rx+spatial`**, verified in `results.csv` rather
than assumed.

| HAD100 (94 scenes) | scene-macro ROC-AUC | PR-AUC macro |
|---|---|---|
| `kernel_rx` | **0.9713** | 0.6529 |
| `global_rx` | 0.9599 | 0.6119 |
| `crd` | 0.9583 | 0.5437 |
| **`fused` (4-component)** | **0.9359** | 0.4144 |
| `local_rx` | 0.9004 | 0.4499 |

Fusion ranks **4th of 5** with the full component set, on the dataset chosen specifically to give
it its best chance. The `index` component does not earn its weight. **The caveat is discharged
against fusion, not in its favour**, and the "do not claim fusion beats its components" rule in
this entry now applies without qualification rather than only to ABU.

Worth noting how close this came to being reported the other way. The harness had only ever built
`rx+ace+spatial`, so every fusion number in this project — including this entry's original
verdict — was the 3-component variant, and the 4-component detector the plan specifies had
**never once run**. The fix was written, asserted out on a stale anchor before writing, and the
run that followed produced entirely plausible HAD100 numbers from the unchanged code. It was
caught only by checking whether the edit had landed instead of trusting the output. **A plausible
number from code you believe you changed is the most dangerous artefact in this project** — it
looks exactly like a result.

---

### D26 — O4 CLOSED. QGIS verification done 2026-08-22 on QGIS 4.2.1: affine plumbing confirmed on Indian Pines, and **real georeferencing confirmed against an independent basemap** on HAD100 — the stronger check Phase 2 was never going to give.

**Check A — affine plumbing (§2.10's actual ask).** `qgis/projects/phase2_verify.qgz`,
Indian Pines, magma 0–1 raster + red ROI outlines labelled by `roi_id`. Raster renders with
structure; the 3 ROIs sit on high-score pixels. No offset, mirror, rotation or scale error.
**Phase 2 exit criterion is signed off.** Note this project deliberately carries **no basemap**:
Indian Pines georeferencing is synthetic (D2/D13.1), so real-world position is meaningless here
and a basemap would invite exactly the wrong conclusion.

**Check B — real georeferencing, and this is the one that matters.**
`qgis/projects/demo_verify.qgz`, HAD100 scene `ang20170821t183707_100`, EPSG:32611, against
OpenStreetMap. The scene lands on **Dogpound Creek, Alberta, Canada** — matching the transform of
the GeoTIFF's own bounds (−114.501…−114.499, 51.440…51.442) computed independently of QGIS. At
1:301 and 1:64 the red outlines trace the bright pixels **including the cross-shaped notches of
the mask boundary**, so the polygons follow the actual connected component rather than its
bounding box.

**Three independent lines of evidence now agree** on the same affine, which is what makes this
worth recording rather than just ticking:
1. the ENVI header's own UTM `map info`, parsed via GDAL (D14.2);
2. an OpenStreetMap basemap placing the scene on a named real-world feature;
3. **a derived GSD** — a 3-pixel ROI reports 26.5 m², implying ~3 m ground sample distance, which
   matches AVIRIS-NG. This one is the most useful of the three because it is independent of the
   basemap: a transform wrong by a scale factor would have to be wrong by a *plausible* factor to
   survive it.

**What this does NOT establish.** Phase 5 Level 2's accept criterion is "polygon centroids for a
**manually identified feature** land within 2 pixels (~2 GSD) of its true position." That needs a
target whose true position is independently known; the ROIs here are unlabelled anomalies over
scrubland with no ground truth to compare against. So: **scene-level placement is verified,
feature-level accuracy is not.** Level 2 stays open on that criterion, and no location-accuracy
number may be quoted from this check.

**Two QGIS-project bugs found and fixed getting here, both of which produced convincing false
signals.** They are recorded because the next person to regenerate these projects will hit them.

*The project had no CRS.* Every layer was individually correct — raster EPSG:32611, ROIs
EPSG:4326, OSM EPSG:3857 — and `QgsProject.crs()` was **empty**, so on-the-fly reprojection
misplaced the basemap. The raster rendered, the polygons rendered, and OSM painted a detailed,
plausible French town underneath, which reads as *"the georeferencing is broken"* rather than
*"the project has no CRS."* The arithmetic that settles it: read the scene's UTM easting/northing
(673818, 5701837) as **Web Mercator** metres and you get lon 6.05, lat 45.5 — Montmélian, France,
exactly where the tiles drew. The data was right the whole time.

*A `Null` default view extent* opened the project at scale 1:1 near the origin while the data sat
at easting ~500 000, giving a blank white canvas with both layers correctly loaded and styled —
again indistinguishable from a styling failure.

**And one import trap worth keeping.** §2.10 requires output under `qgis/projects/`, so this repo
contains a directory named `qgis/`, which Python imports as a PEP 420 **namespace package**,
shadowing the real PyQGIS. `import qgis` *succeeds*; only `qgis.__file__ is None` reveals it. That
check was run, passed, and was reported as "PyQGIS available in the venv" — it is not there at all
(it is an Arch system package under `/usr/lib/python3.14`). `scripts/build_qgis_project.py` now
strips the repo root from `sys.path` before importing and **must be run with system `python3`**.

The common thread across all three, and with D22/D24/D25 earlier the same day: **a check that
passes for the wrong reason is worse than one that fails.**

---

### D27 — Branch 3E built (2026-08-22). Seven of its nine design points are corrections to §6.5, and each is the same error §3E.2 already names: **comparing on unequal ground measures the ground, not the model.**

3E is the last unbuilt arm. Before writing anything, the pinned stack and the compute budget
were spiked, per the project's standard that documentation is wrong until opened. Six measured
facts came back, and they changed the design rather than confirming it.

**D27.0 — what the spike measured.**

1. The pinned trio composes: `qiskit 2.5.2` · `qiskit-aer 0.17.2` · `qiskit-machine-learning
   0.9.1`. Bell on `AerSimulator` at 4096 shots → `{'00': 2039, '11': 2057}`, inside 3σ
   (2048 ± 96). §3E.1 accepts.
2. **V1 primitives are gone.** `from qiskit.primitives import Sampler` raises `ImportError` on
   qiskit 2.x; only `StatevectorSampler` (V2) and `qiskit_aer.primitives.SamplerV2` exist.
   `qiskit_algorithms` is not installed at all — `COBYLA` comes from
   `qiskit_machine_learning.optimizers`. §6.5's "`AerSimulator` with a seeded `Sampler`" names
   an object that no longer exists.
3. **`ZZFeatureMap` / `RealAmplitudes` / `NLocal` classes are deprecated** (Qiskit 2.1+). The
   function forms `zz_feature_map()` / `real_amplitudes()` build the same circuits and emit no
   warning. §3E.2 and §3E.3 name the classes; the code uses the functions.
4. **`FidelityQuantumKernel` costs 28 ms/pair** at 8 qubits, `reps=2`, linear entanglement —
   measured N=50 → 35.3 s (1275 pairs), N=100 → 142.8 s (5050 pairs), clean O(N²). A 400-sample
   train Gram is ~38 min *before* the test Gram. This, not the VQC, is the branch's binding cost.
5. **An exact statevector Gram is ~180× faster and agrees to 1.1e-11.** Forming
   |⟨φ(x)|φ(y)⟩|² as one `Statevector` per sample then a single matrix product: N=40 → 0.122 s
   against `FidelityQuantumKernel`'s 22.4 s, `max|ΔG| = 1.13e-11`; a full N=600 Gram in 1.82 s.
   O(N) simulations instead of O(N²), same object.
6. **VQC costs 10.35 s per COBYLA objective evaluation** at N=600 / 8 qubits / 32 ansatz
   parameters (3.64 s at N=200), so `maxiter=200` is ~35 min. A trap in the measurement itself:
   qiskit's `maxiter` maps to scipy's `maxfun`, which is silently clamped to a minimum of
   `n_params + 2` — a `maxiter=10` smoke run actually performs 34 evaluations. Extrapolate from
   the clamped count or the estimate is off by 3.4×.

**D27.1 — split on flightline, not on patch.** HAD100's 94 test patches come from **only 10
AVIRIS-NG flightlines, one of which holds 38 of them (40%)**: `ang20170821t183707` ×38,
`ang20171012t194435` ×12, `ang20170908t225309` ×10, `ang20170821t195229` ×8,
`ang20210614t141018` ×7, `ang20191004t185054` ×6, `ang20170825t173426` ×5,
`ang20180227t082814` ×4, `ang20191004t215336` ×3, `ang20191027t204454` ×1. (There is no
`aviris_target/` directory — the test set is AVIRIS-NG only.) A patch-level split therefore
trains and tests on the same flightline. This never mattered before because **every detector
shipped so far is unsupervised and per-scene**; 3E is the first arm that trains, so it is the
first arm that can leak. The split is fixed, seeded and committed, and never revisited — the
discipline O6 imposes on the fusion tuning split. It is a fourth kind of leakage alongside
§12's three, and `test_data_hygiene.py`'s scene-group machinery does not cover it, because it
groups the *background pool*, not the test set.

**D27.2 — the classical AE arm is a dense AE on the same 8 features, not §3A.6's conv AE.**
§3E.4 says the QAE "mirrors 3A.6's autoencoder so the comparison is architecture-to-
architecture." §3A.6 does not exist (P3, `DEFERRED` in `run_benchmark.py`), and had it existed
the comparison would still have been wrong: a 15×15×30 spectral-**spatial** conv AE against an
8-dim vector QAE compares input representations, not architectures. §3A.6 stays unbuilt.

**D27.3 — VQC is supervised and every other arm is not.** §3E.6's five arms are four
unsupervised (classical AE, `OneClassSVM`-RBF, QAE, quantum kernel) and one supervised (VQC).
Ranking them in one column measures **supervision**, an effect larger than anything quantum.
Fixed two ways, both required: a `supervision` column in the table, and a supervised classical
partner (`SVC(rbf)` on identical features). Six arms, read as two families of three, never as
one ranking of six.

**D27.4 — wall-clock is simulator time, and at 8 qubits it is not a quantum measurement.**
D27.0's finding 5 is the proof, not an aside: a "quantum kernel" that a laptop reproduces
exactly, 180× faster, by multiplying two matrices. The table labels the column
`wall_clock_s (AerSimulator, CPU)` and the doc states it carries no QPU implication. Circuit
depth is reported **transpiled** to `basis_gates=["rz","sx","x","cx"]`, `optimization_level=1`,
both recorded in the results JSON — depth without a named basis is not comparable between arms.
Measured for reference: feature map depth 33, `{rz:46, cx:28, sx:16}`; ansatz depth 11.

**D27.5 — two files added to §3's layout.** `quantum/data.py` (split + sampling) and
`quantum/classical_baselines.py` (the three classical arms). Folding either into
`feature_map.py` would hide the leakage-critical split behind a name that does not say so.
Logged rather than silently added.

**D27.6 — 3E is HAD100-only, and this is forced rather than chosen.** `classical_reduce` must
reuse `reduce_bands`, which consumes `harmonize` output, which requires wavelengths. ABU and
HYDICE ship none (O8/D13.4). §13 rule 6 binds 3E exactly as it binds 3B: no ABU or HYDICE
quantum number may be produced or implied.

**D27.7 — train balanced, report at natural prevalence — and the honest version is also the
cheap one.** A 40-background/40-anomaly subsample is the right *training* set and the wrong
*reporting* set. PR-AUC is prevalence-dependent by construction, so a balanced PR-AUC set beside
`results_pooled.csv`'s HAD100 row (`kernel_rx` ROC-AUC 0.9713, natural prevalence, every pixel)
invites a comparison that is arithmetic nonsense — and someone will place them side by side,
because both are labelled "HAD100".

The obvious fix — score every pixel of all 28 test patches — is **not** cheap for the quantum
arms. HAD100 test scenes run 64x64 to 100x100 at ~0.4 % anomaly prevalence, so 28 patches is
~200 000 pixels, and VQC's `SamplerQNN` forward pass costs ~17 ms/sample (D27.0 finding 6):
about **an hour of scoring per quantum arm**, three times over.

The correct fix is cheaper *and* exact: score **all anomaly pixels plus a bounded random
background sample**, then pass `sample_weight` to `roc_auc_score` / `average_precision_score`
with the background weighted back up by `n_background_total / n_background_sampled`. ROC-AUC is
a ranking statistic and is prevalence-invariant either way; PR-AUC is not, and the weights
restore it exactly.

Measured on a 200 000-pixel synthetic population at 0.4 % prevalence, using 4 800 scored pixels
(a 42x saving):

| | ROC-AUC | PR-AUC |
|---|---|---|
| full population, 200 000 px | 0.9600 | **0.3395** |
| weighted subsample, 4 800 px | 0.9608 | **0.3259** |
| **unweighted** subsample, 4 800 px | 0.9608 | **0.8695** |

The unweighted PR-AUC reads **0.87 where the truth is 0.34** — it more than doubles, and it
does so in the direction that flatters the result. Nothing about that number looks wrong on
inspection. Same class as D20 and D22.1: a plausible number measuring something other than what
the reader assumes.

**D27.8 — §3E.8's novelty claim is false as written, and the search that found it is recorded
in `docs/experiments.md`.** The proposed claim was *"no existing published work applies VQC/QAE
feature encoding directly to hyperspectral anomaly detection."* A dated search (2026-08-22, five
verbatim queries, arXiv/IEEE/ResearchGate/IOPscience via a general index) found **arXiv
2605.04388**, *Hyperspectral Anomaly Detection Using Einstein Fuzzy Computing and Quantum Neural
Network* (Lin, Young, Langari; 6 May 2026), which fuses a quantum detector with classical ones
for exactly this task — and **arXiv 2605.17587**, *Large-Scale Quantum Kernels for Hyperspectral
Data Classification* (Delilbasic et al., 17 May 2026), which is larger-scale than anything here
on §3E.5's own method, though on classification rather than anomaly detection. A cybersecurity
QAE paper (arXiv 2510.21837) independently arrives at the same 8-feature / dense-angle /
`RealAmplitudes` configuration this branch chose. The claim narrows to one about **protocol** —
no work found evaluates VQC, SWAP-test QAE and fidelity quantum kernel side by side on
hyperspectral AD on a shared feature basis with supervision-matched classical baselines under a
leakage-controlled split. That is a smaller claim than §3E.8 imagined and one this branch's own
numbers can support. §13 rule 4 already said scoped novelty is not advantage; this is the first
time the scope was actually measured, and it moved.

**D27.9 — a third of the test features land outside the training range, and it is flightline
domain shift, not anomalousness.** Measured on the built split (589 train / 574 val / 600 test,
28 test scenes): with the MinMax fitted on train and clipped, **203 of 600 test rows (33.8 %)
have at least one feature saturated at 0 or π**, against 14 of 4 712 train values total — the
handful that define the range. Without the clip those pixels would encode as angles outside
[0, π], and for angle encoding that *wraps*: an extreme value silently maps onto a
normal-looking angle. Clipping saturates instead, which loses discrimination among extremes but
keeps them separable from the background. Clipping is the right choice and it is not free.

The obvious hypothesis — that the out-of-range pixels are the anomalies — is **wrong**, and
checking it was worth the two minutes. Clipped rows are 39.3 % of background against 28.3 % of
anomalies: the shift is *away* from anomalousness. The test flightlines are simply different
flights with a different radiometric range. That is D27.1's honest cost made visible: a
patch-level split would have shown near-zero clipping, because train and test would have shared
flightlines and therefore shared ranges. The saturation is not a defect introduced by the split;
it is the generalization gap the split stopped hiding.

**Consequence for reading the table.** Every 3E arm is being asked to generalize across
flightlines with a third of its test features at the encoding boundary. This depresses all six
arms and should depress them comparably, so the *comparison* stays fair — but it makes the
absolute numbers **not** comparable to `results_pooled.csv`'s classical detectors, which are
unsupervised, per-scene, and never generalize across anything at all. Two rows both labelled
"HAD100" measuring two different tasks; see D27.7 for the other half of that trap.

**The pattern, again.** Six of these sub-notes are cases where a specified comparison
would have produced a number that looked fine and measured the wrong thing — the feature basis
(§3E.2's own warning), the input representation (D27.2), the supervision regime (D27.3), the
simulator overhead (D27.4), the class prevalence (D27.7), or the flightline (D27.1). None would
have failed a test. That is now the branch's own restatement of the lesson D22/D24/D25 taught:
**a check that passes for the wrong reason is worse than one that fails.**

---

## 2. Frozen Contracts v1.0

Implemented in `core/contracts.py`. **No branch may redefine these locally.** Every contract has a validator, and every validator is called at every module boundary in debug mode.

### C1 — Preprocessed scene
```python
@dataclass(frozen=True)
class SceneMeta:
    scene_id: str                  # unique, stable, e.g. "abu_airport_1"
    crs: rasterio.crs.CRS          # native scene CRS
    transform: affine.Affine       # native pixel→world
    wavelengths: np.ndarray        # [B] float32, nanometres, strictly ascending
    bad_bands: np.ndarray          # [B] bool, True = excluded
    gsd_m: float                   # ground sample distance, metres
    source: str                    # "indian_pines" | "abu" | "hydice_urban_anomaly"
                                   # | "had100" | "enmap" | "sentinel2" | "aviris"
                                   # Closed enum, validated in core/contracts.py. NOT free
                                   # text: "hydice_urban" (no _anomaly) is REJECTED, because
                                   # that name belongs to the Copperas Cove unmixing scene
                                   # this project must never score against (D13.3).
    georef: str                    # "real" | "synthetic"   (D2)
    acquired: str | None           # ISO-8601 or None

# cube: np.ndarray [H, W, B], dtype float32, C-contiguous
#       nodata encoded as np.nan (NOT a sentinel value)
#       band order strictly ascending wavelength
```
Validator `validate_scene(cube, meta)` asserts dtype, ndim, band-count agreement, ascending wavelengths, and that nodata is NaN rather than a sentinel.

### C2 — Anomaly score raster
Two products, always written as a pair:
- `{scene_id}_anom_raw.tif` — single-band float32, **unbounded**, native CRS/transform. This is the primary scientific product.
- `{scene_id}_anom_norm.tif` — single-band float32 in [0,1], percentile-clipped per D3.

Required GeoTIFF tags on the `_norm` product:
```
NORM_METHOD=percentile_clip  NORM_P_LO=1.0  NORM_P_HI=99.9
NORM_V_LO=<float>  NORM_V_HI=<float>
SCORE_METHOD=<e.g. "local_rx">  SCENE_ID=<...>  GEOREF=real|synthetic
```

### C3 — Mask
`uint8`, `background=0`, `target=1`. No other values. NaN-nodata pixels are `0`, and a separate `_valid.tif` carries the validity mask.

### C4 — Change score raster
Identical to C2, named `{scene_id}_change_{raw,norm}.tif`, co-registered to the **t1** grid, tag `T1_SCENE_ID`, `T2_SCENE_ID`, `REG_RMSE_PX`.

### C5 — ROI record (in-memory, Stage 4 → Stage 6)
```python
@dataclass
class ROIRecord:
    roi_id: str                    # "{scene_id}:{branch}:{index:04d}"
    source_branch: str             # "anomaly" | "change" | "fused"        (D5)
    target_profile: str            # "object" | "landcover"                (D5)
    bbox: tuple[int, int, int, int]   # row0, col0, row1, col1 (pixel, t1 grid)
    mask: np.ndarray               # uint8 [h, w], C3 convention, bbox-local
    anomaly_score: float | None    # mean normalized, inside mask
    change_score: float | None
    seg_prob: float | None
    clear_fraction: float | None
    shape_plausibility: float | None
    linked_roi_ids: list[str]      # cross-profile overlaps, not merged     (D5)
    parent_roi_ids: list[str]      # populated when source_branch=="fused"
```

### C6 — GeoJSON output
Roadmap §2 schema, **plus** the D5 amendment fields. Written via `geopandas`, `EPSG:4326`, RFC 7946.

```jsonc
{
  "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [...] },   // EPSG:4326
  "properties": {
    // --- Roadmap §2, unchanged ---
    "lat": 40.0123, "lon": -86.5432,
    "area": 4820.5,             // m², computed in EPSG:6933 (equal-area)
    "perimeter": 312.7,         // m, GEODESIC (pyproj.Geod) — not from EPSG:6933
    "anomaly_score": 0.87,      // normalized, mean inside polygon
    "change_score": null,
    "confidence": 0.79,         // D4
    "timestamp": "2026-08-20T11:04:32Z",
    "source_scene": "abu_airport_1",
    "class": "UNKNOWN",         // default per Roadmap §1.6

    // --- D5 amendment ---
    "roi_id": "abu_airport_1:anomaly:0007",
    "source_branch": "anomaly",
    "target_profile": "object",
    "linked_roi_ids": [],
    "confidence_components": ["c_anom"],
    "georef": "synthetic"
  }
}
```
`area` is computed in an equal-area projected CRS (`EPSG:6933`). `perimeter` is **not** — EPSG:6933 is Lambert cylindrical equal-area, which preserves area by construction and distorts length; at Indian latitudes its north-south/east-west scale asymmetry biases perimeter measurably. Perimeter is computed **geodesically** via `pyproj.Geod(ellps="WGS84").geometry_length`. Neither is ever computed in degrees. Mixing these up is the most common georeferencing bug in this class of pipeline.

### C7 — CRS convention
Native scene CRS is carried through every stage. Reprojection to `EPSG:4326` happens **only** in `geospatial/geojson.py`, at export. Any other module calling `to_crs` is a bug.

### C8 — Dependency lock
`requirements.txt` generated by `uv pip compile` against Python 3.12, committed with full transitive pins and hashes. Regenerated only by team-wide sync (Roadmap §9.11).

---

## 3. Repository Layout

```
sih/
├── configs/                 target_profile.yaml · pipeline.yaml · paths.yaml
├── core/                    contracts.py · io.py · logging.py
├── data/
│   ├── raw/                 downloaded, never modified
│   ├── processed/           harmonized cubes + score rasters
│   └── benchmark/
│       ├── indian_pines/    Phase 2 wiring ONLY — no anomaly GT (D2)
│       ├── abu/             13 scenes, real pixel masks
│       ├── hydice_urban_anomaly/   1 scene 80×100×175, 21 anom px (D13.3)
│       ├── had100/          94 raw test + 522 raw background ENVI scenes;
│       │                    unpacked by main.py -> 100 test + 2088 bg patches (D11)
│       └── synthetic/       generated — TRAINING ONLY, never scored (D7)
├── preprocessing/           raster_loader · normalize · harmonize · bands
│                            cloud_mask · registration
├── anomaly/                 rx · local_rx · kernel_rx · crd · streaming_rx
│                            autoencoder · deep_detector · scoring · fusion
├── change_detection/        temporal_difference · spectral_angle · physics_fusion
│                            siamese_net · temporal_baseline
├── segmentation/            synth · datasets · train_unet · train_alt_arch
│                            infer · postfilter
├── geospatial/              polygonize · geojson · projections · roi_fusion
├── edge/                    benchmark · onnx_inference · roi_pipeline · streaming
│                            quantization · profiling · constrained_sim
├── quantum/                 qiskit_basics · feature_map · vqc_encoder
│                            quantum_autoencoder · quantum_kernel · classical_vs_quantum
├── pipeline/                run_pipeline.py
├── qgis/                    styles/ · projects/
├── experiments/             rx_vs_ae/ · seg_arch/ · change_arms/ · edge_benchmarks/
│                            quantum_results/ · cascade_recall_audit/
├── scripts/                 fetch_*.py · verify_had100.py · verify_benchmarks.py
├── tests/
├── docs/                    architecture · datasets · experiments · validation
├── requirements.txt
└── pyproject.toml
```

---

## 4. Phase 1 — Environment & Repo Bootstrap

**Fully specified and startable now. Nothing here depends on any open question.**

### 1.1 Create the environment
```bash
cd /home/mayaskara/projects/sih
uv venv --python 3.12 .venv
source .venv/bin/activate
```
`uv` resolves `--python 3.12` to **3.12.13** and downloads it if the system has no
3.12 — verified on a machine whose only system interpreter was 3.14.7. D1's pin
therefore costs nothing to honour; nobody needs a system Python 3.12 installed.

**There is exactly one project environment: `.venv`.** A second venv exists at
`.tooling/venv` — it was created to run the HAD100 verification (D11) *before*
Phase 1 had produced a lockfile, so it is deliberately unlocked and holds only
`gdown numpy scipy h5py`. It is throwaway provenance tooling, not a parallel
runtime. **No module under `preprocessing/`, `anomaly/`, `segmentation/`,
`geospatial/`, `edge/` or `quantum/` may ever be run from it**, or C8's dependency
lock stops meaning anything. Delete it once `scripts/verify_had100.py` has been
run from `.venv`.

### 1.1b Credentials — configured once, outside the repo

EnMAP (DLR Geoservice / EOC UMS) and Sentinel-2 (Copernicus) need accounts. Those accounts are held by the
project owner; **the credentials are not in this repository and must never be.**

```
~/.config/sih/credentials.env     mode 600, OUTSIDE the repo   <- real values
.env.example                      committed                    <- variable NAMES only
core/credentials.py               get() · require() · status()
scripts/check_credentials.py      prints configured/not-configured, never values
```

**Why outside the repo rather than a gitignored `.env`.** `.gitignore` stops `git add`. It does
not stop an agent globbing the working tree, a `tar`/`rsync` of the project folder, a `git add
-f`, or an editor backup file. Keeping the file outside the tree removes the entire class rather
than one member of it. `core/credentials.py` additionally refuses to read the file if its mode is
group- or world-readable.

**Copernicus: S3 keys, not the account password, and not Sentinel Hub OAuth.** Verified against
the live services on 2026-08-21:

| Leg | Endpoint | Auth |
|---|---|---|
| Catalogue **search** | `catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=…` | **none** — HTTP 200 unauthenticated |
| Product **download** | redirects to `download.dataspace.copernicus.eu/…/$value` | required — HTTP 401 without |

Only the download leg needs credentials, and S3 covers it. Generate a key pair at
`https://eodata-s3keysmanager.dataspace.copernicus.eu/` → *Add Credentials* (verified live, HTTP
200); the secret is displayed once. Endpoint `https://eodata.dataspace.copernicus.eu`, bucket
`s3://eodata`. S3 keys are revocable without touching the account, so
**`CDSE_USERNAME`/`CDSE_PASSWORD` are deliberately absent** — CDSE's own documentation warns
against storing the account password, and nothing in this pipeline needs it.

> **Not Sentinel Hub.** Earlier drafts of this section pointed at the Sentinel Hub dashboard's
> OAuth *client credentials* flow. That dashboard was **sunset 2026-03-20**, and
> `documentation.dataspace.copernicus.eu` still described it months afterwards — a live instance
> of the §15 rule that documentation is wrong until checked. It was also the wrong service:
> Sentinel Hub's Process API returns rendered imagery, whereas §5.3 needs SAFE products with the
> 20 m SCL band for cloud masking (§9.4) and the 10/20/60 m grid mixing recorded in §15.

EnMAP uses `DLR_USERNAME`/`DLR_PASSWORD` against EOC UMS SSO — a CAS ticket flow with **no API token**; see §8.0a for the probe that established this, and O11 for the entitlement risk that is *not* yet discharged.

**Failure mode to avoid:** a fetcher that runs without credentials and fails at the HTTP layer
with `401 Unauthorized`. `require()` fails at startup instead, naming the missing *variable* and
how to obtain it — never the value, including in exception text.

**Agent rule (also in `CLAUDE.md`):** never print the credentials file. Reading it into a message
leaks it into a transcript that outlives the session, and nothing needs to — `require()` loads it
and `check_credentials.py` reports booleans.

### 1.2 Author `requirements.in`, compile the lock
```
numpy · scipy · scikit-learn · rasterio · geopandas · shapely · fiona · pyproj
torch · torchvision · onnx · onnxruntime · opencv-python-headless · scikit-image
matplotlib · qiskit · qiskit-aer · qiskit-machine-learning · h5py
pyyaml · tqdm · pandas · psutil · spectral · pytest · pytest-cov · ruff
```
```bash
uv pip compile requirements.in --python-version 3.12 -o requirements.txt --generate-hashes
uv pip sync requirements.txt
```

### 1.3 Acceptance test — `tests/test_env.py`
Asserts `sys.version_info[:2] == (3, 12)`, that every module in the list above imports, and that `rasterio.__gdal_version__ >= "3.8"`. Prints the full version table into `docs/datasets.md`.

### 1.4 Scaffold — `scripts/init_repo.py`
Creates the §3 tree, writes `__init__.py` in every package, writes the three `configs/*.yaml` with the D6 defaults, and adds a `.gitignore` excluding `data/raw`, `data/processed`, `data/benchmark`, `*.tif`, `*.mat`, `.venv`.

### 1.5 `core/contracts.py` — write this before any other module
Implements C1–C6 as dataclasses plus:
```python
def validate_scene(cube: np.ndarray, meta: SceneMeta) -> None
def validate_score_raster(path: str | Path) -> None
def validate_mask(mask: np.ndarray) -> None
def validate_roi(roi: ROIRecord) -> None
def validate_geojson(path: str | Path) -> None   # every C6 field present + typed
```
Each raises `ContractViolation` with the offending field named. **Accept:** `tests/test_contracts.py` covers one passing and one failing case per validator.

### 1.6 Dataset fetchers — `scripts/`
| Script | Source | Notes |
|---|---|---|
| `fetch_indian_pines.py` | `huggingface.co/datasets/danaroth/indian_pines` (`Indian_pines_corrected.mat`, `Indian_pines_gt.mat`) — the ehu.eus/ccwintco URLs now return HTML error pages | 145×145×200 uint16 + 145×145 uint8 GT. **No CRS, no wavelengths** (D13.1) |
| `fetch_abu.py` | `xudongkang.weebly.com/uploads/1/6/4/6/16465750/abu-{airport-1..4,beach-1..4,urban-1..5}.mat` | 13 scenes, no auth. **Seven distinct band counts, three dtypes** — never assume 205 (D13.2) |
| `fetch_hydice.py` | `github.com/sxt1996/HYDICE` → `HYDICE-urban.mat` + its `README.md` (keep both; the README is the provenance) | 1 scene, **80×100×175 float64, 21 anomaly px in 10 components**. Despite the filename this is the *Michigan anomaly* scene, **not** the 307×307×162 Copperas Cove unmixing scene also called "HYDICE Urban" — D13.3 |
| `fetch_had100.py` | Google Drive `10Z6f-8ELatxdAi31IDg2Mgbl92_4EnCJ` → `HAD100.zip` (3 754 846 290 B, sha256 `6f91035…27c19f`), then run the repo's **own** `main.py` to unpack — do not re-implement its cropping (D11.2) | **Raw:** 94 test + 260 NG / 262 Classic background ENVI scenes. **Unpacked:** 100 test patches 64×64×276 + 1 040 NG / 1 048 Classic background patches 64×64. Verify with `scripts/verify_had100.py` (D11). |
| `fetch_enmap.py` | DLR Geoservice STAC → `download.geoservice.dlr.de` (§8.0a) | background pool extension; 5 000-product download cap |
| `fetch_aviris.py` | NASA AVIRIS/AVIRIS-NG open portal | Phase 5 L2 |
| `fetch_sentinel2.py` | Copernicus Data Space (account held) | Phase 5 L3 **only** — never the background pool (D7 / Roadmap §9.1) |
| `fetch_speclib.py` | USGS splib07 (`doi:10.5066/F7RR1WDJ`), ECOSTRESS/ASTER | target endmembers for D7 implantation. **Built, but cannot fetch — BOTH sources are human-gated (D21).** splib07's 5.48 GB archive is `__s3__`-staged behind a download-request page and its advertised URI answers a Range request with HTTP 206 `text/html`; ECOSTRESS is a request form. `--check` re-tests the gate and exits 2 if it opens. This blocks `unet_implanted_lib`, i.e. half of §3B.8's headline comparison. |

Each fetcher verifies a SHA256 manifest and writes a provenance record into `docs/datasets.md` (URL, date, size, license, citation). **HAD100 must be cited as Li et al., IEEE TGRS 2023, doi:10.1109/TGRS.2023.3258067.**

**Note:** HAD100 ships its own dedicated anomaly-free background sets. Use those as the primary background pool. EnMAP L2A is the *extension* for sensor diversity, not the starting point.

### 1.7 Phase 1 exit criterion
`pytest tests/test_env.py tests/test_contracts.py` green, and `python -m scripts.fetch_indian_pines && python -m preprocessing.raster_loader --preview data/benchmark/indian_pines` writes an RGB composite PNG and a single-pixel spectral-signature plot.

---

## 5. Phase 2 — Walking Skeleton

**Serial. One owner. Nothing in Phase 3 starts until this is green.** Everything else in the system attaches to this spine, so it is specified to the function.

### 2.1 `preprocessing/raster_loader.py`
```python
def load_scene(path: str | Path, *, source: str) -> tuple[np.ndarray, SceneMeta]:
    """Dispatch on extension. Returns C1-compliant (cube [H,W,B] float32, meta).

    .mat  → scipy.io.loadmat; NO georeferencing and NO wavelengths exist in any
            .mat we use (D13.1/D13.4), so a synthetic affine is attached per D2,
            meta.georef == "synthetic" and meta.wavelengths is None.
    .tif  → rasterio; real CRS/transform read from the file, meta.georef == "real".
    .hdr  → spectral.envi; wavelengths AND real UTM/WGS-84 map info parsed from
            the header, so meta.georef == "real" (D11.5 — HAD100 ENVI scenes are
            genuinely georeferenced; do not assume benchmark data never is).

    DTYPE IS READ, NEVER ASSUMED. ABU alone ships int16 (8 scenes), uint16 (4)
    and float64 (1) -- see cast_to_float32 below. meta records the source dtype
    per scene so a downstream surprise is traceable to the file, not to a guess.
    """

def cast_to_float32(raw: np.ndarray, *, source_dtype: np.dtype) -> np.ndarray:
    """Explicit, dtype-aware widening to float32. NEVER `raw.astype(np.float32)`
       on an array whose dtype was assumed.

       The real hazard is reading SIGNED data as UNSIGNED. Verified in ABU: 8 of
       13 scenes are int16 and genuinely contain small negatives (min -50 .. -1,
       up to 45 626 negative pixels in abu-urban-2) -- normal residuals after
       dark-current/atmospheric correction. Reinterpreted as uint16, -1 becomes
       65535: a value 3x the scene maximum, at tens of thousands of pixels, fed
       straight into a detector whose entire job is to flag extreme values. RX
       would light up on decoding artefacts and the output would look like a
       working anomaly map.

       The reverse (a genuine DN above 32767 read as int16 and wrapping negative)
       does NOT occur in the current benchmark set -- max observed is 19 492 --
       but is asserted against anyway, because nothing guarantees the next
       dataset is as well behaved.

           if raw.dtype not in (np.int16, np.uint16, np.float32, np.float64):
               raise ValueError(f"unhandled source dtype {raw.dtype}")
           assert raw.dtype == source_dtype, "dtype changed between header and read"
    """

def save_score_raster(score: np.ndarray, meta: SceneMeta, out_path: Path,
                      *, method: str, normalize: bool = True) -> tuple[Path, Path]:
    """Writes the C2 raw/norm pair with all required tags. Returns both paths."""

SYNTHETIC_AFFINE_ORIGIN = (500_000.0, 4_480_000.0)   # UTM 16N, NW Indiana
SYNTHETIC_CRS = "EPSG:32616"
SYNTHETIC_GSD_M = 20.0
```
**Accept:** loading Indian Pines yields `(145, 145, 200) float32`, `meta.wavelengths is None` (the file ships none — D13.1; the earlier "200 ascending wavelengths" criterion here was unsatisfiable) and `georef == "synthetic"`; loading a HAD100 `.hdr` yields `georef == "real"` with 425 or 224 parsed wavelengths; each ABU scene reports its true source dtype and the int16 scenes retain their negative values after `cast_to_float32`; `validate_scene` passes; a round-trip `save_score_raster` → `rasterio.open` recovers `NORM_V_LO`/`NORM_V_HI` to within float32 epsilon.

### 2.2 `preprocessing/normalize.py`
```python
def drop_bad_bands(cube, meta) -> tuple[np.ndarray, SceneMeta]
    # Indian Pines: the 20 standard water-absorption bands [104-108,150-163,220]
def l2_normalize(cube) -> np.ndarray          # per-pixel spectral L2, brightness-invariant
def standardize(cube) -> np.ndarray           # per-band zero-mean unit-var, NaN-safe
```
**Accept:** NaN-safe on a cube with 5 % NaN pixels; no band's output variance is 0.

### 2.3 `anomaly/rx.py`
```python
def global_rx(cube: np.ndarray, *, reg: float = 1e-6) -> np.ndarray:
    """Global Reed-Xiaoli detector.
       r(x) = (x - mu)^T @ inv(Sigma + reg*I) @ (x - mu)
       mu, Sigma estimated over ALL valid pixels.
       Returns [H, W] float32, NaN where the input pixel was NaN.
       Uses scipy.linalg.cho_factor / cho_solve — never an explicit inverse.
    """
```
**Accept — a correctness gate, not an accuracy gate.** Phase 2 is the serial critical path and Indian Pines has no anomaly ground truth (D2), so no AUC target is set here:
1. Matches a reference Mahalanobis implementation (`scipy.spatial.distance.mahalanobis` over all pixels) to `rtol=1e-6`.
2. On a fixture built by implanting one high-abundance (`a=1.0`) target into a clean patch, the implanted pixels rank in the top 0.1 % of scores.
3. Output is NaN exactly where the input pixel was NaN.
4. Runs on 145×145×200 in < 5 s single-core.

**No Phase 2 acceptance criterion uses Indian Pines class labels as an anomaly positive set.** That is the rare-class pseudo-label approach that was explicitly not chosen, and Indian Pines — large contiguous agricultural fields, not point targets — is poor at it, so an AUC gate here would send someone debugging correct code on the critical path. Detector accuracy is measured in 3A, on ABU/HYDICE/HAD100, where real anomaly masks exist.

### 2.4 `anomaly/scoring.py` (D3 — write this early, everything uses it)
```python
def percentile_normalize(score, *, p_lo=1.0, p_hi=99.9, valid=None
                         ) -> tuple[np.ndarray, float, float]
def rank_normalize(score, *, valid=None) -> np.ndarray
def threshold_by_percentile(norm_score, *, pct: float) -> np.ndarray   # → C3 uint8
```
**Accept:** `percentile_normalize` output is in [0,1] and NaN-preserving; inverting with the returned `v_lo/v_hi` recovers the clipped input exactly; `rank_normalize` on 10 000 distinct values yields a uniform histogram (KS test p > 0.05).

### 2.5 `segmentation/postfilter.py` (Phase 2 uses morphology only)
```python
def morphological_cleanup(mask, *, open_radius=1, close_radius=2) -> np.ndarray
def connected_components(mask, *, connectivity=8) -> tuple[np.ndarray, int]
def shape_plausibility(region_props, profile: dict) -> float   # → [0,1]; Phase 3B
def filter_rois(rois: list[ROIRecord], profile: dict) -> list[ROIRecord]   # Phase 3B
```
**Accept (Phase 2 subset):** a synthetic mask with a 1-px salt-noise field and one 30-px blob leaves exactly one component after cleanup.

### 2.6 `geospatial/polygonize.py`
```python
def mask_to_rois(mask, meta, *, source_branch: str, target_profile: str
                 ) -> list[ROIRecord]
    # scene_id comes from meta.scene_id — never passed separately, or the two
    # copies diverge and roi_id stops matching source_scene.
    """Connected components → C5 ROIRecords. source_branch and target_profile are
       set HERE, at ROI birth, per D5. They are never inferred downstream."""

def rois_to_polygons(rois, meta) -> list[shapely.Polygon]
    """rasterio.features.shapes on each ROI mask, pixel→native-CRS via meta.transform.
       Holes preserved. Polygons are NOT reprojected here (C7)."""
```
**Accept:** a mask with a single 10×10 pixel square at a known offset produces one polygon whose native-CRS bounds equal `meta.transform * (row, col)` to within 1e-6.

### 2.7 `geospatial/projections.py`
```python
def to_wgs84(geoms, src_crs) -> list[shapely.geometry.BaseGeometry]
def area_m2(geom, src_crs) -> float
    """EPSG:6933 (Lambert cylindrical equal-area). NEVER degrees — see C6."""
def perimeter_m(geom, src_crs) -> float
    """GEODESIC: pyproj.Geod(ellps="WGS84").geometry_length on the WGS84 geometry.
       NOT EPSG:6933 — that projection preserves area, not length, and its scale
       asymmetry biases perimeter at Indian latitudes (C6)."""
def centroid_latlon(geom, src_crs) -> tuple[float, float]
```
**Accept — both quantities tested separately**, since an area-only test passes with perimeter broken:
- a 1 km × 1 km square in UTM reports **area** within 0.5 % of 1e6 m² after the EPSG:6933 round-trip;
- the same square reports **perimeter** within 0.5 % of 4000 m geodesically, tested at both 8°N and 35°N (the span of Indian latitudes) to catch equal-area length distortion if anyone reintroduces it.

### 2.8 `geospatial/geojson.py`
```python
def rois_to_geojson(rois: list[ROIRecord], meta: SceneMeta, out_path: Path,
                    *, timestamp: str | None = None) -> Path:
    """The ONLY place EPSG:4326 reprojection happens (C7).
       Emits every C6 field including the D5 amendment fields.
       Computes `confidence` via D4 over whatever components are non-None,
       and records which ones in `confidence_components`."""

def compute_confidence(roi: ROIRecord) -> tuple[float, list[str]]   # D4
```
**Accept:** `validate_geojson` passes; every feature has all 16 C6 properties; `class == "UNKNOWN"`; loading with `geopandas.read_file` gives `crs == EPSG:4326`.

### 2.9 `pipeline/run_pipeline.py`
```bash
python -m pipeline.run_pipeline \
  --scene data/benchmark/indian_pines/indian_pines.mat \
  --source indian_pines \
  --detector global_rx \
  --threshold-pct 99.0 \
  --profile object \
  --out experiments/phase2/
```
Stages, each behind a config-selected implementation so Phase 4 swaps are a config edit and not a rewrite:
`load → drop_bad_bands → standardize → detector → percentile_normalize → threshold → morphology → connected components → mask_to_rois → rois_to_polygons → geojson`

Writes `_anom_raw.tif`, `_anom_norm.tif`, `_mask.tif`, `_rois.geojson`, and `run_manifest.json` (config hash, git SHA, timings per stage, package versions).

### 2.10 QGIS verification
`qgis/projects/phase2_verify.qgz` loads the RGB composite + `_anom_norm.tif` (magma, 0–1) + `_rois.geojson` (red outline, no fill, labelled by `roi_id`).

**What this verifies:** polygons land correctly *relative to the raster* — i.e. the affine plumbing is right. **It does not verify real-world geographic accuracy**, because Indian Pines' georeferencing is synthetic (D2). Real georeferencing is verified first in 3A.1 on HAD100's real UTM headers (D11.5), then independently in Phase 5 Level 2.

### Phase 2 exit criterion
A GeoJSON with correctly-placed polygons, produced end to end by one command, passing `validate_geojson`, visually confirmed in QGIS, with a laptop timing baseline recorded in `run_manifest.json`. **This is the spine. Phase 3 does not start before this is green.**

---

## 6. Phase 3 — Parallel Branch Build-Out

All five branches start only after Phase 2 is green, and all build against §2 contracts via `core/contracts.py`.

---

### 6.1 — Branch 3A · Anomaly Detection (Person A + B)

#### 3A.1 `preprocessing/harmonize.py` (D9 — blocks 3A.6, 3A.7, and all of 3B)
```python
CANONICAL_WL   = np.arange(400, 2501, 10, dtype=np.float32)   # 211  (verified)
WATER_WINDOWS  = [(1350, 1450), (1800, 1950)]                 # endpoints INCLUSIVE
RETAINED_BANDS = 184                                          # 211 - 27  (verified)

def water_mask(wl=CANONICAL_WL) -> np.ndarray:
    """True where the band is inside a water-absorption window. Endpoints are
       INCLUSIVE — 1350 and 1450 nm are inside the feature, not clean shoulders.
       Exact equality is safe: every canonical wavelength is exactly
       representable in float32 (verified). See D9."""
    m = np.zeros(wl.shape, bool)
    for lo, hi in WATER_WINDOWS:
        m |= (wl >= lo) & (wl <= hi)
    return m

def sort_spectral_axis(cube, wl) -> tuple[np.ndarray, np.ndarray]:
    """Return (cube, wl) with the band axis sorted by ASCENDING wavelength and
       duplicate wavelengths collapsed by mean.

       MANDATORY before any interpolation. AVIRIS's four spectrometers overlap,
       so band index order != wavelength order: 262/262 HAD100 aviris_normal
       headers descend at three seams (D11.4). np.interp requires ascending xp
       and DOES NOT CHECK -- it returns silent garbage across those seams. The
       cube keeps its shape and passes every assertion while being wrong in
       three spectral regions.

           order = np.argsort(wl, kind="stable")
           ... collapse near-duplicates ...
           assert np.all(np.diff(wl_out) > 0), "spectral axis not strictly increasing"
    """

def harmonize(cube, meta, *, target_wl=CANONICAL_WL) -> tuple[np.ndarray, SceneMeta]:
    """sort_spectral_axis -> linear interpolation onto the canonical grid ->
       drop water windows (water_mask, endpoints inclusive) -> RETAINED_BANDS.

       Bands outside the sensor's range are NaN-filled and flagged in
       meta.bad_bands. No-data sentinels are read from the file header, never
       hardcoded: HAD100 alone uses both -9999.0 and 1e-34 (D11.5), and an
       unmasked 1e-34 reads as near-zero radiance and biases every covariance.

       This is also the JOIN between sensors. AVIRIS-NG reduces to 276 bands and
       AVIRIS-Classic to 162 (D11.3), so pooling the two background halves is
       only legal AFTER this runs -- never before.

       SOURCE IS THE RAW ENVI CUBE, never main.py's band-selected .npy (D11.6):
       band_select leaves interior holes up to 276 nm wide, and np.interp bridges
       them with a straight line instead of raising. Raw covers all 184 retained
       canonical bands with zero gaps for both sensors -- verified.

           assert coverage_ok(wl_src, CANONICAL_WL[~water_mask()]), \
               "canonical band with no source within one sensor step"
    """

def coverage_ok(wl_src, wl_target, *, tol=None) -> bool:
    """Every target wavelength has a source band within one median source step.
       False means interpolation would fabricate data across a hole (D11.6)."""

def reduce_bands(cube, *, n_components=30, method="pca", fit_on=None
                 ) -> tuple[np.ndarray, object]:
    """method in {"pca", "kpca"}. Returns [H,W,C] and the fitted transformer.
       The transformer is pickled to data/processed/ so train and inference
       provably share one basis. Refitting at inference is a bug.

       fit_on MUST be the TRAIN split of the background pool only, never the
       whole pool and never a scoring scene (ABU/HYDICE/HAD100-test). Fitting
       on the whole pool leaks test-scene spectra into the representation
       itself -- a subtler leak than a scene appearing in a training manifest,
       and the one §3B.5b's LODO matrix exists to prevent. This is why
       reduce_bands is called from §3B.3 (`segmentation/datasets.py`), which is
       where the train/eval split already exists as a hard boundary
       (`RealSegDataset` refuses `split="train"`), not from harmonize() itself,
       which runs per-scene before any split is decided. See the 3B.3 Accept
       criterion below for the C=30 numbers this used to claim here."""
```
**Accept:** HAD100 NG (raw 425) and HAD100 Classic (raw 224) both harmonize to `shape[-1] ==
RETAINED_BANDS == 184` and stack into one array — this is the D11.3 join: the two background
pool halves become one homogeneous tensor only after this runs. **This criterion is
intentionally narrower than earlier drafts of this line**, which additionally claimed a shared
`C=30` PCA/kPCA tensor here; that claim is now §3B.3's, because `reduce_bands`'s fit **must**
happen behind the train/eval split (see `reduce_bands`'s docstring above) and harmonize.py runs
per-scene, before any split exists to fit behind. **Plus:** feeding a raw AVIRIS-Classic header's
wavelength array to `sort_spectral_axis` yields a strictly increasing axis, and calling
`harmonize` on an unsorted axis raises rather than interpolating (D11.4). **Plus:** `coverage_ok`
returns True for both raw sensors (0/184 uncovered) and False for a `band_select`-style gapped
input, which then raises (D11.6) — **this is the self-defending property**: feeding harmonize
band-selected input must raise, never emit a plausible-looking but partly-fabricated cube.

**Accept — first REAL georeference check (D11.5, moved forward from Phase 5 Level 2).** HAD100 ENVI headers carry genuine UTM/WGS-84 `map info`. Load one NG scene (2.3 m GSD) and one Classic scene (17 m), run the full `pixel → world → EPSG:4326 → GeoJSON` path from §2.6–2.8, and confirm the ROI centroid falls within one pixel of the coordinate computed independently from the header's `map info` origin and pixel size. This is the first time real coordinates are exercised anywhere in the plan; Indian Pines cannot do it (D2) and Phase 5 Level 2 is now the first *independent* check rather than the first check.

#### 3A.2 `anomaly/local_rx.py` — the real baseline
```python
def local_rx(cube: np.ndarray, *, inner: int = 5, outer: int = 21,
             reg: float = 1e-4, n_components: int = 20) -> np.ndarray:
    """Dual concentric window RX. Background mu/Sigma come from the ANNULUS ONLY
       (outer window minus inner guard window), estimated per pixel.

       Band reduction to n_components runs FIRST and is MANDATORY, not optional:
       with B=200 an annulus of (21²-5²)=416 samples gives a rank-deficient
       covariance, and the detector silently degenerates.

       assert outer > inner and (outer**2 - inner**2) > n_components * 2

       Implementation: integral images over the outer and inner windows give
       running sums and co-moment sums in O(H*W*C²) total rather than per-pixel
       re-accumulation. Edge pixels use the truncated annulus with a sample-count
       guard; pixels with fewer than n_components*2 valid neighbours return NaN.
    """
```
**Constraint from the data:** HAD100 **unpacked patches** are 64×64 (D11.2 — the *raw* scenes are not; they range to 120×120 across 8 distinct shapes). `outer=21` is already a third of a 64-px patch. For the unpacked HAD100 patches use `outer=15, inner=3, n_components=12`, set per-dataset in `configs/pipeline.yaml`, never hardcoded. Do **not** carry `outer=15` over to raw scenes — at 120×120 it needlessly starves the annulus.

**Accept:** ROC-AUC on ABU-Airport-1 ≥ global RX on the same scene; the assertion fires on `outer=7, inner=5, n_components=20`; runs on 145×145×200 in < 60 s single-core.

#### 3A.3 `anomaly/kernel_rx.py`
```python
def kernel_rx(cube, *, gamma: float | None = None, n_background: int = 2000,
              reg: float = 1e-6, seed: int = 0) -> np.ndarray:
    """RBF-kernel RX in feature space. O(N²) in background samples, so the
       background is a fixed random subsample of n_background pixels (seeded —
       an unseeded subsample makes the reported AUC irreproducible).
       gamma=None → median-heuristic: 1 / (2 * median(pdist(bg)²)).
    """
```
**Accept:** deterministic across runs at fixed seed; AUC on ABU-Beach-2 reported alongside global and local RX.

#### 3A.4 `anomaly/crd.py` — Collaborative Representation Detector
```python
def crd(cube, *, inner: int = 5, outer: int = 15, lam: float = 1e-2) -> np.ndarray:
    """Represent each pixel as a linear combination of its annulus neighbours
       under a Tikhonov-weighted constraint:
           w = (X^T X + lam * Gamma^T Gamma)^{-1} X^T y
       where Gamma = diag(||y - x_i||_2) biases against dissimilar neighbours.
       Score = ||y - X w||_2  (large residual = poorly represented = anomalous).
    """
```
**Accept:** AUC on all 13 ABU scenes tabulated; residual is strictly non-negative; `lam → ∞` collapses the score to `||y||₂` (sanity check that the regularizer is wired the right way round).

#### 3A.5 `anomaly/streaming_rx.py` — the real-sensor-behaviour claim
```python
class StreamingCovariance:
    """Welford / Chan online mean + co-moment. Never materializes the full cube."""
    def __init__(self, n_bands: int, dtype=np.float64): ...
    def update(self, strip: np.ndarray) -> None:      # [rows, W, B] or [N, B]
    @property
    def mean(self) -> np.ndarray: ...                 # [B]
    @property
    def cov(self) -> np.ndarray: ...                  # [B, B], ddof=1

def streaming_rx(scene_path, *, strip_rows: int = 16, reg: float = 1e-6,
                 out_path: Path | None = None) -> np.ndarray:
    """Two passes over the file, strip by strip, per pushbroom sensor behaviour:
         pass 1 — accumulate StreamingCovariance
         pass 2 — Cholesky-factor once, score each strip, write incrementally
       Peak RSS is O(strip_rows * W * B + B²), NOT O(H * W * B).
    """
```
**Accept:** output matches `global_rx` to `rtol=1e-5` on Indian Pines; peak RSS measured by `tracemalloc` is < 1/8 of the full-cube path at `strip_rows=16`. Accumulation is in **float64** — float32 co-moment accumulation loses precision over 20 000+ pixels and quietly biases the covariance.

#### 3A.6 `anomaly/autoencoder.py` — kPCA features, not linear PCA
```python
class KPCAAutoencoder:
    """Pipeline: harmonize → reduce_bands(method="kpca", n_components=30)
                 → conv autoencoder on 15×15 spectral-spatial patches
       Score = per-pixel reconstruction error (L2 over the reduced spectrum).

       Trained on the HAD100 anomaly-free background pool, HARMONIZED FIRST —
       the pool spans two sensors with different band counts and cannot be
       stacked in native bands (D11). Scene count per D11, not asserted here.
       an autoencoder trained on a scene containing the targets it must detect
       learns to reconstruct them, which is self-defeating.
    """
    def fit(self, background_scenes, *, epochs=50, batch=64, lr=1e-3, amp=True)
    def score(self, cube, meta) -> np.ndarray
    def export_onnx(self, path: Path, *, opset=17) -> Path
```
Budget (D8): 15×15×30 patches, batch 64, AMP — comfortable at 4 GB.
**Accept:** AUC on ABU exceeds global RX on ≥ 8 of 13 scenes; ONNX export round-trips to within `atol=1e-4` of the torch output.

#### 3A.7 `anomaly/deep_detector.py` — scoped down, and say so (D8)
```python
class SpectralSpatialTransformer(nn.Module):
    """Compact ViT-style detector: 64×64×30 input, patch 8, dim 128, depth 4,
       heads 4. Self-supervised on the background pool via masked-band
       reconstruction; anomaly score = reconstruction error.

       SCOPED DOWN from the graph-transformer detector named in
       blueprint_upgrades_changelog.md §1 — that model OOMs at 4 GB VRAM.
       The comparison report MUST state this. Do not claim a graph-transformer
       result that was never run (Roadmap §1.10 reasoning, applied honestly).
    """
```
**Accept:** trains without OOM on the GTX 1650 at batch 8 + AMP; AUC tabulated as the deep-learning tier of the 3-way taxonomy.

#### 3A.8 `anomaly/scoring.py` — extended for fusion inputs
```python
def estimate_target_signature(cube, base_score, *, top_frac=0.001) -> np.ndarray:
    """Unsupervised target signature: mean spectrum of the top top_frac pixels by
       base_score. ACE needs a target; unsupervised anomaly detection has none,
       so it is bootstrapped from the base detector."""

def ace_score(cube, signature, *, mean=None, cov=None, reg=1e-6) -> np.ndarray:
    """Adaptive Cosine Estimator:
         ACE(x) = (s^T S^-1 x)² / ((s^T S^-1 s)(x^T S^-1 x))    ∈ [0,1]"""

def spectral_index_score(cube, meta, index_names: list[str]) -> np.ndarray:
    """Index set comes from configs/target_profile.yaml (D6), NOT hardcoded.
       object    → ndbi, iron_oxide_ratio, clay_ratio, brightness
       landcover → ndvi, ndwi, nbr, bsi
       Band selection is by nearest wavelength via preprocessing/bands.py —
       never by band index, which differs per sensor."""

def spatial_context_score(score, *, k: int = 7) -> np.ndarray:
    """Local contrast: score - median_filter(score, k), clipped at 0.
       Suppresses broad regional trends, keeps compact deviations."""
```

#### 3A.9 `anomaly/fusion.py`
```python
def fuse_scores(components: dict[str, np.ndarray], weights: dict[str, float],
                *, valid=None) -> np.ndarray:
    """Every component is rank_normalized (D3) BEFORE weighting — they are on
       incomparable native scales (Mahalanobis distance vs. a cosine ratio vs.
       an NDVI). Weighted sum, then renormalized to [0,1].
       Default weights: {rx: 0.40, ace: 0.25, index: 0.15, spatial: 0.20}
       Tuned by grid search on an ABU validation split; the tuning split is
       recorded and NEVER reused for reporting."""
```
**Accept:** fused AUC ≥ best single component on ≥ 10 of 13 ABU scenes; the weight sweep is logged to `experiments/rx_vs_ae/fusion_weights.json`.

#### 3A.10 Deliverable — `experiments/rx_vs_ae/`
`report.md` + `results.csv`, structured by the field's own 3-way taxonomy (per the changelog §8):

| Tier | Methods |
|---|---|
| Statistical | global RX · local RX · kernel RX · streaming RX |
| Representation-based | CRD |
| Deep learning | kPCA-autoencoder · spectral-spatial transformer |
| Fusion | multi-signal fused score |

Columns: ROC-AUC · PR-AUC · precision · recall · F1 · runtime (s) · peak RSS (MB) — **per scene**, **pooled-macro** and **pooled-micro** — over ABU (13) + HYDICE (1) + HAD100 (100).

**"Pooled" is defined, because with ABU it changes the answer.** ABU's anomaly density spans
**0.084 % to 2.72 %, a 32× range** (D13.2), so a pixel-weighted pool is effectively a report on
`abu-urban-4` and `abu-beach-2` and says almost nothing about `abu-beach-1`.

- **PRIMARY — scene-macro-average.** Each of the 13 ABU scenes contributes equally, regardless
  of how many anomaly pixels it contains. This is the headline number everywhere.
- **SECONDARY — pixel-micro-average**, pooling all pixels across scenes. May be reported, but
  **must be labelled `micro`** and may never appear without its macro counterpart beside it.

An unlabelled "pooled" figure is banned. Both columns appear in every deliverable table below.


---

### 6.2 — Branch 3B · Segmentation (Person C)

Implements D7 in full. **Governing rule: train on synthetic, score on real, never both on the same data.**

#### Data layout
```
data/benchmark/
  indian_pines/     Phase 2 wiring only — NO anomaly GT, never used in 3B
  abu/              13 real scenes  + real pixel masks   → SCORING
  hydice_urban_anomaly/  1 real scene 80×100×175 + mask   → SCORING (classical only, O8)
  had100/
    HAD100/         94 raw test + 522 raw background ENVI scenes (as shipped)
    HAD100Dataset/  produced by the repo's own main.py — THIS is what code reads
      test/         100 AVIRIS-NG 64×64×276 patches + GT      → SCORING
      train/
        aviris_ng/  1040 patches 64×64×276  (260 scenes × 4 corner crops)
        aviris/     1048 patches 64×64×162  (262 scenes × 4 corner crops)
                                                            → BACKGROUND POOL
  synthetic/        generated                            → TRAINING ONLY
```

**Background pool sourcing.** Primary: HAD100's own anomaly-free background pool — **2 088 patches of 64×64** (522 source scenes × 4 corner crops, per D11), already curated, target-free and aligned with the test scenes, which is precisely why HAD100 exists. Two hard constraints from D11: the NG (276-band) and Classic (162-band) halves **cannot be pooled before `harmonize()` runs**, and train/val splitting is **by source scene, not by patch** — the four crops of one 81×81 scene overlap by ~47 px and a patch-level split puts near-duplicates on both sides. Extension for sensor diversity: bulk **EnMAP L2A** (account held) and additional **AVIRIS-NG** flightlines, tiled to 64×64 and screened for target-free-ness by rejecting tiles whose global-RX 99.9th percentile exceeds a fixed threshold.

**Sentinel-2 is excluded from the background pool.** It is multispectral (~13 bands) and incompatible with the hyperspectral band count assumed downstream (Roadmap §9.1). Sentinel-2 appears only in Phase 5 Level 3.

#### 3B.1 `segmentation/synth.py` — implantation branch
```python
SPECTRA_POOLS = {
    "lib":      ("usgs_splib07", "ecostress_aster"),   # library endmembers, no dataset identity
    "abu_real": ("abu",),                              # target pixels from ABU GT masks
    "hyd_real": ("hydice_urban_anomaly",),             # target pixels from HYDICE GT mask
}

def load_target_spectra(pools: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """[K, B] on the canonical grid, plus a per-spectrum provenance tag.

       Provenance is a first-class property, not a comment. A spectrum harvested
       from an ABU ground-truth mask carries ABU's identity even though no ABU
       *scene* enters training — so scoring that model on ABU is leakage. The
       LODO matrix below is what prevents it; the scene-ID hygiene check cannot
       see this class of leak at all."""

def implant_targets(background: np.ndarray, target_spectra: np.ndarray, *,
                    n_targets: int, abundance_range=(0.1, 1.0),
                    size_range_px=(1, 40), shape="blob", seed: int
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Linear mixing model:
           m = a * t + (1 - a) * s
       t = target spectrum, s = the background pixel actually at that location,
       a = abundance fraction swept over abundance_range.

       a → 0.1 gives faint sub-pixel targets; a → 1.0 gives fully-visible ones.
       Sweeping a IS the difficulty control.

       Masks are exact and free: the implant location is CHOSEN, not annotated.
       Returns (cube, mask[C3], meta) where meta records per-target
       (a, spectrum_id, provenance, centroid, area_px) for the ablation curve.
    """
```

#### 3B.2 `segmentation/synth.py` — pretext branch (zero real target spectra)
```python
def pseudo_anomaly_patch(background: np.ndarray, *, n_regions: int,
                         perturbation: str = "mixed", seed: int
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Self-supervised pretext task: real patch vs. synthetic pseudo-anomaly patch.

       Region shape: arbitrary polygon (random convex hull / random walk blob).
       Spectral perturbation: ARBITRARY, deliberately NOT a real target spectrum —
         "shift"  : additive per-band offset drawn from the scene's own band std
         "scale"  : multiplicative gain
         "swap"   : substitute a spectrum from a distant scene in the pool
         "noise"  : structured (band-correlated) noise injection
         "mixed"  : sample one of the above per region

       Trains on the background pool ONLY. Requires no target library at all,
       which is what makes it the honest zero-prior comparison arm."""
```

#### 3B.3 `segmentation/datasets.py`
```python
class SyntheticSegDataset(Dataset):    # implanted OR pretext, flag-selected
class RealSegDataset(Dataset):    # abu | hydice_urban_anomaly | had100/test — EVAL ONLY
    def __init__(self, ...):
        assert split == "eval", (
            "RealSegDataset is scoring-only. Training on real GT violates D7.")
```
Both emit `(patch [C=30, 64, 64] float32, mask [1, 64, 64] uint8)`, produced by
`preprocessing.harmonize.reduce_bands` fit **only on the train split of the background pool**.
**Accept:** a unit test asserts `RealSegDataset` cannot be constructed with `split="train"`; a second test asserts no `roi_id` appears in both a train and an eval manifest.
**Accept — the C=30 criterion moved here from §3A.1 (2026-08-21):** HAD100 NG (harmonized 184)
and Classic (harmonized 184) reduce to the same `C=30` tensor shape via one `reduce_bands`
transformer, **fit on the train split's background patches only** — never on ABU/HYDICE/HAD100
*-test* scenes, and never on the whole pool before the split is decided. A fit/transform
round-trip on held-out train-split pixels reconstructs with < 2 % mean relative error at
`n_components=30`. Fitting on anything outside the train split is a data-hygiene bug of the
same class §3B.5b's LODO matrix exists to prevent, just one level deeper: not a scene leaking
into training, but a scoring scene's spectral statistics leaking into the representation every
model is built on. **ABU and HYDICE are excluded from this criterion by the O8 decision** — they
ship no wavelength array (D13.4), and ABU spans seven band counts (205/204/193/191/188/102), so
"ABU (205 bands)" as previously written was wrong for 8 of 13 scenes.

#### 3B.4 `segmentation/train_unet.py`
```python
class LightUNet(nn.Module):
    """Encoder 32-64-128-256, decoder mirror + skips, GroupNorm, output 1 logit.
       ~1.9 M params. Input [B, 30, 64, 64]."""
```
Training: Dice + BCE (0.5/0.5), AdamW `lr=3e-4` `wd=1e-4`, cosine schedule, AMP fp16, batch 16, early stop on the synthetic *validation* split.
Runs are kept separate and never averaged together. The implanted branch is not one run but a **leave-one-dataset-out (LODO) set**, because the scoring set determines which spectra pools are legal — see the matrix in §3B.5b.

#### 3B.5 `segmentation/train_alt_arch.py`
```python
class CompactSegFormer(nn.Module):
    """SegFormer-B0-class: 4-stage hierarchical transformer encoder, all-MLP
       decoder. ~3.7 M params. batch 8 + grad-accum 2 + AMP at 4 GB (D8)."""
```
Same training branches as 3B.4, under the same LODO matrix.

#### 3B.5b — Leave-one-dataset-out spectra matrix (closes the spectrum-level leak)

Which spectra pools a model may train on is determined by what it will be scored on:

| Model | Spectra pools | Legal scoring sets | Leak status |
|---|---|---|---|
| `implanted_lib` | `lib` | abu · hydice · had100 | clean — library endmembers carry no dataset identity |
| `implanted_lodo_abu` | `lib` + `hyd_real` | **abu only** | clean for ABU |
| `implanted_lodo_hyd` | `lib` + `abu_real` | **hydice only** | clean for HYDICE |
| `implanted_all_real` | `lib` + `abu_real` + `hyd_real` | **had100 only** | clean for HAD100 — no HAD100 spectra were ever harvested |
| `pretext` | *none* | abu · hydice · had100 | structurally clean — uses no target spectra at all |

`implanted_lib` and `pretext` are the two rows scoreable on all three sets, so they are the **headline comparison**. The LODO rows exist to answer "does using real harvested target spectra help?" without buying that answer with leakage.

Per architecture (U-Net, SegFormer) this is 5 models → **10 trained models total**.

**Enforcement:** `synth.implant_targets` stamps the pool set into the output manifest, and `RealSegDataset` refuses to score a model whose manifest lists a pool derived from the dataset being scored. This is an assertion at load time, not a convention.

#### 3B.6 `segmentation/infer.py`
```python
def segment_rois(cube, meta, rois: list[ROIRecord], model, *,
                 patch: int = 64, batch: int = 16) -> list[ROIRecord]:
    """ROI-only inference (Roadmap §6 stage 5) — crops each ROI bbox to a
       patch-aligned window, runs the model, writes seg_prob back onto the
       ROIRecord. Full-scene inference exists ONLY as the edge/ comparison
       baseline; it is never the operational path."""
```

#### 3B.7 `segmentation/postfilter.py` — profile-driven (D6)
```python
def shape_plausibility(props, profile: dict) -> float:
    """[0,1] from area / solidity / elongation against the ACTIVE profile.
       0.0 = implausible → filtered. Feeds c_shape in the D4 confidence."""

def filter_rois(rois, profile) -> tuple[list[ROIRecord], list[ROIRecord]]:
    """Returns (kept, dropped). Dropped ROIs are NOT discarded — they are
       written to experiments/cascade_recall_audit/ so a stage-2 false negative
       caused by the post-filter is traceable rather than invisible."""
```
Profile thresholds per D6. **Sanity constraint:** the `object` profile's `max_area_px = 2000` sits against 64×64 HAD100 patches (4 096 px total) — a target may legitimately occupy half a patch, so `max_area_px` is expressed as `min(2000, 0.5 * scene_px)` and computed per scene, not as a constant.

> **Correction (2026-08-22, found by execution).** The rule is right; **this paragraph's worked example was wrong.** It claimed the per-scene term makes 64×64 and 120×120 behave differently, but `0.5 * scene_px` is **2 048** at 64×64 and **7 200** at 120×120, and both exceed the base `max_area_px = 2000` — so the resolved ceiling is **2000 at both sizes** and the scaling term is a no-op at exactly the two shapes named to demonstrate it. It binds only below ~4 000 scene px (roughly ≤ 63×63). `resolve_profile` implements the plain reading — a cap stopping one ROI from exceeding half the scene — and is correct; the claim that these two sizes yield opposite verdicts for the same ROI does not hold. Pinned in `tests/test_postfilter_profile.py` at both the genuinely-differing sizes (32×32 → ceiling 512 vs 120×120 → ceiling 2000) **and** at the plan's own named sizes, so the no-op stays recorded in the suite rather than only here. Same class as D20's arithmetic slip: the normative prose was right and the illustration was not.

#### 3B.8 Deliverable — `experiments/seg_arch/`
**O8 is decided: option 2 of D13.4 is adopted.** Learned models are scored on **HAD100 only** —
it is the one benchmark that ships real wavelengths, so it is the only one `harmonize()` can put
on the canonical grid. ABU and HYDICE remain full scoring sets for the **classical** detectors
(RX family, CRD), which run on native bands and need no wavelengths at all.

| Model | Kind | Train data | **Spectra provenance** | **Scored on** |
|---|---|---|---|---|
| `unet_implanted_lib` | learned | synthetic implanted | `lib` | **had100/test only** |
| `unet_pretext` | learned | synthetic pretext | *none* | **had100/test only** |
| `unet_lodo_abu` | learned | synthetic implanted | `lib`+`hyd_real` | — *(suspended, see below)* |
| `unet_lodo_hyd` | learned | synthetic implanted | `lib`+`abu_real` | — *(suspended, see below)* |
| `unet_all_real` | learned | synthetic implanted | `lib`+`abu_real`+`hyd_real` | — *(suspended, see below)* |
| `segformer_*` | learned | *(same five)* | *(same five)* | *(same five)* |
| `global_rx` · `local_rx` · `kernel_rx` · `crd` | **classical** | *(none — unsupervised)* | *n/a* | **abu (13) · hydice_urban_anomaly (1) · had100/test (100)** |

**All three real-spectra rows are suspended, not deleted.** `unet_lodo_abu` exists to be scored
on ABU and `unet_lodo_hyd` on HYDICE; with O8 in force neither has a legal scoring set, so
training them would produce a model with nowhere to report. `unet_all_real` is suspended for a
related but distinct reason (D19): its *training* input needs `abu_real`/`hyd_real` target
spectra pulled from the ABU/HYDICE GT masks, and implanting those into a canonically-harmonized
background patch needs a wavelength array to put them on the same 184-band grid — which ABU and
HYDICE don't ship (D13.4). Same root cause as O8, different side of the pipeline. All three stay
in the table, marked suspended, and return automatically if **O9** recovers wavelengths. Deleting
them would hide that the design was cut down by a data limitation rather than by choice.

**What this costs, stated plainly:** §3B.5b's leave-one-dataset-out matrix collapses from five
arms to three, and the cross-sensor generalization claim — train on synthetic, score on three
independent real sensors — is reduced to one sensor. The `implanted_lib` vs. `pretext`
comparison survives intact, because both are scoreable on HAD100 and both are leak-free there.
That comparison was always the headline; it is now also the only learned-model claim.

The **spectra-provenance column is mandatory in the published table.** A row without it is unreadable — the reader cannot tell whether the number is leak-free, and neither can you six weeks later.

Reported as: IoU and Dice **per real scene, pooled-macro (primary) and pooled-micro (secondary, labelled)** — the same definition as §3A.10, and for the same reason; architecture comparison (U-Net vs. SegFormer); and **implanted-trained vs. pretext-trained as its own comparison row, never merged into a single number** — using `implanted_lib` vs. `pretext`, the two leak-free rows — scoreable on HAD100, which under O8 is the whole learned-model scoring set. Plus the abundance ablation: detection rate vs. sub-pixel abundance `a`, swept 0.1 → 1.0, which is the curve that makes the study look rigorous rather than anecdotal.

---

### 6.3 — Branch 3C · Change Detection (Person D)

#### 3C.1 `preprocessing/registration.py` — build it properly, don't assume alignment
```python
def coregister_subpixel(cube_t1, cube_t2, meta_t1, meta_t2, *,
                        upsample_factor: int = 20, refine: bool = True
                        ) -> tuple[np.ndarray, dict]:
    """Two-stage:
       1. skimage.registration.phase_cross_correlation on a band-averaged
          panchromatic proxy → sub-pixel translation at 1/upsample_factor px.
       2. refine=True → cv2.findTransformECC (MOTION_AFFINE) for rotation/scale.
       Resamples t2 onto the t1 grid (C4). Returns (cube_t2_aligned, report)
       with report = {shift_px, rmse_px, ecc_score, converged}.
       Raises RegistrationFailure if rmse_px > 1.0 — a silently misregistered
       pair produces change maps that are pure artefact, and misregistration is
       the single most-cited real-world failure source in this literature."""
```
**Accept:** a synthetically shifted pair (known shift 3.7, -2.3 px) recovers to within 0.1 px; a deliberately un-alignable pair raises rather than returning garbage.

#### 3C.2 `change_detection/spectral_angle.py` — the primary signal
```python
def spectral_angle(cube_t1, cube_t2) -> np.ndarray:
    """SAM(x1, x2) = arccos( <x1,x2> / (||x1|| ||x2||) )  ∈ [0, π]
       Invariant to uniform brightness scaling by construction, which is exactly
       what makes it robust to the illumination/seasonal shifts that generate
       most pseudo-change. Returns [H, W] float32 radians."""
```

#### 3C.3 `change_detection/temporal_difference.py` — the baseline arm
```python
def magnitude_difference(cube_t1, cube_t2, *, norm="l2") -> np.ndarray
    """||x2 - x1||. Kept as the classical comparison arm, NOT as the primary
       signal — it is what SAM is being measured against."""
```

#### 3C.4 `change_detection/physics_fusion.py`
```python
def difference_structure(cube_t1, cube_t2, *, patch: int = 7
                         ) -> dict[str, np.ndarray]:
    """Patch-wise statistics of the difference space. Genuine change and
       pseudo-change have measurably different structure there:
         variance     — local variance of the per-band difference
         entropy      — Shannon entropy of the difference histogram per patch
         coherence    — cross-band correlation of the difference vector
       Returns all three as [H, W] float32."""

def fuse_change_signals(sam, structure, cloud_mask, weights) -> np.ndarray:
    """rank_normalize each (D3), weighted sum, zeroed where cloud_mask==1.
       Default {sam: 0.50, variance: 0.20, entropy: 0.15, coherence: 0.15}."""
```

#### 3C.5 `change_detection/siamese_net.py` — the learned arm
```python
class SiameseChangeNet(nn.Module):
    """Shared-weight encoder over both epochs, attention-based fusion of the
       two feature stacks, lightweight decoder to a change logit.
       ~2.4 M params, input 2×[30, 64, 64]. batch 8 + AMP at 4 GB (D8).
       Trained on synthetic change pairs generated by segmentation/synth.py —
       implant a target into t2 only, leaving t1 clean. Same D7 rule:
       train on synthetic, score on real."""
```

#### 3C.6 `change_detection/temporal_baseline.py`
```python
class TemporalBaseline:
    """Per-pixel running median + MAD across an N-epoch stack, so a change score
       can be computed against a seasonal baseline rather than a single prior
       date. Uses the same StreamingCovariance discipline — the stack is never
       fully materialized. Feeds Phase 5 Level 3."""
```

#### 3C.7 `preprocessing/cloud_mask.py`
```python
def cloud_shadow_mask(cube, meta, *, method="auto") -> np.ndarray:
    """Sentinel-2 → read the SCL band (classes 3,8,9,10 = shadow/cloud/cirrus).
       Hyperspectral → spectral thresholds: high 450nm brightness + low 1600nm
       + low NDVI. Returns C3 uint8. Feeds c_clear in the D4 confidence."""
```

#### 3C.8 Deliverable — `experiments/change_arms/`
Three-arm comparison on registered pairs: **classical magnitude diff** vs. **SAM + physics fusion** vs. **learned Siamese net**. Metrics: precision · recall · F1 · ROC-AUC on changed pixels, plus a **pseudo-change rate** measured on pairs with known illumination-only differences (same scene, different acquisition time, no real change) — that number is the one that justifies choosing SAM over raw differencing.

---

### 6.4 — Branch 3D · Edge / Systems + Geospatial (Person E)

**Read §0.3 first. There is no Raspberry Pi. Every measured number in this branch is `SIMULATED` and must be labelled as such.**

#### 3D.1 `edge/streaming.py`
```python
class StripPipeline:
    """Strip-based streaming across the WHOLE pipeline, not just RX.
       Stages register with a required look-ahead in rows; the scheduler feeds
       overlapping strips so a stage needing k rows of context gets them without
       the caller managing halos.
       Enforces a hard RSS ceiling (default 6 GB, headroom under a Pi 5's 8 GB)
       and raises MemoryBudgetExceeded rather than swapping — a pipeline that
       silently swaps invalidates every latency number taken from it."""

    def register(self, name: str, fn: Callable, *, lookahead_rows: int = 0)
    def run(self, scene_path, *, strip_rows: int = 16) -> dict
```

#### 3D.2 `edge/quantization.py` — mixed precision, per the changelog §5
```python
def export_onnx(model, sample_input, path, *, opset=17) -> Path

def quantize_mixed(onnx_path, out_path, calibration_data, *,
                   fp16_ops: set[str], int8_ops: set[str]) -> Path:
    """Mixed FP16/INT8, NOT uniform INT8.
       FP16 — covariance/statistics-sensitive stages (RX matmuls, the AE encoder,
              anything feeding a Cholesky). INT8 quantization of a covariance
              path destroys the conditioning and the detector degrades silently.
       INT8 — threshold-stage and late decoder convolutions, where the output is
              about to be binarized anyway.
       Target from the literature: ~12x size reduction, ~6x compute reduction,
       ≥99% of FP32 accuracy retained. That is a beatable benchmark, and the
       report must state measured-vs-target rather than assuming it was hit."""

def accuracy_delta(fp32_scores, quantized_scores, labels) -> dict
    """AUC / F1 / IoU delta. A quantization that costs >1% AUC is rejected."""
```

#### 3D.3 `edge/onnx_inference.py`
```python
class ONNXRunner:
    """CPUExecutionProvider only — no CUDA, so the code path is identical to
       what would run on a Pi. Thread count is explicit, not left to the default,
       because the default reads the host's core count and makes results
       non-comparable across machines."""
```

#### 3D.4 `edge/profiling.py` — continuous, per module
```python
@contextmanager
def profile_stage(name: str, sink: ProfileSink):
    """Wall time, CPU time, peak RSS delta (tracemalloc + psutil), thread count.
       Power is NOT recorded — there is no instrumented hardware, and an
       estimated wattage reported as measured would be a fabrication."""

class ProfileSink:
    """Appends to experiments/edge_benchmarks/{run_id}.jsonl.
       Every record carries "measurement": "SIMULATED" (§0.3)."""

def regression_check(baseline_run, current_run, *, tol=0.15) -> list[str]
    """Fails CI when any stage regresses >15% vs. the committed baseline. This is
       what makes profiling continuous rather than a one-off at the end."""
```

#### 3D.5 `edge/constrained_sim.py` — the Pi stand-in
```python
def run_constrained(cmd: list[str], *, cores: int = 4, mem_mb: int = 8192,
                    cpu_quota_pct: int = 100) -> dict:
    """taskset -c 0-3 + cgroup v2 memory.max + cpu.max to approximate a Pi 5.
       Returns timings tagged measurement="SIMULATED", host_cpu=<model>.

       This is a REGRESSION GUARD AND A RELATIVE COMPARISON. A constrained
       x86 laptop core is not a Cortex-A76. Never present these as Pi numbers.
    """
```

#### 3D.6 `edge/roi_pipeline.py` + `edge/benchmark.py`
```python
def roi_vs_full_comparison(scene, detector, seg_model) -> dict:
    """The core edge-value claim, measured rather than asserted (Roadmap §1.8):
         full-scene segmentation latency  vs.  ROI-only latency
         pixels processed by stage 2      vs.  total pixels
         % of image discarded by ROI screening
         bandwidth: full cube bytes       vs.  GeoJSON bytes transmitted
    """
```
**Accept:** on ABU-Airport-1, the ROI path processes < 10 % of pixels at stage 2 **while stage-1 recall meets the calibrated target set in §4.2 (`target_recall = 0.98`)** — this branch does not define its own recall floor, because two different recall numbers in two sections is exactly how a cascade quietly ships at the weaker one. The bandwidth ratio is reported as an explicit multiple.

#### 3D.7 Geospatial ownership
Person E owns `geospatial/` (built in Phase 2) and integrates every branch's output into the C6 schema — including the D5 provenance fields and the D5 fusion rule. Also owns `qgis/styles/*.qml` (score raster magma ramp; ROI outlines coloured by `source_branch`; labels from `roi_id`) and `qgis/projects/*.qgz`.

**Reminder (Roadmap §7):** this is the project's confirmed literature gap — almost all hyperspectral-ML work stops at pixel-space AUC/IoU and never closes the loop into georeferenced GIS output. It is the central differentiator and should be reported as such.

---

### 6.5 — Branch 3E · Quantum (Person F)

> **Read D27 before this section.** 3E was built on 2026-08-22 and seven of the eight
> design points in D27 are *corrections* to what follows: `Sampler` (V1) no longer exists,
> `ZZFeatureMap`/`RealAmplitudes` are deprecated classes, §3E.4's "mirrors 3A.6" compares
> input representations rather than architectures, §3E.6's five arms mix supervision
> regimes in one ranking, and §3E.8's novelty claim is false as written. The subsections
> below are kept as the original specification; D27 is what was built.

Never a dependency of the operational pipeline (Roadmap §1.5, §9.10). Never runs on the edge device (Roadmap §6). PC + simulator only.

#### 3E.1 `quantum/qiskit_basics.py`
Environment check: `qiskit 2.5.2`, `qiskit-aer 0.17.2`, `qiskit-machine-learning 0.9.1`. Builds and runs a Bell-state circuit on `AerSimulator`. **Accept:** measured distribution within 3σ of 50/50 at 4096 shots.

#### 3E.2 `quantum/feature_map.py`
```python
def build_feature_map(n_features: int, *, kind="zz", reps=2, entanglement="linear"):
    """kind in {"zz", "z", "pauli"}. n_features in 8..16 — set by qubit count,
       which is the hard constraint on the whole branch."""

def classical_reduce(cube, meta, *, n_features=8) -> np.ndarray:
    """PCA → n_features, then MinMax to [0, π] for angle encoding.
       Reuses preprocessing/harmonize.reduce_bands so the quantum branch and the
       classical AE are compared on the SAME reduced features. Comparing a
       quantum model on one feature basis against a classical model on another
       measures the basis, not the model."""
```

#### 3E.3 `quantum/vqc_encoder.py`
`VQC` from `qiskit_machine_learning.algorithms`, `RealAmplitudes` ansatz, `COBYLA`, `AerSimulator` with a seeded `Sampler`. Binary anomaly/background classification on the reduced features.

#### 3E.4 `quantum/quantum_autoencoder.py`
Compression from `n_qubits` to `n_latent` with a SWAP-test trash-state fidelity cost. Anomaly score = reconstruction infidelity. Mirrors 3A.6's autoencoder so the comparison is architecture-to-architecture.

#### 3E.5 `quantum/quantum_kernel.py`
`FidelityQuantumKernel` → precomputed Gram matrix → `sklearn.svm.OneClassSVM(kernel="precomputed")`. The second established QML paradigm alongside the variational circuit; covering both is what makes it a complete comparative study rather than one arbitrary architecture choice.

#### 3E.6 `quantum/classical_vs_quantum.py`
Head-to-head on identical PCA features: classical AE · classical OneClassSVM (RBF) · VQC · QAE · quantum kernel. Metrics: ROC-AUC · PR-AUC · F1 · circuit depth · qubit count · wall-clock. **The comparison table is the deliverable, not an advantage claim** (Roadmap §1.10, §9.6).

#### 3E.7 Real hardware run — **CONDITIONAL**
Requires an IBM Quantum account, which is **not currently held** (§0.2). When available: transpile one demonstration circuit to a free-tier backend, run with and without error mitigation, and report simulator-vs-hardware side by side. Until then this is a prerequisite task, **not a deliverable**, and no hardware result may appear in any report.

#### 3E.8 Novelty framing — `docs/experiments.md`
State the scoped claim in writing: *no existing published work applies VQC/QAE feature encoding directly to hyperspectral anomaly detection.* Back it with a dated literature search recorded in the doc (query strings, databases, date), and with the classical-vs-quantum comparison numbers. It is a **scoped novelty claim, not an advantage claim** — the distinction is the whole point.

---

## 7. Phase 4 — Integration

Every branch built against the same frozen contracts, so integration is a **config edit**, not a rewrite. If it turns out to be a rewrite, a contract was violated somewhere and that is the bug to find.

### 4.1 Swap placeholders for winners
`configs/pipeline.yaml` selects each stage's implementation by name:
```yaml
detector:      fused          # was: global_rx
threshold:     recall_calibrated   # was: percentile
segmentation:  unet_implanted # was: (none)
change:        sam_physics    # was: (none)
```
`pipeline/run_pipeline.py` resolves names through a registry. **Accept:** switching `detector: global_rx → fused` changes no code and still passes `validate_geojson`.

### 4.2 Recall-first threshold calibration
```python
# anomaly/scoring.py
def calibrate_threshold_for_recall(scores, labels, *, target_recall=0.98) -> float:
    """Pick the LOWEST threshold achieving target_recall on the validation set.
       Deliberate over-triggering: in a cascade, a stage-1 false positive costs
       cheap extra stage-2 compute; a stage-1 false negative is UNRECOVERABLE,
       because stage 2 never sees that region at all.
       Returns the threshold AND the resulting false-positive rate, so the
       compute cost of the recall target is explicit rather than hidden."""
```
**Accept:** stage-1 recall ≥ 0.98 on the ABU validation split, with the induced FP rate recorded in the run manifest.

### 4.3 ROI-level fusion — implements the D5 rule
```python
# geospatial/roi_fusion.py   (new)
def fuse_rois(anomaly_rois, change_rois, *, iou_threshold=0.3
              ) -> list[ROIRecord]:
    """Per D5:
       - Merge ONLY parents sharing the same target_profile.
         Result: source_branch="fused", profile inherited, parent_roi_ids set.
       - Different-profile spatial overlaps are NOT merged. Both ROIs survive;
         each records the other in linked_roi_ids.
       Rationale: a merged cross-profile ROI would silently carry post-filter
       thresholds it was never screened against, destroying provenance."""
```
**Accept:** a test with one `object` anomaly ROI and one `landcover` change ROI at IoU 0.8 produces **two** output ROIs with reciprocal `linked_roi_ids`, not one merged ROI.

### 4.4 End-to-end re-run
Full pipeline on Indian Pines and ABU. **The GeoJSON schema and the QGIS project must work unchanged** — that is the actual test of whether the frozen contracts held. Any required change to `qgis/projects/*.qgz` is a contract-violation signal, not a styling task.

---

## 8. Phase 5 — Validation (all three levels)

### 8.0 PREREQUISITE — verify the four remaining datasets against their files. Do this first.

**No Phase 5 code is written against an assumed spec.** Four datasets are still
documentation-only: **EnMAP L2A, Sentinel-2 L2A, AVIRIS/AVIRIS-NG flightlines, USGS splib07**.
Every number the plan states about them came from a project page, and that is precisely the
evidence standard that produced five wrong HAD100 facts (D11) and three wrong ABU/HYDICE facts
(D13).

Run the same file-opened verification, on the model of `scripts/verify_had100.py` and
`scripts/verify_benchmarks.py`, producing `scripts/verify_phase5_datasets.py` →
`docs/phase5_datasets_verified.json`. For each dataset record, **from the files**:

| Check | Why it is on this list specifically |
|---|---|
| actual band count per product | HAD100 shipped 425/224; ABU shipped seven different counts |
| actual dtype per product | ABU shipped three dtypes inside one dataset (D13.2) |
| **wavelength array present?** | the single question that decided O8 — answer it *before* writing Level 2/3 code, not after |
| real CRS/affine present? | HAD100's ENVI headers carried real UTM; the plan had assumed otherwise (D11.5) |
| file structure and no-data sentinels | HAD100 used two different sentinels in one pool |
| Sentinel-2 only: resolution mixing | 10/20/60 m bands per product; Level 3 code must not assume one grid |
| licence / redistribution terms | D12 rule 4 forbids uploading benchmark data to third-party notebooks unchecked |

**This is scheduled here deliberately, not in Phase 1–3.** None of the four is on the near-term
critical path (§11), so verifying them early would be busywork — but discovering an O8-class
problem *while* Phase 5 is being written would not be. Do it as the first Phase 5 task, under no
time pressure, before any Level 2 or Level 3 module is authored.

#### 8.0a Access paths, verified 2026-08-21 — *access only, not a single data fact*

Both Phase 5 accounts were probed against the live services. **This verifies how to reach the
data, not what is in it.** EnMAP L2A and Sentinel-2 L2A stay in the documentation-only tier of
§15; not one file has been opened, so band count, dtype, wavelengths, CRS and no-data are all
still unverified and remain the job of `verify_phase5_datasets.py` above.

| Service | Search leg | Download leg |
|---|---|---|
| Copernicus (Sentinel-2) | `catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=…` — **HTTP 200, no credentials** | `download.dataspace.copernicus.eu/…/$value` — **HTTP 401**, honest failure. S3 keys (§4.1b) |
| DLR (EnMAP) | `geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A` — **HTTP 200, no credentials** | `download.geoservice.dlr.de/ENMAP/files/…` — **HTTP 200 + HTML login page**, silent failure. EOC UMS SSO (CAS), `DLR_USERNAME`/`DLR_PASSWORD` |

Three consequences, all already implemented:

1. **Neither fetcher needs credentials to search.** Catalogue queries, footprint filtering and
   scene selection can be written and tested with no account at all. Only retrieval is walled.
2. **DLR fails silently and must be guarded on content.** An unauthenticated asset GET returns
   `200 text/html` with a 50 KB login page — every status-code check in the world passes it
   straight into the parser. This is the *third* HTML-as-data incident here (ehu.eus Indian
   Pines, GitHub-raw HYDICE, now DLR), so it is now a shared assertion, `core/http_guard.py`
   (`assert_not_html` / `assert_magic`), tested in `tests/test_http_guard.py` against the real
   login-page bytes. **`fetch_enmap.py` and `fetch_sentinel2.py` must call it on every payload.**
   Related: DLR **ignored a `Range` request** (`-r 0-2047` returned 50 448 bytes), so resumable
   download must not assume byte ranges are honoured.
3. **EnMAP is username/password, not a token.** The wall is CAS
   (`sso.eoc.dlr.de/eoc/auth/login?service=…`), a ticket flow with no API token. The earlier
   `EOWEB_TOKEN` variable was therefore **deleted, not left blank** — advertising a variable that
   cannot exist is worse than not offering one. The `EOWEB_*` prefix was renamed `DLR_*` because
   the automatable route is the Geoservice STAC API, not the EOWEB GeoPortal's cart-and-order UI.

**Trap found while writing `verify_access.py`.** The EOC login page serves **two** POST forms: the
credential form (`username`/`password`/`execution`/`submitBtn`, **no** `_csrf`) and a separate
SSO-button form (`_csrf`/`client_name`/`submitButton`) carrying its **own, different** `execution`
token. A page-wide regex for hidden inputs silently splices the two, and CAS answers **401 — which
is indistinguishable from a wrong password.** Scope every scrape to the form containing
`name="password"`. `submitBtn` is a `<button>`, not an `<input>`, so an input-only scrape drops it.
CAS here is stateless (the `execution` token is ~2.3 KB and self-contained); **no `JSESSIONID` is
issued on GET**, so its absence is not a bug and must not be used as a health check. Success is
signalled by a ticket-granting cookie appearing *after* the POST.

**Exit criterion:** every row of §15's documentation-only tier has moved to the verified tier, or
carries a recorded reason why it could not (e.g. credentials unavailable — §14).


### Level 1 — Benchmark
Indian Pines · ABU (13) · HYDICE Urban · HAD100 (100) · optionally Pavia, Salinas.
Metrics: ROC-AUC · PR-AUC · precision · recall · F1 · pixel IoU where masks exist.
**Stage-wise, not just end-to-end** — stage-1 (detector) recall is reported as its own number, separate from final pipeline accuracy. This is what answers the standard cascade-bottleneck critique with evidence instead of assertion.

**False-negative audit — `experiments/cascade_recall_audit/`:** every real labelled target the stage-1 detector fails to flag is logged individually with its scene, pixel extent, mean score, and the margin by which it missed the threshold. Additionally logged: targets that stage 1 caught but `postfilter.filter_rois` then dropped (§3B.7). A miss that is invisible cannot be argued about.

### Level 2 — Real hyperspectral
EnMAP L2A and AVIRIS/AVIRIS-NG scenes (accounts held). This is where **real georeferencing is verified for the first time** (D2) — geospatial metrics from Roadmap §8 (location error, polygon IoU, area error, coordinate accuracy) are computed here and **never** from Indian Pines.
**Accept:** polygon centroids for a manually identified feature land within 2 pixels (~2 GSD) of its true position, verified in QGIS against an independent basemap.

### Level 3 — Real multitemporal geospatial case study
Sentinel-2 time series over a real Indian location, via Copernicus Data Space (account held). Runs the `landcover` profile (D6). Pipeline: fetch stack → cloud mask → co-register → `TemporalBaseline` → SAM + physics fusion → ROIs → GeoJSON → QGIS, compared against known dated events.

**Reporting constraint, non-negotiable (Roadmap §1.9, §9.7):** report **observed physical change only**. No conflict attribution, no causal claim, no inference about intent from image differences. If a disaster or Kashmir case study is included, it is framed strictly as observed physical change with dates and extents. This is a wording rule that applies to the report, the slides, and anything said on stage.

---

## 9. Phase 6 — Edge Benchmarking — **CONDITIONAL / SIMULATED**

**No Raspberry Pi exists (§0.2).** This phase runs in two tiers, and the distinction between them must survive into every table.

### Tier A — runs now, on the laptop, labelled SIMULATED
1. Full ONNX pipeline under `edge/constrained_sim.py` (4 cores, 8 GB cap).
2. Streaming pipeline over ABU + HAD100 scenes with the RSS ceiling enforced.
3. Recorded: per-scene latency, per-ROI latency, peak RSS, CPU time, model size, bandwidth-before/after, % of image discarded by ROI screening.
4. `regression_check` wired into CI against a committed baseline.
5. Stage-1 recall logged explicitly; every miss audited into `experiments/cascade_recall_audit/`.

**Power is not reported.** There is no instrumented hardware, and an estimated wattage presented as measured would be a fabrication.

### Tier B — blocked on hardware
Real Pi 5 latency/RAM/power, and the FPGA/NPU comparison arm. The code is written Pi-ready; only the measurement is blocked. `docs/validation.md` states plainly which numbers are simulated and which are measured — the reader should never have to work that out.

---

## 10. Phase 7 — Demo Assembly

`pipeline/demo.py` — a single scripted run producing the Roadmap §5 Phase-7 sequence:

1. Load a real hyperspectral scene (EnMAP/AVIRIS, Level 2 validated).
2. Local preprocessing — harmonize, bad bands, cloud mask.
3. Fused anomaly detection, with the live score raster displayed.
4. Recall-calibrated ROI extraction; **stage-1 recall shown on screen**.
5. Segmentation on ROI crops only, with the pixel-count saving shown live.
6. Polygons with the full C6 attribute set including D5 provenance.
7. GeoJSON written locally; opened in QGIS beside the source imagery.
8. **Network disabled for the entire inference stage**, and demonstrated — `demo.py` takes `--assert-offline`, which asserts no socket is opened during inference and fails loudly if one is. Claiming offline operation without proving it is the weakest possible version of this demo.
9. Latency / memory / bandwidth-saved on screen, each labelled `SIMULATED` where it is (§0.3).
10. If temporal data is loaded: t1-vs-t2 with the SAM + physics-fusion signal.
11. Classical-vs-quantum comparison shown explicitly as a research branch.

**Judge-facing narrative** is Roadmap §5 verbatim — do not improvise a stronger claim on stage than the numbers support.

---

## 11. Execution Order (dependency DAG)

```
Phase 1  ─ env · contracts · fetchers                       [SERIAL, blocks all]
   │
Phase 2  ─ walking skeleton                                 [SERIAL, one owner]
   │        loader → normalize → global_rx → scoring →
   │        postfilter → polygonize → projections → geojson →
   │        run_pipeline → QGIS verify
   │
   ├── 3A ─ harmonize ──┬─ local_rx · kernel_rx · crd · streaming_rx
   │                    ├─ autoencoder · deep_detector      [needs harmonize]
   │                    └─ scoring+ · fusion                [needs all detectors]
   │
   ├── 3B ─ synth ──── datasets ── train_unet ─┐
   │        [needs 3A.harmonize + HAD100 bg pool]
   │                              train_alt_arch┼─ infer ── postfilter
   │                                            ┘
   ├── 3C ─ registration ─ spectral_angle ─ physics_fusion ─ siamese_net
   │        [siamese_net needs 3B.synth]
   │
   ├── 3D ─ streaming · profiling · constrained_sim         [start immediately]
   │        quantization · onnx_inference                   [needs 3A/3B models]
   │        roi_pipeline · benchmark                        [needs 3A+3B]
   │
   └── 3E ─ qiskit_basics ─ feature_map ─┬─ vqc_encoder
            [needs 3A.harmonize]         ├─ quantum_autoencoder
                                         └─ quantum_kernel ─ classical_vs_quantum
   │
Phase 4  ─ registry swap · recall calibration · roi_fusion · e2e re-run
Phase 5  ─ L1 benchmark → L2 real hyperspectral → L3 Sentinel-2 case study
Phase 6  ─ Tier A simulated benchmarks   [Tier B blocked on hardware]
Phase 7  ─ demo assembly
```

**Critical path:** `contracts → Phase 2 → 3A.harmonize → 3B.synth → 3B training → Phase 4 → Phase 5`.

---

### 11.1 PRIORITY ORDER — read this before claiming any task (added 2026-08-21)

Capacity is finite and the DAG is wider than the schedule. **Build P0 to completion before any
P1 work starts.** An agent that finishes its task should take the next P0 item, not the most
interesting one.

| tier | scope | rationale |
|---|---|---|
| **P0 — ship this** | finish `3B` (`train_unet` → `infer`) · `local_rx` · `kernel_rx` · `crd` · `streaming_rx` · `fusion` · **Phase 4** · **Phase 5 Level 1** · **Phase 7 `demo.py`** | the critical path plus the benchmark that makes it defensible and the demo that makes it presentable. A working, benchmarked, demoable system. |
| **P1 — if P0 is done** | `3D`: `profiling` · `constrained_sim` · `roi_pipeline` · `benchmark` · **Phase 6 Tier A** | edge story; simulated only, and §9 forbids reporting power |
| **P2 — genuinely optional** | `3C`: `registration` · `spectral_angle` · `physics_fusion` · `siamese_net` — and ~~`3E`: the whole quantum arm~~ (**3E BUILT 2026-08-22, D27**) | research breadth. Phase 7 step 11 shows quantum "as a research branch"; a missing branch is a smaller loss than a broken P0. |
| **P3 — deferred, do not start** | `train_alt_arch` · `deep_detector` · `autoencoder` · `quantization` · `onnx_inference` | architecture comparison and deployment polish. Valuable, not load-bearing. |
| **BLOCKED — never start** | Phase 6 **Tier B** | no instrumented hardware exists (§0.2, §9). Not a scheduling choice. |

**P0 STATUS, 2026-08-22.** Built and tested unless noted:

| item | state |
|---|---|
| `3B` train → infer | **done.** `unet_pretext` trained to convergence (40 epochs, best val 0.1243 @ epoch 27, 5.1 h local GTX 1650). `infer.segment_rois` + profile-driven `postfilter` built. **Only 1 of §3B.8's 5 arms is trainable** — three suspended pending O9 (D19), `unet_implanted_lib` blocked on D21. |
| `local_rx` · `kernel_rx` · `crd` · `streaming_rx` | **done.** Three regularization defects found and fixed along the way (D22, D22.2, D24); `crd` checked and cleared (D22.3). |
| `fusion` | **done**, component-adaptive per D20. **Default weights do NOT meet §3A.9's accept criterion** — the grid search is still owed (D22.1). |
| Phase 4 | **done.** §4.1 registry accepts a config-only detector swap; §4.2 `calibrate_threshold_for_recall` returns `(threshold, fp_rate)`. §4.3 `roi_fusion` built. |
| Phase 5 L1 | in progress — harness + `cascade_recall_audit`. |
| Phase 7 `demo.py` | in progress. Runs on **HAD100, not EnMAP** (O11); steps 10–11 skip with a stated reason, since 3C and 3E are P2 and unbuilt. |

**The one thing P0 still owes beyond the two in progress:** §3A.9's fusion weight grid search
(D22.1), including carving out and naming the tuning split — ABU is 13 scenes, so tuning and
reporting on the same 13 is train-on-test.

**Why this order.** Six half-finished arms score worse than one complete system with its gaps
documented. §9 and §14 already record what is simulated and what is deferred; deferring
deliberately is a defensible position, and running out of time mid-arm is not.

**Parallelism note.** P1 and P2 are disjoint from P0 in the filesystem (`edge/`,
`change_detection/`, `quantum/` versus `segmentation/` and `anomaly/`), so a *separate*
contributor can take them without touching the critical path. They are P1/P2 for the person
holding the critical path, not for everyone.

**Nothing in P0 depends on another human.** Every input it needs is on disk. The only external
waits in the whole project are the QGIS eyeball on Phase 2, the SWIRB verification in D16, and
Phase 6 Tier B — none of which block P0.

**Hard ordering constraint (D11.3):** `harmonize` is not merely *upstream* of the background pool — it is the **join** that makes the pool a single tensor. AVIRIS-NG reduces to 276 bands and AVIRIS-Classic to 162, so `HAD100 download → main.py unpack → harmonize → pool → 3B.synth`. Pooling before harmonizing does not raise; it just cannot be stacked, and the first person to hit it will assume the download is corrupt.
**Longest pole:** 3B, because it depends on both `harmonize` and the HAD100 background pool download.
**Start-immediately, no dependencies:** `3D.profiling`, `3D.constrained_sim`, `3E.qiskit_basics`, and every `scripts/fetch_*.py`. Kick the downloads off on day one — they are large and they gate 3B.

---

## 12. Test Suite

`pytest`, run in CI on every commit. Contract tests are the ones that matter — they are the mechanism that keeps "frozen contracts" from being merely a document.

| File | Covers |
|---|---|
| `test_env.py` | Python 3.12, all imports, GDAL version |
| `test_contracts.py` | every C1–C6 validator, one pass + one fail each |
| `test_loader.py` | `.mat` / `.tif` / `.hdr` dispatch, synthetic-affine flag, tag round-trip, **HAD100 `.hdr` → `georef == "real"`**, and **D13.2 dtype safety**: an int16 fixture containing `-1` loads as `-1.0` and never `65535.0` (the evidenced hazard); a uint16 fixture containing `40000` loads as `40000.0` and never `-25536.0` (the unevidenced-but-guarded reverse); an unhandled dtype raises rather than silently casting |
| `test_benchmarks.py` | D13 invariants: Indian Pines has no georeference key (D2's premise); ABU is 13 scenes / 4-4-5 with the recorded per-scene band counts; HYDICE is 80×100×175 with 21 anomaly px — the anomaly scene, not the unmixing one (D13.3). |
| `test_harmonize.py` (built, D15) | **D9 band arithmetic pinned:** `len(CANONICAL_WL) == 211`, `water_mask().sum() == 27`, retained `== RETAINED_BANDS == 184`. **D11.4:** a real AVIRIS-Classic wavelength array sorts to strictly increasing (`sort_spectral_axis`); a NaN-poisoned wavelength array raises rather than silently sorting to a non-monotonic axis. **D11.3, the join:** real NG (425) and Classic (224) raw scenes both `harmonize` to `shape[-1] == RETAINED_BANDS == 184` and stack into one array. **D11.6, self-defending:** `coverage_ok` is True for both raw sensors (reproduces the plan's measured 0/184 uncovered) and False for a reconstructed `band_select`-style gapped axis (reproduces 43/184 uncovered exactly), and `harmonize` on that gapped axis raises rather than emitting a plausible-looking partly-fabricated cube. **Interpolation correctness:** the gather-based production path is cross-checked against `np.interp` directly. **NaN locality (D15):** a single poisoned source band propagates NaN only to the ≤2 target bands that actually bracket it, not to all 184 — the regression test for the matmul-vs-gather bug D15 found and fixed. |
| `test_scoring.py` | percentile + rank normalization, invertibility, NaN handling |
| `test_rx.py` | `streaming_rx == global_rx` (rtol 1e-5); local RX assertion fires |
| `test_geospatial.py` | pixel→world round-trip; EPSG:6933 area **and** geodesic perimeter tested separately at 8°N/35°N; EPSG:4326-only-at-export |
| `test_roi_provenance.py` | D5: profile set at birth; cross-profile ROIs NOT merged |
| `test_synth.py` | mixing-model correctness at `a=0` and `a=1`; mask exactness |
| `test_data_hygiene.py` | **D7: no real scene in any training manifest; no ID in both splits; no model scored on a dataset its spectra pools were harvested from (§3B.5b); and — D11.5 — no two patches sharing a source scene split across train/val** |
| `test_pipeline_e2e.py` | full run on a 32×32 fixture, `validate_geojson` green |
| `test_profiling.py` | every profiling record carries `measurement: "SIMULATED"` (§0.3); every checkpoint/result whose `train_host != "local"` carries the D12 cloud tag |

`test_data_hygiene.py` is the most important test in the suite, and it checks **three independent kinds of leakage**:
1. **Scene-level** — a real scene appearing in a training manifest, or a scene ID in both splits.
2. **Spectrum-level** — a model scored on a dataset that contributed target spectra to its training set (§3B.5b).
3. **Crop-level** — two patches derived from the same source scene landing on opposite sides of the train/val split. HAD100's four corner crops of an 81×81 scene overlap by ~47 px per axis (D11.5), so they are near-duplicates; a patch-level split inflates validation while every scene-level and spectrum-level check stays green.

None of the three follows from the others. A model can be trained on zero ABU scenes and still be leaking ABU, because target spectra harvested from ABU ground truth carry ABU's identity into the synthetic set. Check all three, separately, or the suite passes while a leak is live.

Train/test leakage would invalidate every number in the report, and it is the kind of mistake that stays invisible until someone else finds it.

---

## 13. Reporting Rules — apply to every document, slide, and spoken claim

1. **Simulated ≠ measured.** Every Phase 6 number is labelled `SIMULATED` until a real Pi exists (§0.3).
2. **Synthetic trains, real scores.** No metric is ever computed on synthetic data (D7). And **every segmentation number carries its spectra provenance** — leak-free is a property of the train/score pairing, not of the model, so the pairing is published beside the number (§3B.5b).
3. **Observed change ≠ attribution.** Physical change with dates and extents. Nothing further (Roadmap §1.9).
4. **Scoped novelty ≠ quantum advantage.** The claim is that the application is unexplored, backed by a dated literature search — not that quantum wins (Roadmap §1.10).
5. **Scoped-down models are named as scoped down.** The deep detector is a compact spectral-spatial transformer, not a graph transformer (D8).
6. **Learned-model scores are reported only where source wavelengths are verified.** Learned models are scored on HAD100 alone (O8 / D13.4). An ABU or HYDICE number is **never** implied, extrapolated, or interpolated for a learned model — those datasets ship no wavelength array, so no canonical-grid input exists for them. ABU/HYDICE numbers in any table are classical-detector numbers, and the table says which.
7. **Synthetic georeferencing is declared.** Indian Pines results carry `georef: "synthetic"` and no geospatial accuracy metric is drawn from them (D2).
8. **Unknown is a valid answer.** `class` defaults to `"UNKNOWN"` and stays there unless a classifier justifies otherwise (Roadmap §1.6).
9. **Stage-1 recall is reported separately** from end-to-end accuracy, always, with the false-negative audit attached.

---

## 14. Open Items — need a human decision

**Nothing here blocks starting.** O8 is closed — decided, not deferred. O9 is a research lead that gates nothing. The rest should be closed before Phase 5, and §8.0 is itself a Phase 5 prerequisite task rather than an open question.

| # | Item | Why it is open | Default if unanswered |
|---|---|---|---|
| O1 | **IBM Quantum account** | Not held (§0.2). Gates 3E.7. | **Default taken, 2026-08-22 (D27).** 3E is built and ships **simulator-only**. No hardware result appears in `docs/experiments.md`, `plan.md`, or any report. Note D27.4: at 8 qubits the simulator reproduces the "quantum" kernel exactly and 180× faster, so acquiring an account would change the *provenance* of these numbers, not their standing — 3E.7 remains a demonstration, never a source of a comparison-table row. |
| O2 | **Raspberry Pi 5 procurement** | Not owned. Gates Phase 6 Tier B. | Everything ships as `SIMULATED`. |
| O3 | **FPGA/NPU comparison device** | Not owned. Gates the accelerator arm. | The arm is dropped from the report and stated as dropped, not silently omitted. |
| O10 | **Credential expiry** | CDSE S3 key pairs are created with a caller-chosen expiry date (§4.1b). An expired key fails Phase 5 Level 3 at the download leg, and the symptom looks like a service outage rather than an auth problem. | Choose a long expiry at creation; diarise it. `scripts/check_credentials.py` reports configuration but **cannot** detect an expired-but-present key — only a real request can. Re-check before Phase 5. |
| O11 | **EnMAP download entitlement — CONFIRMED, blocking Phase 5 L2** | Resolved from open question to established fact on 2026-08-21 by `scripts/verify_access.py`. The account authenticates: a CAS login naming **no service** returns **HTTP 200 with a TGC cookie**. The *same* credentials in a login naming the EnMAP download service return **401**, and an HTTP-Basic request to the asset returns **403** whose body reads *"insufficient privileges to download this dataset"*. CAS authorises **per service**, so a missing entitlement is byte-indistinguishable from a wrong password unless the two logins are compared — which cost this project an hour of misdiagnosis. EnMAP archive access needs a **role assignment** via the Instrument Planning Portal, and `planning.enmap.org` / `enmap-planning.eoc.dlr.de` resolve in DNS but **refuse TCP connections**. §2.1 makes EnMAP half the background pool, so Phase 5 Level 2 is blocked until this clears. | **Diagnosis refined 2026-08-21.** There is no separate "EnMAP Access Service account": the register link on the EnMAP login page points at the *same* `sso.eoc.dlr.de/geoservice/selfservice` registration, so one account covers both. The EnMAP Access Service was subscribed in the Geoservice Permission Management App (`/eoc/kc/realms/geoservice/account/#/permissions`) and still shows in *Permissions you are subscribed to*, yet CAS continues to deny the service. Authentication succeeds and **authorization** fails — the browser error reads *"Service access denied due to missing privileges… you are actually logged into another of our services with another account."* Note the two systems: permissions live in **Keycloak** (realm `geoservice`), the download wall is **CAS** (`/eoc/auth/login`); a stale session in one can shadow the other, so test in a window with no other DLR tab open. 1. If it persists, contact **`eoc-ums-helpdesk@dlr.de`** (the address the login page itself gives for user-management problems) — *not* `erdbeobachtung@dlr.de`, which handles the 5 000-product contingent, a different question. 2. Re-run `scripts/verify_access.py` — it PASSes only on real TIFF magic bytes. 3. **Until it passes, build the background pool from AVIRIS-NG alone** and mark every affected §3B row accordingly. Do not write Level 2 code against assumed EnMAP access. |
| O12 | **SpectralEarth — candidate for the self-supervised arm** | The EOC Geoservice EnMAP catalogue (`geoservice.dlr.de/web/datasets/enmap`) holds four collections, not one. Besides L2A: **SpectralEarth** — 538 974 patches / 415 153 locations / 11 636 EnMAP scenes, ~3.3 TB, built expressly for self-supervised hyperspectral pretraining (arXiv 2408.08447) — and **HyBiomass** (EnMAP L2A + GEDI L4A labels). §5.2's masked-band SSL arm currently pretrains on the local background pool; SpectralEarth is the same idea three orders of magnitude larger. Caveats: it is **non-georeferenced** (fine for pretraining, useless for the geospatial arm), it is 3.3 TB against 7.8 GB currently on disk so only a subset is usable, and it sits behind the **same** EnMAP Access Service wall as L2A (verified 2026-08-21: all of L2A, SPECTRAL_EARTH and HYBIOMASS redirect to the same CAS login), so O11 blocks it too. **L0 Quicklooks is open with no login at all** (HTTP 200) but ships quicklooks and quality masks, not cubes — no use as a background pool. | Decide only after O11 clears and after §8.0 verifies EnMAP band/wavelength facts. Check `github.com/AABNassim/spectral_earth` for a subset or mirror first — a 3.3 TB pull is out of scope regardless. Do not let a foundation-model detour displace the critical path in §11. |
| ~~O4~~ | ~~**QGIS install**~~ | **CLOSED 2026-08-22 (D26).** QGIS 4.2.1 installed; `qgis/projects/phase2_verify.qgz` built and checked — affine plumbing confirmed, **Phase 2 exit signed off**. `demo_verify.qgz` additionally confirmed real georeferencing on HAD100 against OpenStreetMap (Dogpound Creek, Alberta). Feature-level Level 2 accuracy remains open — no independently-known target exists in that scene. | — |
| O5 | **Level-3 case-study site** | Depends on the `landcover` profile and Sentinel-2 coverage of a known dated event. | Pick during Phase 5 from actual data availability; record the selection rationale in `docs/validation.md`. |
| O6 | **Fusion weight tuning split** | Which ABU scenes are the tuning split vs. the reporting set. Must be fixed **before** any number is reported, or the fused AUC is optimistically biased. | Fix a seeded 4/9 split at the start of 3A.9, commit the split file, never revisit it. |
| ~~O8~~ | **ABU/HYDICE/Indian Pines carry no wavelength arrays (D13.4)** | **CLOSED — decided, not open.** Option 2 is adopted: learned models score on HAD100 only; classical detectors keep ABU + HYDICE + HAD100. Written into §3B.8, §13 rule 6, and the C1 contract. **No longer blocks 3B.** The cost — LODO cut from five arms to three, single-sensor generalization — is stated in §3B.8 rather than absorbed silently. |
| O9 | **Recover true per-scene wavelengths for ABU/HYDICE** | Their parent AVIRIS flight lines may be identifiable, and NASA's public per-flight calibration archive publishes wavelength/FWHM tables per flight. If a scene can be matched to its flightline, its real wavelength array is recoverable — which would restore ABU/HYDICE for learned models and un-suspend the two LODO arms. ABU's seven distinct band counts make the matching non-trivial: the retained-band subset differs per scene and is undocumented. | **Does not block anything currently scheduled.** Not on the critical path, not a Phase 3B gate, not a Phase 5 gate. Pursue opportunistically; if it fails or is never attempted, O8's decision stands unchanged and the plan is complete without it. |
| O7 | **PRISMA access** | Not held (§0.2 — listed separately from EnMAP/AVIRIS, which *are* held). Needs an ASI proposal, which has a lead time; AVIRIS and EnMAP are open and already cover Level 2. | Level 2 proceeds on EnMAP + AVIRIS. PRISMA is a bonus, not a dependency. |

---

## 15. Provenance of this plan

Derived from `roadmap.md` and `blueprint_upgrades_changelog.md`. Every deviation is recorded in §1 with its reason.

**Verified empirically, not assumed:**
- Python 3.14.7 dependency install failure, and its precise cause (`fiona` cp314 wheel absent) — §D1
- Python 3.12.13 full-stack install and import success, with versions — §D1
- **HAD100 — downloaded and every ENVI header parsed (D11).** `HAD100.zip`, 3 754 846 290 B,
  sha256 `6f91035543b7bc7806ebc555cd5411d320f2810f0af8dfcf20fe4f331227c19f`, `unzip -t` clean.
  Raw: 94 test + 260 NG + 262 Classic scenes. Unpacked by the repo's `main.py`: 100 test
  patches + 2 088 background patches. Raw test scenes span 8 distinct shapes up to 120×120
  (**not** uniformly 64×64); Classic background spans 7 shapes with mixed BIL/BIP interleave
  and mixed int16/float32. Band counts after `main.py`'s own `band_select`: NG 276, Classic 162.
  AVIRIS-Classic wavelengths non-monotonic in 262/262 files. All 616 scenes georeferenced
  (UTM/WGS-84). Two no-data sentinels in use (`-9999.0`, `1e-34`).
  Re-derivable via `scripts/verify_had100.py` → `docs/had100_verified.json`.
- **D9 band arithmetic — executed, not asserted.** `len(np.arange(400,2501,10))` = 211;
  water windows inclusive drop 27 → **184 retained**. The previously written `189` is
  unreachable under inclusive (184), half-open (186) or exclusive (188) masking.
- `uv venv --python 3.12` resolves to and fetches **3.12.13** on a host whose only system
  interpreter is 3.14.7 — D1's pin needs no system Python 3.12.
- Local GPU: GTX 1650, 4 GB VRAM — the constraint behind §D8
- **Indian Pines, ABU, HYDICE — downloaded and every variable loaded (D13).** Indian Pines
  (145,145,200) uint16 + (145,145) uint8 GT, **no CRS key of any kind — D2 confirmed**. ABU 13
  scenes 4/4/5, **seven band counts** (205×5, 191×2, 188×2, 193, 204, 207, 102), three dtypes,
  spatial 100×100 ×11 + 150×150 ×2, anomaly fraction 0.084–2.72 %. HYDICE is the **Michigan
  anomaly scene, 80×100×175**, 21 px in 10 components — not the 162-band Copperas Cove unmixing
  scene the plan previously described. **None of the three ships a wavelength array (O8).**
  Re-derivable via `scripts/verify_benchmarks.py` → `docs/benchmarks_verified.json`.
- **Phase 1–2 walking skeleton — built and run, not just specified (D14, 2026-08-21).** `.venv`
  synced against the full §1.2 lock; `.tooling/venv` deleted per §1.1. 54 tests pass; the full
  pipeline runs on `Indian_pines_corrected.mat` end to end. Two spec errors found by execution:
  §2.2's water-band indices target the wrong (220-band) layout for the shipped 200-band cube
  (D14.1), and a hand-rolled ENVI `map info` parse silently drops real, nonzero rotation
  (verified 33°/90° on two HAD100 headers) — confirmed wrong by diffing against GDAL's own ENVI
  driver, now fixed by delegating to it (D14.2). §2.10's QGIS visual check remains undone — see
  O4.

**Read off a web page, NOT verified against the data — treat as provisional.**
**Scheduled for file-opened verification as the Phase 5 prerequisite (§8.0).**

This tier exists because of what happened to HAD100. Its composition sat in the list above for
one draft, sourced from the project page, and the page was *right about the dataset* and
*wrong about the archive*: 94 files not 100, background 4× larger than stated, test scenes in
8 shapes not one, wavelengths non-monotonic, and every scene georeferenced. None of that is
discoverable by reading. **Assume the same class of error is live in every row below**, and
promote a row to the tier above only by loading the files.

- Li et al., IEEE TGRS 2023, doi:10.1109/TGRS.2023.3258067 — HAD100 citation; Drive + Baidu distribution
| Dataset | What is claimed | Unverified |
|---|---|---|
| **EnMAP L2A** ⚠️ *partially verified — D16* | **224 bands (91 VNIR + 133 SWIR), 418.42–2445.30 nm, backgroundValue −32768** — verified from ONE product's metadata (D16). 30 m GSD, DLR Geoservice STAC, 5 000-product cap. **8/184 canonical bands uncovered; `harmonize()` raises.** | band count · dtype · **wavelength availability** · tile structure · no-data handling |
| **Sentinel-2 L2A** | ~13 bands multispectral, Copernicus Data Space | band subset per product · **10/20/60 m resolution mixing** · Level 3 only |
| **AVIRIS / AVIRIS-NG** | 224 / 425 bands, NASA open portal | per-flightline band count · **wavelength file availability** (this is also what O9 depends on) |
| **USGS splib07** | `doi:10.5066/F7RR1WDJ`, plus ECOSTRESS/ASTER | record format · resampling convention · wavelength grid · licence terms |

**"Not yet flagged as a problem" is not "verified."** HAD100 sat in this tier for exactly one
draft, and its page was wrong about the archive in five ways. Assume the same is live here.

**Write `scripts/verify_<dataset>.py` for each, on the model of `verify_had100.py`: parse every
file, assert the invariants this plan depends on, exit non-zero on drift.** Until then §1.6's
"each fetcher verifies a SHA256 manifest" is satisfied for HAD100 only.

**Sources:** [HAD100 repo](https://github.com/ZhaoxuLi123/HAD100) · [HAD100 site](https://zhaoxuli123.github.io/HAD100/) · [ABU dataset](http://xudongkang.weebly.com/data-sets.html) · [EnMAP data access](https://www.enmap.org/data_access/) · [DLR EnMAP L2A geoservice](https://geoservice.dlr.de/web/datasets/enmap_l2_hsi) · [USGS Spectral Library v7](https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d)
