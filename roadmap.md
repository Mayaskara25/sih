# ROADMAP.md
## AI-Based Hyperspectral Anomaly Detection & Geospatial Semantic Mapping System
### Smart India Hackathon — Team Execution Plan

---

## 0. Purpose & Positioning

Build an offline-capable, edge-oriented geospatial intelligence system that processes hyperspectral/multispectral imagery, detects spectral anomalies and physical changes, segments suspicious regions at pixel level, and converts detections into georeferenced GIS outputs — with a small hybrid quantum-classical research branch that is never a dependency of the operational system.

**Position as:** *"An offline-capable, resource-aware edge intelligence pipeline that converts high-volume remote-sensing imagery into actionable geospatial intelligence at the point of acquisition."*

**Not:** "another AI image classifier."

**Core value props:** low bandwidth · low latency · offline operation · data locality · processing only suspicious regions with expensive models · standard GIS-compatible outputs · detecting unknown targets, not just known classes.

---

## 1. Core Design Principles (locked in — do not violate)

1. **Anomaly ≠ Change** — anomaly detection needs one image; change detection needs two, registered.
2. **Detection ≠ Segmentation** — detector says "interesting here"; segmentation says "these exact pixels."
3. **GIS ≠ AI** — QGIS visualizes/analyses georeferenced results; your AI produces those results.
4. **Edge ≠ ESP32 only** — Raspberry Pi is the compute node; ESP32 (if used at all) is telemetry/control only.
5. **Quantum ≠ production requirement** — always an experimental comparison branch.
6. **Unknown targets are a feature** — don't force every anomaly into a known class; default `"UNKNOWN"` is valid output.
7. **Validation is multi-level** — benchmark → real hyperspectral → real multitemporal geospatial case study.
8. **Edge value must be measured** — latency, bandwidth saved, resource consumption, accuracy retained, reported explicitly, not assumed.
9. **Never claim conflict attribution from image differences alone** — report observed physical change only.
10. **Never claim quantum advantage without demonstrating it experimentally.**

---

## 2. Data Contracts (freeze before writing any model code)

All six branches build against these — this is what prevents integration conflicts later.

| Artifact | Format |
|---|---|
| Preprocessed scene | `[H, W, Bands]` array, dtype `float32`, fixed band order, nodata flagged, rasterio profile (CRS + transform) attached |
| Anomaly score raster | Single-band `float32`, normalized 0–1, same georeferencing as input |
| Mask | `uint8`, `background=0`, `target=1` |
| Change score raster | Single-band `float32`, normalized 0–1, co-registered to t1 grid |
| GeoJSON output | Fields: `geometry`, `lat`, `lon`, `area`, `perimeter`, `anomaly_score`, `change_score`, `confidence`, `timestamp`, `source_scene`, `class` (default `"UNKNOWN"`) |
| CRS convention | Native scene CRS carried through pipeline; reproject to `EPSG:4326` only at GeoJSON export |
| Dev dataset (Week 1, whole team) | One fixed benchmark — Indian Pines |
| Hardware target | Raspberry Pi 5, 8GB |
| Dependency pins | `requirements.txt` / `pyproject.toml` locked on day 1 (GDAL/rasterio version drift is the #1 source of silent team conflicts) |

---

## 3. Repository Structure

```
project/
├── data/{raw,processed,benchmark}/
├── preprocessing/{raster_loader,normalize,cloud_mask,registration}.py
├── anomaly/{rx,local_rx,kernel_rx,crd,autoencoder,deep_detector,scoring,fusion}.py
├── change_detection/{temporal_difference,spectral_angle,physics_fusion,siamese_net,temporal_baseline}.py
├── segmentation/{train_unet,train_alt_arch,infer,postfilter}.py
├── geospatial/{polygonize,geojson,projections}.py
├── edge/{benchmark,onnx_inference,roi_pipeline,streaming,quantization,profiling}.py
├── quantum/{qiskit_basics,feature_map,vqc_encoder,quantum_autoencoder,quantum_kernel,classical_vs_quantum}.py
├── qgis/{styles,projects}/
├── experiments/{rx_vs_ae,edge_benchmarks,quantum_results,cascade_recall_audit}/
├── docs/{architecture,datasets,experiments,validation}.md
└── README.md
```

---

## 4. Team Structure (6 people + AI agents)

| Owner | Branch | Folders owned |
|---|---|---|
| **Person A + B** | Anomaly Detection | `anomaly/` |
| **Person C** | Segmentation | `segmentation/` |
| **Person D** | Change Detection | `change_detection/` |
| **Person E** | Edge/Systems + Geospatial | `edge/`, `geospatial/`, `qgis/` |
| **Person F** | Quantum | `quantum/` |

One person (rotate, or whoever is fastest) owns **Phase 2 — the walking skeleton** solo before the team splits into these branches. Everyone works against the frozen contracts in Section 2; no branch owner changes a contract format without a team-wide sync.

---

## 5. Execution Phases (step by step)

### **Phase 1 — Contracts, Skeleton, Environment** *(whole team, together)*
1. Agree and write down the data contracts (Section 2) — do not skip this even under time pressure.
2. Set up the repo structure (Section 3).
3. Pin dependencies (`requirements.txt`).
4. Install QGIS, Python env, Rasterio/GDAL/GeoPandas across all machines.
5. Everyone independently confirms they can load the Indian Pines benchmark and see an RGB composite — this is the "does everyone's environment actually work" checkpoint.

### **Phase 2 — Walking Skeleton (vertical slice)** *(1 person/pair, fast, not parallelized)*
Build the thinnest possible end-to-end path, using the simplest version of every stage:

1. Load Indian Pines → inspect bands, plot RGB composite, plot one pixel's spectral signature.
2. Implement **global RX** → produce anomaly score raster.
3. Threshold (start with a simple percentile cutoff) → binary mask.
4. Morphological cleanup → connected components.
5. Polygonize → attach georeferencing → export GeoJSON exactly matching the Section 2 schema.
6. Open GeoJSON + source raster in QGIS → visually confirm polygons land in the right place.
7. Run the same script on a normal laptop to sanity-check timing before anyone touches the Pi.

**Exit criterion:** a GeoJSON file with correctly-placed polygons, produced by a script that runs preprocessing → RX → threshold → polygonize → GeoJSON end to end. This is the spine everything else attaches to.

### **Phase 3 — Parallel Branch Build-Out** *(all 6, in parallel, against frozen contracts)*

**3A. Anomaly Detection** (Person A + B)
- Implement Local RX (dual concentric window) as the real baseline.
- Add Kernel-RX and CRD (Collaborative Representation Detector) as classical/mid-tier comparisons.
- Train autoencoder on **kernel-PCA** features (not linear PCA).
- Add one stronger deep detector for the top-tier comparison point.
- Build multi-signal fusion: RX + matched-filter score + spectral index + spatial score → single fused anomaly score.
- Refactor RX to incremental/streaming statistics (strip-by-strip, not full-cube batch).
- Deliverable: `experiments/rx_vs_ae/` comparison report (AUC, PR-AUC, precision, recall, F1 across all methods).

**3B. Segmentation** (Person C)
- Train lightweight U-Net on labeled patches.
- Train a second compact architecture (lightweight transformer-based segmenter) for real comparison.
- Add post-filtering: discard ROIs implausible in size/shape relative to expected targets.
- Deliverable: IoU/Dice comparison between the two architectures.

**3C. Change Detection** (Person D)
- Build automated sub-pixel co-registration (don't assume pre-aligned inputs).
- Replace raw differencing with **Spectral Angle Mapper** as the primary signal.
- Add physics-grounded fusion signal (patch-wise variance/entropy of the difference space) to separate real change from pseudo-change.
- Train a lightweight learned Siamese change-detection network as a third comparison arm.
- Deliverable: classical-diff vs. SAM+physics-fusion vs. learned-network comparison.

**3D. Edge/Systems + Geospatial** (Person E)
- Build the strip/line-based streaming architecture for the full pipeline (not just RX).
- Export trained models to ONNX; apply **mixed-precision quantization** (FP16 for covariance-sensitive stages, INT8 for threshold stages).
- Build the continuous per-module profiling harness (latency/RAM/CPU/power, tracked per stage, not just once at the end).
- Set up an FPGA or NPU-class dev board as a comparison point against Pi CPU inference, if hardware is available.
- Own the geospatial vectorization module and QGIS project/style files, integrating outputs from all other branches into the shared GeoJSON schema.

**3E. Quantum** (Person F)
- Implement PCA → 8–16 features → VQC/QAE pipeline (Qiskit Aer, local simulation).
- Add a quantum kernel method as a second quantum approach, alongside the variational circuit.
- Run at least one demonstration circuit on real IBM Quantum hardware (free tier) alongside simulator results.
- Write up the explicit novelty framing: no existing published work applies VQC/QAE feature encoding directly to hyperspectral anomaly detection — state this as the scoped claim, backed by classical-vs-quantum comparison numbers, not an advantage claim.

### **Phase 4 — Integration**
1. Each branch's best-performing method is wired into the main pipeline in place of the Phase 2 placeholder (RX → best anomaly method; simple threshold → fused scoring; single U-Net → best segmentation model, etc.).
2. Re-run the full pipeline end to end on the benchmark dataset — confirm the GeoJSON schema and QGIS visualization still work unchanged, since every branch built against the same frozen contracts.
3. Fuse anomaly score + change score + cloud/confidence score at the ROI level per the fusion logic in Section 6.

### **Phase 5 — Validation** *(three levels, do all three — see Section 7)*
1. Benchmark validation (Indian Pines / Pavia / Salinas).
2. Real hyperspectral validation (EnMAP / PRISMA / AVIRIS).
3. Real multitemporal geospatial case study (Sentinel-2 time series over a real Indian location; optional disaster/Kashmir case study framed strictly as *observed physical change*, never attribution).

### **Phase 6 — Edge Deployment & Benchmarking**
1. Deploy the full ONNX pipeline to Raspberry Pi 5 (8GB).
2. Run the streaming pipeline on real/benchmark scenes on-device.
3. Measure and record: per-scene latency, per-ROI latency, RAM, CPU, power (if measurable), bandwidth saved, % of image discarded by ROI screening.
4. Run the same benchmark on the FPGA/NPU comparison device if available.
5. Log stage-1 (anomaly detector) recall explicitly, and audit every missed real event in the validation set as a false-negative log (`experiments/cascade_recall_audit/`).

### **Phase 7 — Demo Assembly**
Target end-to-end demonstration:
1. Load real satellite/hyperspectral scene on Raspberry Pi.
2. Run local preprocessing.
3. Run best anomaly detector (with fused scoring).
4. Identify candidate ROIs; show stage-1 recall number live if possible.
5. Run best segmentation model on ROIs only.
6. Produce anomaly polygons with full attribute set.
7. Save GeoJSON locally; open in QGIS alongside source imagery.
8. Show internet is not required at any point in the inference stage.
9. Report latency/memory/bandwidth-saved numbers on screen.
10. If temporal data available, show previous-vs-current comparison with SAM+physics-fusion signal.
11. Present classical-vs-quantum comparison as an explicit research branch, not a production claim.

**Judge-facing narrative:** *"We do not send the entire high-volume remote-sensing scene to a remote service and wait for analysis. The edge node screens the scene locally, concentrates compute on suspicious regions, generates georeferenced intelligence, and transmits only actionable results. QGIS provides the standard GIS interface, while the quantum component is a research branch evaluating compact hybrid quantum-classical spectral representations."*

---

## 6. Pipeline Reference (technical spec, stage by stage)

| # | Stage | What it does | Where it runs |
|---|---|---|---|
| 1 | Data source | EnMAP/PRISMA/AVIRIS (real hyperspectral), Sentinel-2 (multispectral/temporal), benchmark datasets (dev) | Dev PC |
| 2 | Preprocessing | Radiometric handling, bad-band removal, normalization, cloud/shadow masking, registration | Trained/tuned on PC, executed on Pi |
| 3 | Anomaly detection | Local RX / Kernel-RX / CRD / autoencoder+kPCA / deep detector → fused score | RX-family + fusion: Pi. Training (AE, deep detector): PC |
| 4 | ROI extraction | Recall-calibrated threshold → morphology → connected components → cropped ROI patches | Pi |
| 5 | Segmentation | Lightweight U-Net + alt architecture, inference on ROI patches only | Training: PC (GPU). Inference: Pi, via ONNX |
| 6 | Geospatial vectorization | Polygonize → georeference → attach attributes → GeoJSON | Pi |
| 7 | QGIS | Visualize raster + score map + mask + GeoJSON; measurement, layer comparison, debugging georeferencing | Dev PC / GIS workstation only, never Pi |
| — | Temporal change detection | Co-registration → SAM + physics-fusion signal → classical diff / learned Siamese net → change score | Classical: Pi. Siamese net training: PC, inference: Pi via ONNX |
| — | Edge optimization | Mixed-precision quantization, ONNX export, streaming architecture, profiling | PC (export/quantize), Pi (benchmark) |
| — | Quantum experiment | PCA → VQC/QAE + quantum kernel → compare vs. classical AE | PC only (simulator + optional real IBM backend). Never reaches Pi |
| — | ESP32 (optional) | GPS / IMU / telemetry / power-health only | Standalone, alongside Pi, not part of the AI pipeline |

---

## 7. Validation Strategy

**Level 1 — Benchmark:** Indian Pines / Pavia / Salinas / Houston / Botswana. Metrics: ROC-AUC, PR-AUC, precision, recall, F1, pixel IoU where labels exist.

**Level 2 — Real hyperspectral:** EnMAP / PRISMA / AVIRIS. Goal: prove the method works on real sensor data, not just curated benchmarks.

**Level 3 — Real multitemporal geospatial case study:** Sentinel-2 time series over a real Indian location — co-register, mask, build temporal baseline, detect changes, compare to known events, export GeoJSON, inspect in QGIS. This demonstrates the full operational system.

---

## 8. Metrics to Report

**ML:** precision, recall, F1, ROC-AUC, PR-AUC, IoU/mIoU (segmentation), Dice.
**Systems:** inference latency, throughput, CPU/RAM usage, model size, power/energy, bandwidth before/after edge processing, % of image discarded by ROI screening.
**Geospatial:** location error, polygon IoU, area error, coordinate accuracy.
**Comparative:** RX family vs. CRD vs. deep detector · full-image vs. ROI inference · classical vs. quantum encoder · FP32 vs. mixed-precision quantized · Pi vs. FPGA/NPU (if available) · stage-1 recall vs. end-to-end accuracy.

---

## 9. Guardrails — What Not To Do

1. Don't call Sentinel-2 hyperspectral — it's multispectral.
2. Don't claim anomaly detection means only "compare today to yesterday."
3. Don't train a massive Transformer before a working end-to-end pipeline exists.
4. Don't try to run the full hyperspectral AI stack on ESP32.
5. Don't build a GIS engine from scratch — use QGIS/GDAL/GeoPandas.
6. Don't claim quantum advantage without demonstrating it experimentally.
7. Don't make conflict-attribution claims from image differences alone — report observed physical change only.
8. Don't manually label every pixel before an unsupervised baseline exists.
9. Don't optimize for model complexity before the walking skeleton (Phase 2) works end to end.
10. Don't make the quantum module a dependency of the operational edge pipeline.
11. Don't let any branch change a Section 2 data contract without a team-wide sync — this is the single most likely source of late-stage integration breakage.

---

## 10. Reference Sources

Copernicus Data Space / Sentinel-2 docs · EnMAP mission docs · PRISMA (ASI) · NASA AVIRIS/AVIRIS-NG · QGIS docs · Orfeo ToolBox · xBD disaster dataset · Qiskit / Qiskit Aer / IBM Quantum docs.

**Note:** Verify current API availability, dataset access policy, and QPU plan terms against official documentation before final submission — these change independently of this roadmap.
