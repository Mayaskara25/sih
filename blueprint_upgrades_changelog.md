# Upgrade Changelog — From Original Blueprint to Research-Grounded Plan

**Context:** No development-time constraint. Team of 6 + AI agents executing in parallel.
This document lists every change/addition to the original blueprint, stage by stage, with the reasoning behind each one.

---

## 1. Spectral Anomaly Detection

| Original Plan | Upgrade | Why |
|---|---|---|
| Global RX as the sole classical baseline | **Local RX (dual concentric window)** as the actual baseline | Global RX assumes one Gaussian background for the whole scene — fails on heterogeneous terrain. Local RX estimates background per-neighborhood instead. |
| RX vs. Autoencoder, two methods | **Three-tier comparative study**: statistical (RX, Local RX, Kernel-RX) → representation-based (CRD — Collaborative Representation Detector) → deep learning (autoencoder, and a stronger model like a graph-transformer-based detector) | Mirrors the actual structure the field uses to benchmark itself. Gives you a legitimate, citable comparison table instead of a two-method ad hoc test. |
| Autoencoder trained on raw PCA features | **Kernel-PCA preprocessing before the autoencoder**, not linear PCA | Published onboard-hyperspectral work (HYPSO group) specifically uses kernel-PCA preprocessing for autoencoder-based anomaly detection — captures non-linear structure linear PCA misses. |
| RX assumes full cube loaded in memory (batch) | **Streaming/incremental RX** — running mean & covariance updated strip-by-strip | Real pushbroom hyperspectral sensors emit data line by line, not as a complete cube. Batch RX doesn't reflect real sensor behavior or true low-memory edge operation. |

**New addition, not in original plan:** Multi-signal fusion at the scoring level — combine RX score + a matched-filter score (e.g., ACE) + a simple spectral index + local spatial-context score into one fused anomaly score, instead of relying on RX alone. Published edge/FPGA work fuses exactly these signal types for more robust candidate generation.

---

## 2. ROI Extraction / Cascade Logic

| Original Plan | Upgrade | Why |
|---|---|---|
| Threshold RX score, take top-N regions | **Recall-first threshold calibration** — deliberately over-trigger stage-1 candidates | Cascade architectures fail hardest when stage 1 misses something real, since stage 2 never gets a chance to see it. False positives here just cost extra (cheap) Pi compute; false negatives are unrecoverable. |
| No tracked stage-specific metric | **Explicitly measure and report stage-1 recall** as its own number, separate from end-to-end pipeline accuracy | Directly answers the literature's most common critique of coarse-to-fine systems: the first stage becomes an invisible bottleneck if nobody measures it separately. |
| Stage 1 and Stage 2 fully independent | **(Stretch, feasible with 6 people)** Add a lightweight ROI-recalibration link between RX output and segmentation input — segmentation confidence can optionally re-weight/adjust future RX thresholds | Published coarse-to-fine architectures (medical imaging, defect detection) show independently-trained stages are inconsistent and asynchronous; a differentiable/feedback link measurably improves this. |

---

## 3. Semantic Segmentation

| Original Plan | Upgrade | Why |
|---|---|---|
| Lightweight U-Net only | **Benchmark U-Net against a second compact architecture** (e.g., a lightweight transformer-based segmenter) as a real comparison, not a "later" footnote | No literature gap here specifically — this is just pulling a "later" item into main scope now that time/team capacity isn't the constraint. |
| Raw connected-component regions passed to segmentation, cleaned only by morphology | **Add anatomy/context-aware post-filtering** — discard candidate ROIs that are implausibly small/large relative to expected target size, or geometrically inconsistent with plausible target shapes | Cascaded segmentation literature (medical, defect-detection domains) shows this kind of lightweight, rule-based filtering meaningfully reduces false positives passed into the expensive stage, at near-zero cost. |

---

## 4. Temporal Change Detection

| Original Plan | Upgrade | Why |
|---|---|---|
| `difference = abs(image_t2 - image_t1)` as core signal | **Spectral Angle Mapper (SAM)** as the primary difference signal, not raw magnitude difference | SAM is inherently insensitive to uniform brightness/illumination scaling — directly targets the single most-cited cause of false "pseudo-changes" (illumination/seasonal shifts) in the change-detection literature. |
| Cloud/shadow masking + thresholding as the only false-alarm defense | **Add a physics-grounded decomposition signal** — patch-wise variance/entropy structure of the feature-difference space, fused alongside SAM and the cloud mask | Recent change-detection research shows genuine changes have measurably different statistical structure than pseudo-changes in the difference space — this gives you a principled extra signal instead of relying purely on masking heuristics. |
| Registration described as a preprocessing requirement, implementation unspecified | **Build actual automated sub-pixel co-registration**, don't assume "good enough" alignment | Misregistration is repeatedly cited as one of the most common real-world failure sources in change detection — worth building properly rather than approximating. |
| Classical change detection only; learned networks listed as a "later" stretch goal | **Build an actual lightweight learned change-detection network** (e.g., compact Siamese network with attention-based fusion) as a real comparison arm, not a stretch goal | With no time constraint and 6 people, this becomes a legitimate third comparison point (classical diff vs. SAM+physics-fusion vs. learned network) rather than something you never reach. |

---

## 5. Edge Deployment / Optimization

| Original Plan | Upgrade | Why |
|---|---|---|
| Quantize model uniformly to INT8 | **Mixed-precision quantization** — keep covariance/statistics-heavy stages at FP16, push only simple threshold-based stages to INT8 | Published low-power edge work on hyperspectral anomaly pipelines reports ~12x model size reduction and ~6x compute reduction using this mixed strategy while retaining ~99% of full-precision accuracy — a concrete, beatable benchmark target instead of guessing at a quantization scheme. |
| End-of-project single latency/memory benchmark | **Continuous, per-module profiling suite** — latency/RAM/CPU tracked per pipeline stage throughout development, not just once at the end | With 6 people building in parallel, a shared benchmark harness lets you catch regressions per-module immediately instead of discovering a bottleneck the week before the demo. |
| Raspberry Pi as the only edge target considered | **Add an FPGA or NPU-class device as a comparison point** (even a small dev board), benchmarked against plain Pi CPU inference | The actual state of the art in onboard hyperspectral processing is FPGA-accelerated, not general-purpose CPU. Even a modest comparison here substantially strengthens your "edge value should be measured" claims and shows you understand where the field is actually headed. |
| Whole-cube batch processing assumed at every stage | **Strip/line-based streaming architecture across the whole edge pipeline**, not just RX | Matches how real pushbroom sensors work, and is the only way to make "true low-memory edge operation" a real, demonstrable claim rather than an assumption. |

---

## 6. Quantum Branch

| Original Plan | Upgrade | Why |
|---|---|---|
| VQC/QAE experiment, simulator only | **Add a small real-hardware run** (IBM Quantum free-tier backend) alongside simulator results, for at least one demonstration circuit | Standard credibility move in the quantum-anomaly-detection literature: noiseless emulation *and* real hardware validation, not simulator-only. |
| Single quantum approach (VQC/QAE) | **Add a quantum kernel method as a second quantum approach**, alongside the variational circuit | The two established paradigms in quantum ML for anomaly detection are variational circuits and kernel-based circuits — covering both gives a genuinely complete comparative study, not just one architecture choice. |
| Quantum framed only as "not a dependency" | **Explicitly document that no existing published work applies VQC/QAE feature encoding directly to hyperspectral anomaly detection** — state this as your specific, scoped novelty claim | Turns an honest gap in prior art into a legitimate, defensible claim, rather than leaving the quantum branch's value implicit or vague. |

---

## 7. Geospatial / GIS Stage

**No changes required.** This is the one part of the pipeline where the surrounding literature has an actual, confirmed gap: essentially all pure hyperspectral-ML papers stop at pixel-space metrics (AUC, IoU) and never close the loop into georeferenced, GIS-usable output. Recommendation: **elevate this in your reporting/pitch** as the project's central differentiator — you now have literature evidence to back that framing, not just an assertion.

---

## 8. Validation & Reporting

| Original Plan | Upgrade | Why |
|---|---|---|
| End-to-end metrics only (precision, recall, F1, AUC, IoU) | **Stage-wise metrics**, especially stage-1 (RX) recall reported separately from final pipeline accuracy | Surfaces exactly where the cascade fails instead of hiding it inside one aggregate number. |
| No explicit tracking of missed detections | **False-negative audit log** — every real event in your validation set that RX/Local RX fails to flag gets explicitly recorded and reported | Directly answers the cascade-bottleneck critique with evidence rather than assertion — shows you tested for the failure mode the literature warns about. |
| Comparative structure ad hoc | **Adopt the field's own 3-way taxonomy** (statistical / representation-based / deep learning) as your comparison framework | Legitimate, citable structure — also gives you ready-made content for the paper/report rather than inventing your own categorization. |

---

## Summary of Net-New Components (not in original blueprint at all)

1. Local RX + Kernel-RX + representation-based detector (CRD) as additional classical/mid-tier baselines
2. Kernel-PCA preprocessing for the autoencoder
3. Streaming/incremental statistics architecture across RX and the edge pipeline generally
4. Multi-signal score fusion (RX + matched filter + spectral index + spatial score)
5. Recall-first cascade calibration + stage-wise recall reporting + false-negative audit
6. Spectral Angle Mapper + physics-grounded difference-space signal for change detection
7. Automated sub-pixel co-registration
8. A real learned change-detection network as a third comparison arm
9. Mixed-precision (FP16/INT8) quantization strategy, benchmarked against published numbers
10. Continuous per-module profiling harness
11. FPGA/NPU comparison point alongside the Raspberry Pi
12. Real quantum-hardware validation run + a second quantum approach (kernel method)
13. Explicit, scoped novelty claim for the quantum branch, in writing

---

## Team Allocation Note (6 people, given the above)

A natural split that keeps ownership aligned with the repo structure discussed earlier:

- **2 people — Anomaly detection branch:** RX family (global/local/kernel), CRD, autoencoder + kernel-PCA, deep detector, fusion scoring
- **1 person — Segmentation:** U-Net + second architecture, post-filtering
- **1 person — Change detection:** SAM, physics-fusion signal, co-registration, learned Siamese network
- **1 person — Edge/systems:** streaming architecture, quantization, profiling harness, FPGA/NPU comparison, geospatial vectorization + QGIS integration
- **1 person — Quantum branch:** VQC/QAE, quantum kernel method, simulator + real hardware validation

All six build against the frozen data contracts established at the start (preprocessing output, score format, mask convention, GeoJSON schema) — so this parallel expansion doesn't reintroduce the integration-conflict risk from the very first question in this conversation.
