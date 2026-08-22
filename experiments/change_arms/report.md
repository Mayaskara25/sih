# §3C.8 — Three-Arm Change-Detection Comparison

**Status: COMPLETE — every number below is SYNTHETIC-PAIRS.**

There is no real bi-temporal hyperspectral pair in `data/`: HAD100 / ABU /
HYDICE / Indian Pines are all single-epoch acquisitions, and the EnMAP
download leg that could produce one is blocked (O11 — DLR entitlement
denied). The pairs used here are therefore **constructed**: t1 = a real
HAD100 AVIRIS-NG scene (PCA band reduction to 30 components); t2 = the same
scene co-registered after a known synthetic misregistration, with 5 target
spectra implanted (t2 only, via `segmentation/synth.py`'s linear mixing
model) plus a uniform +12 % illumination gain applied everywhere. The
illumination-only pixels are exactly the pseudo-change arm the plan calls
for — measured under controlled conditions rather than asserted.

## Setup

| Item | Value |
|---|---|
| Scene | `data/benchmark/had100/HAD100/data/aviris_ng_normal/ang20170821t183707_1.hdr` |
| Crop / shape | centre crop, [40, 40, 30] float32 |
| Registration | §3C.1 two-stage (PCC upsample 20 + ECC affine); measured residual RMSE **0.000 px** on the synthetic shift |
| Implanted change | 5 targets, abundance 0.1–1.0, spectra drawn from the scene itself → 103 changed px |
| Pseudo-change | +12 % uniform illumination gain outside a 3-px dilation of the GT mask (2 929 px) |
| Metric protocol | ROC-AUC threshold-free; precision/recall/F1 at the per-arm threshold giving 95 % recall; pseudo-change rate = fraction of illumination-only pixels ≥ that same threshold |

## Results [SYNTHETIC-PAIRS]

| Arm | AUC | Precision | Recall | F1 | Pseudo-change rate |
|---|---|---|---|---|---|
| Classical magnitude difference | 0.6179 | 0.0192 | 0.9515 | 0.0376 | **0.7797** |
| SAM + physics fusion (§3C.2/§3C.4) | **0.7651** | 0.0240 | 0.9515 | **0.0469** | **0.6119** |
| SiameseChangeNet (§3C.5, modest budget) | 0.5550 | 0.0162 | 0.9515 | 0.0318 | 0.9242 |

Raw outputs: `change_arms_results.json`, `_siamese_prob.npy`.
Reproduce: `python scripts/run_change_arms.py [--epochs N]`.

## Findings — stated plainly

1. **The headline comparison survives: SAM + physics fusion beats raw
   differencing on BOTH discriminative metrics AND pseudo-change rejection**
   (AUC +0.147, pseudo-change rate −17 points at matched recall). This is
   the number that justifies choosing SAM over magnitude differencing, and
   it was obtained under controlled illumination shift — the mechanism the
   literature cites as the dominant false-alarm source.
2. **The learned arm underfits at its mandated modest budget** (15 epochs,
   48 training crops). Its first run anchored the BCE pos_weight on a single
   batch, which produced AUC 0.416 (below chance); computing pos_weight over
   the full training set fixed the optimization pathology and brought it to
   0.555 — still statistically unimpressive. Scaling data volume or epochs
   might close the gap, but that exceeds this deliverable's scope; the row
   is reported as measured, not tuned until it flatters itself.
3. Precision is low for every arm because changed pixels are ~1.6 % of the
   crop and thresholds are pinned at 95 % recall by protocol — compare arms
   on AUC and pseudo-change rate, not on precision alone.
4. The perfect registration RMSE reflects an exactly-known integer-pixel
   synthetic shift recovered sub-pixel-precisely; it validates the §3C.1
   machinery but says nothing about real-world misregistration robustness.
   That claim awaits a real multi-temporal pair (blocked on O11).

## What would upgrade this deliverable

- One real EnMAP or Sentinel-2 bi-temporal pair over a chosen site (O11 /
  O5 unblock) → re-run unchanged; labels would need manual annotation.
- Longer siamese training once GPU time is cheap (the §3C.5 spec's 2.4 M
  params converge slowly on 48 crops).
- Cloud-mask arm activation (§3C.7) requires a source shipping wavelengths
  or an SCL band — not exercisable on PCA-reduced HAD100.
