"""PLAN.md §3C.8 -- three-arm change-detection comparison.

Runs on SYNTHETIC bi-temporal pairs derived from a real HAD100 scene:
there is no real multi-temporal hyperspectral pair in data/ (EnMAP download
is blocked, O11), so every number this script produces is stamped
SYNTHETIC-PAIRS and must be quoted as such. The pseudo-change arm is a
controlled illumination gain -- that comparison (SAM vs raw differencing
under illumination shift) is the deliverable's reason to exist and remains
valid under controlled synthetic shift.

Arms: classical magnitude difference | SAM + physics fusion | Siamese net.
Metrics per arm: ROC-AUC (threshold-free) + precision/recall/F1 at the
threshold giving 95% recall on true changed pixels, plus the pseudo-change
rate (fraction of illumination-only pixels flagged at that same threshold).

Usage:
    python scripts/run_change_arms.py [--scene HDR] [--out experiments/change_arms]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.scoring import rank_normalize                     # noqa: E402
from change_detection.physics_fusion import (                  # noqa: E402
    difference_structure, fuse_change_signals)
from change_detection.siamese_net import (                     # noqa: E402
    make_change_pair, predict_change_map, train_siamese)
from change_detection.spectral_angle import spectral_angle     # noqa: E402
from change_detection.temporal_difference import magnitude_difference  # noqa: E402
from preprocessing.harmonize import reduce_bands               # noqa: E402
from preprocessing.raster_loader import load_scene             # noqa: E402
from preprocessing.registration import coregister_subpixel     # noqa: E402

DEFAULT_SCENE_GLOB = "data/benchmark/had100/HAD100/data/*/*.hdr"
CROP = 160
N_TARGETS = 5
ILLUMINATION_GAIN = 0.12
RECALL_TARGET = 0.95


def find_default_scene() -> Path:
    hits = sorted(ROOT.glob(DEFAULT_SCENE_GLOB))
    if not hits:
        raise SystemExit(
            f"no HAD100 scene under {DEFAULT_SCENE_GLOB} -- run fetch scripts first")
    return hits[0]


# --- metrics -----------------------------------------------------------------

def rank_auc(score: np.ndarray, gt: np.ndarray) -> float:
    """Threshold-free ROC-AUC via ranks (no sklearn dependency here)."""
    valid = np.isfinite(score)
    s, g = rank_normalize(np.where(valid, score, np.nan))[valid], gt[valid]
    n_pos = int(g.sum())
    n_neg = int((~g).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(s)) + 1          # average-free tie handling is fine here
    return float((ranks[g.astype(bool)].sum()
                  - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_recall(score_norm: np.ndarray, gt: np.ndarray,
                        recall_target: float) -> tuple[float, dict]:
    valid = np.isfinite(score_norm)
    s, g = score_norm[valid], gt[valid]
    order = np.argsort(s)[::-1]
    gs = g[order]
    tp_cum = np.cumsum(gs)
    recall = tp_cum / max(int(g.sum()), 1)
    idx = int(np.searchsorted(recall, recall_target))
    idx = min(idx, len(gs) - 1)
    thr = float(s[order][idx])
    pred = s >= thr
    tp = int((pred & (g == 1)).sum())
    fp = int((pred & (g == 0)).sum())
    fn = int(((~pred) & (g == 1)).sum())
    precision = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * precision * rec / max(precision + rec, 1e-9)
    return thr, dict(precision=precision, recall=rec, f1=f1)


def pseudo_change_rate(score_norm: np.ndarray, illum_mask: np.ndarray,
                       thr: float) -> float:
    vals = score_norm[illum_mask & np.isfinite(score_norm)]
    return float((vals >= thr).mean()) if vals.size else float("nan")


# --- arms --------------------------------------------------------------------

def arm_classical(t1, t2):
    raw = magnitude_difference(t1, t2)
    return rank_normalize(raw)


def arm_sam_fusion(t1, t2):
    sam = spectral_angle(t1, t2)
    structure = difference_structure(t1, t2, patch=7)
    cloud = np.zeros(t1.shape[:2], dtype=np.uint8)
    fused = fuse_change_signals(sam, structure, cloud)
    return rank_normalize(np.where(np.isfinite(fused), fused, np.nan))


def train_siamese_on_scene(scene: np.ndarray, *, crop: int, epochs: int,
                           seed: int):
    """Modest training set: crops from OTHER parts of the same scene."""
    rng = np.random.default_rng(seed)
    h, w, b = scene.shape
    spectra = scene[rng.integers(0, h, 8), rng.integers(0, w, 8)].astype(np.float32)

    def _pair(seed_off):
        r0 = int(rng.integers(0, h - crop))
        c0 = int(rng.integers(0, w - crop))
        bg = scene[r0:r0 + crop, c0:c0 + crop]
        t2, mask, meta = make_change_pair(
            bg, spectra, n_targets=rng.integers(2, 6),
            illumination_gain=float(rng.choice([0.0, 0.05, 0.10, 0.15])),
            seed=seed + seed_off)
        return bg.transpose(2, 0, 1), t2.transpose(2, 0, 1), mask

    train = [_pair(i) for i in range(48)]
    val = [_pair(100 + i) for i in range(8)]
    model, history = train_siamese(train, val, epochs=epochs, batch_size=8,
                                   device=None, in_channels=b, seed=seed,
                                   verbose=False)
    return model, history


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "change_arms")
    ap.add_argument("--epochs", type=int, default=15,
                    help="modest siamese training budget (§3C.5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    scene_path = args.scene or find_default_scene()
    print(f"scene: {scene_path}")
    cube, meta = load_scene(scene_path, source="had100")

    # PCA band reduction keeps every arm comparable and matches §3C.5's
    # 30-band siamese input; fit is scene-internal (one-off transform).
    cube, _tf = reduce_bands(cube, n_components=30)
    h, w, b = cube.shape
    crop = min(CROP, h, w)
    r0, c0 = (h - crop) // 2, (w - crop) // 2
    base = np.ascontiguousarray(cube[r0:r0 + crop, c0:c0 + crop])

    rng = np.random.default_rng(args.seed)
    spectra = base[rng.integers(0, crop, 8), rng.integers(0, crop, 8)]

    # -- build the bi-temporal pair ------------------------------------------
    # t2 = registered copy of t1 with a known residual misregistration test,
    # then implants + illumination gain.
    shifted = np.roll(base, shift=(3, -2), axis=(0, 1))
    aligned, reg_report = coregister_subpixel(
        base, shifted, meta, meta, upsample_factor=20)
    print(f"registration rmse: {reg_report['rmse_px']:.3f} px "
          f"(converged={reg_report['converged']})")

    t2, gt_mask, implant_meta = make_change_pair(
        aligned, spectra, n_targets=N_TARGETS,
        illumination_gain=ILLUMINATION_GAIN, seed=args.seed + 999)
    gt = gt_mask.astype(bool)

    # pseudo-change region: pixels far from any implanted target --
    # they differ between epochs ONLY through the illumination gain.
    from scipy.ndimage import binary_dilation
    illum_only = ~binary_dilation(gt, iterations=3)

    results = {"SYNTHETIC_PAIRS": True,
               "scene": str(scene_path.relative_to(ROOT)),
               "crop_shape": list(base.shape),
               "n_targets": N_TARGETS,
               "illumination_gain": ILLUMINATION_GAIN,
               "registration": {k: v for k, v in reg_report.items()
                                 if k != "shift_px"} | {"shift_px": reg_report["shift_px"]},
               "implant_meta": {k: v for k, v in implant_meta.items() if k != "targets"},
               "changed_px": int(gt.sum()),
               "illum_only_px": int(illum_only.sum()),
               "arms": {}}

    # -- arm 1: classical ------------------------------------------------------
    norm = arm_classical(base, t2)
    thr, m = threshold_at_recall(norm, gt, RECALL_TARGET)
    results["arms"]["classical_diff"] = {
        "auc": rank_auc(norm, gt), **m, "pseudo_change_rate":
        pseudo_change_rate(norm, illum_only, thr)}

    # -- arm 2: SAM + physics fusion ------------------------------------------
    norm = arm_sam_fusion(base, t2)
    thr, m = threshold_at_recall(norm, gt, RECALL_TARGET)
    results["arms"]["sam_physics_fusion"] = {
        "auc": rank_auc(norm, gt), **m, "pseudo_change_rate":
        pseudo_change_rate(norm, illum_only, thr)}

    # -- arm 3: siamese ---------------------------------------------------------
    print(f"training siamese ({args.epochs} epochs, modest budget)...")
    model, history = train_siamese_on_scene(cube, crop=64,
                                            epochs=args.epochs, seed=args.seed)
    prob = predict_change_map(model, base, t2, patch=64, stride=32)
    norm = rank_normalize(prob)
    thr, m = threshold_at_recall(norm, gt, RECALL_TARGET)
    results["arms"]["siamese_net"] = {
        "auc": rank_auc(norm, gt), **m, "pseudo_change_rate":
        pseudo_change_rate(norm, illum_only, thr),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"]}
    results["wall_clock_s"] = time.perf_counter() - t_start
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "_siamese_prob.npy", prob)
    (args.out / "change_arms_results.json").write_text(json.dumps(results, indent=2))

    print("\n=== THREE-ARM COMPARISON [SYNTHETIC-PAIRS] ===")
    hdr = f"{'arm':<22}{'AUC':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'pseudo-rate':>13}"
    print(hdr)
    for name, r in results["arms"].items():
        print(f"{name:<22}{r['auc']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['f1']:>8.4f}{r['pseudo_change_rate']:>13.4f}")
    print(f"\nresults -> {args.out / 'change_arms_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
