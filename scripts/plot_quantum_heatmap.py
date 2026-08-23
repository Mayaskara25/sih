"""Out-of-harness diagnostic: render a quantum anomaly-score heatmap for ONE
real HAD100 test scene.

This script is deliberately OUTSIDE the frozen 3E branch: it imports
quantum/ read-only and modifies nothing under it. Its output is a VISUAL aid,
not a reported result -- by default it runs a REDUCED VQC budget so it finishes
in minutes; any figure must carry that label. Reported numbers live only in
experiments/quantum_results/ and docs/experiments.md.

Usage:
  python scripts/plot_quantum_heatmap.py                        # fast visual
  python scripts/plot_quantum_heatmap.py --ansatz-reps 3 --maxiter 200   # frozen config
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from core.contracts import SceneMeta
from preprocessing.harmonize import harmonize
from quantum.data import (
    _DUMMY_CRS,
    _DUMMY_TRANSFORM,
    _NG_DATA,
    _NG_GT,
    _load_scene,
    build_split,
    test_scene_ids,
)
from quantum.vqc_encoder import VQCArm


def _pick_scene(min_anomalies: int = 8) -> str:
    """Smallest test-split scene carrying a usable number of anomaly pixels,
    so whole-scene scoring stays cheap."""
    best_key, best_id = None, None
    for sid in test_scene_ids():
        gm = sio.loadmat(_NG_GT / f"{sid}.mat")
        gt = gm[next(k for k in gm if not k.startswith("__"))].astype(bool)
        if int(gt.sum()) < min_anomalies:
            continue
        key = (gt.size, abs(int(gt.sum()) - 60))
        if best_key is None or key < best_key:
            best_key, best_id = key, sid
    if best_id is None:
        raise SystemExit("no test scene with enough anomaly pixels")
    return best_id


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-features", type=int, default=8)
    p.add_argument("--fm-reps", type=int, default=2)
    p.add_argument("--ansatz-reps", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    split = build_split(n_features=args.n_features)
    arm = VQCArm(n_features=split.n_features, reps=args.fm_reps,
                 ansatz_reps=args.ansatz_reps, maxiter=args.maxiter, seed=args.seed)

    scene = _pick_scene()
    print(f"[scene] {scene}")
    arm.fit(split)
    print(f"[fit] ansatz_reps={args.ansatz_reps} maxiter={args.maxiter} "
          f"evals={arm.n_objective_evals} wall={arm.fit_seconds:.1f}s")

    _, cube, gt, wl = _load_scene(_NG_DATA / f"{scene}.hdr")
    meta = SceneMeta(scene_id=scene, crs=_DUMMY_CRS, transform=_DUMMY_TRANSFORM,
                     wavelengths=wl, bad_bands=np.zeros(cube.shape[-1], dtype=bool),
                     gsd_m=1.0, source="had100", georef="real")
    cube_h, _ = harmonize(cube, meta)
    h, w, b = cube_h.shape
    flat = cube_h.reshape(-1, b).astype(np.float64)
    valid = ~np.isnan(flat).any(axis=-1)

    Xq = split.transformer.transform(flat[valid])
    scores = np.full(flat.shape[0], np.nan)
    scores[valid] = np.asarray(arm.score(Xq), dtype=np.float64)
    heat = scores.reshape(h, w)

    gt_flat = gt.reshape(-1)
    an = scores[valid & gt_flat]
    bg = scores[valid & ~gt_flat]
    print(f"[score] anomalies mean={np.nanmean(an):.4f}  background "
          f"mean={np.nanmean(bg):.4f}  gap={np.nanmean(an) - np.nanmean(bg):+.4f}")

    rgb = np.nan_to_num(cube_h[:, :, [30, 20, 10]], nan=0.0)
    lo, hi = np.percentile(rgb, [1, 99])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"HAD100 {scene}\nfalse colour (harmonized bands 30/20/10)")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title(f"ground truth ({int(gt.sum())} anomaly px)")
    im = axes[2].imshow(heat, cmap="hot", vmin=0.0, vmax=1.0)
    axes[2].set_title("VQC P(anomaly)")
    fig.colorbar(im, ax=axes[2], shrink=0.85, label="P(anomaly)")
    for ax in axes:
        ax.axis("off")
    label = ("DIAGNOSTIC VISUAL - reduced budget "
             f"(zz reps={args.fm_reps}, real_amplitudes reps={args.ansatz_reps}, "
             f"maxiter={args.maxiter}) - NOT the frozen reporting configuration"
             if (args.ansatz_reps, args.maxiter) != (3, 200)
             else "frozen configuration (zz reps=2, real_amplitudes reps=3, maxiter=200)")
    fig.suptitle(f"Quantum anomaly heatmap | HAD100 AVIRIS-NG | seed={args.seed}\n{label}",
                 fontsize=10)
    fig.tight_layout()
    out = f"experiments/quantum_results/quantum_heatmap_{scene}.png"
    fig.savefig(out, dpi=150)
    print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
