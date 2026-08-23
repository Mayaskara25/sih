"""Out-of-harness diagnostic: side-by-side anomaly heatmaps on ONE real HAD100
test scene -- best arm of each family, scored on the SAME pixel population.

Read-only imports of the frozen 3E branch; modifies nothing under quantum/.
Reduced variational budgets so it finishes in minutes: every panel is a
DIAGNOSTIC VISUAL, not a reported number. Reported numbers live in
experiments/quantum_results/ and docs/experiments.md.

All arms are scored on the identical population: every valid anomaly pixel plus
a capped random background sample, reweighted to natural prevalence, so the
AP under each title is comparable across panels AND to score_scene_natural's
definition (D27.7).
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
from sklearn.metrics import average_precision_score, roc_auc_score

from core.contracts import SceneMeta
from preprocessing.harmonize import harmonize
from quantum.classical_baselines import MahalanobisArm, SVCArm
from quantum.classical_vs_quantum import _ScaledQuantumKernelArm
from quantum.data import (
    _DUMMY_CRS,
    _DUMMY_TRANSFORM,
    _NG_DATA,
    _natural_prevalence_sample,
    build_split,
    flightline_of,
    FLIGHTLINE_SPLIT,
    _load_scene,
)
from quantum.quantum_autoencoder import QuantumAutoencoderArm
from quantum.vqc_encoder import VQCArm


def _pick_scene(min_anomalies: int = 8) -> str:
    from quantum.data import test_scene_ids
    import scipy.io as sio

    best_key, best_id = None, None
    for sid in test_scene_ids():
        gm = sio.loadmat(_NG_GT_PATH(sid))
        gt = gm[next(k for k in gm if not k.startswith("__"))].astype(bool)
        if int(gt.sum()) < min_anomalies:
            continue
        key = (gt.size, abs(int(gt.sum()) - 60))
        if best_key is None or key < best_key:
            best_key, best_id = key, sid
    if best_id is None:
        raise SystemExit("no usable test scene")
    return best_id


def _NG_GT_PATH(sid: str) -> Path:
    from quantum.data import _NG_GT
    return _NG_GT / f"{sid}.mat"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--max-bg", type=int, default=1500)
    p.add_argument("--ansatz-reps", type=int, default=1)
    p.add_argument("--maxiter-vqc", type=int, default=60)
    p.add_argument("--maxiter-qae", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    split = build_split()
    scene = args.scene or _pick_scene()
    fl = flightline_of(scene)
    if fl not in FLIGHTLINE_SPLIT["test"]:
        raise SystemExit(f"{scene} is not a test-split scene")
    print(f"[scene] {scene}")

    _, cube, gt, wl = _load_scene(_NG_DATA / f"{scene}.hdr")
    meta = SceneMeta(scene_id=scene, crs=_DUMMY_CRS, transform=_DUMMY_TRANSFORM,
                     wavelengths=wl, bad_bands=np.zeros(cube.shape[-1], dtype=bool),
                     gsd_m=1.0, source="had100", georef="real")
    cube_h, _ = harmonize(cube, meta)
    h, w, b = cube_h.shape
    flat = cube_h.reshape(-1, b).astype(np.float64)
    valid = ~np.isnan(flat).any(axis=-1)
    gt_flat = gt.reshape(-1)

    idx, weight = _natural_prevalence_sample(gt_flat, valid, max_bg_per_scene=args.max_bg,
                                             seed=args.seed)
    X = split.transformer.transform(flat[idx])
    y = gt_flat[idx].astype(np.uint8)
    print(f"[eval] {idx.size} pixels ({int(y.sum())} anomalies)")

    arms = [
        ("rx_8feat", MahalanobisArm(seed=args.seed), None),
        ("classical_svc", SVCArm(seed=args.seed), None),
        ("vqc", VQCArm(n_features=split.n_features, reps=2,
                       ansatz_reps=args.ansatz_reps, maxiter=args.maxiter_vqc,
                       seed=args.seed),
         f"reps={args.ansatz_reps}, maxiter={args.maxiter_vqc}"),
        ("qae", QuantumAutoencoderArm(n_features=split.n_features, n_latent=4, reps=2,
                                      ansatz_reps=args.ansatz_reps,
                                      maxiter=args.maxiter_qae, seed=args.seed),
         f"reps={args.ansatz_reps}, maxiter={args.maxiter_qae}"),
        ("quantum_kernel_valscale", _ScaledQuantumKernelArm(kind="zz", reps=2,
                                                            angle_scale=0.5,
                                                            seed=args.seed),
         "zz reps=2, angle_scale=0.5"),
    ]

    results = []
    heat = np.full((h * w,), np.nan)
    for name, arm, note in arms:
        t0 = __import__("time").perf_counter()
        arm.fit(split)
        s = np.asarray(arm.score(X), dtype=np.float64)
        ap = average_precision_score(y, s, sample_weight=weight)
        roc = roc_auc_score(y, s)
        wall = __import__("time").perf_counter() - t0
        results.append((name, ap, roc, wall, note))
        panel = heat.copy()
        panel[idx] = s
        results[-1] = (name, ap, roc, wall, note, panel.reshape(h, w))
        print(f"[arm] {name:26s} AP={ap:.4f} ROC={roc:.4f} wall={wall:.1f}s")

    rgb = np.nan_to_num(cube_h[:, :, [30, 20, 10]], nan=0.0)
    lo, hi = np.percentile(rgb, [1, 99])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-9), 0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"HAD100 {scene}\nfalse colour")
    axes[0, 1].imshow(gt, cmap="gray")
    axes[0, 1].set_title(f"ground truth ({int(gt.sum())} px)")

    slots = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    for (r, c), (name, ap, roc, wall, note, panel) in zip(slots, results):
        ax = axes[r, c]
        im = ax.imshow(panel, cmap="hot", vmin=0.0, vmax=1.0)
        ax.set_title(f"{name}\nAP={ap:.3f}  ROC={roc:.3f}")
        fig.colorbar(im, ax=ax, shrink=0.75)

    summary = "\n".join(
        f"{name:26s} AP={ap:.3f}" for name, ap, *_ in sorted(results, key=lambda r: -r[1]))
    axes[1, 3].axis("off")
    axes[1, 3].set_title("natural-prevalence AP (weighted)")
    axes[1, 3].text(0.02, 0.55, summary, family="monospace", fontsize=11,
                    va="top", transform=axes[1, 3].transAxes)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 3]]:
        ax.axis("off")
    for r, c in slots:
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

    fig.suptitle(
        f"HAD100 quantum-vs-classical heatmap comparison | scene {scene} "
        f"(held-out test flightline)\n"
        f"DIAGNOSTIC VISUAL - reduced variational budgets (vqc/qae ansatz_reps="
        f"{args.ansatz_reps}, maxiter={args.maxiter_vqc}/{args.maxiter_qae}) "
        f"- NOT the frozen reporting configuration",
        fontsize=11)
    fig.tight_layout()
    out = f"experiments/quantum_results/quantum_comparison_heatmap_{scene}.png"
    fig.savefig(out, dpi=140)
    print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
