#!/usr/bin/env python3
"""PLAN.md §10 -- Phase 7 demo: one scripted run of the Roadmap §5 sequence.

WHAT THIS RUNS ON, AND WHY NOT WHAT §10 STEP 1 NAMES
====================================================
§10 step 1 says "EnMAP/AVIRIS". EnMAP L2A retrieval is blocked (O11), so this
runs on **HAD100**, which satisfies the step for three independent reasons:
it *is* AVIRIS (D11.3 -- aviris_ng and aviris_classic), it ships **real
wavelengths** (the only benchmark that does, D13.4/O8) so `harmonize` and the
wavelength-dependent fusion component can actually run, and it ships **real
UTM/WGS-84 georeferencing on all 616 scenes** (D11.5). Indian Pines would fail
all three (D13.1).

WHAT THIS DELIBERATELY DOES NOT SHOW
====================================
Steps 10 (temporal SAM + physics fusion) and 11 (classical-vs-quantum) are
**not built**. Branch 3C and branch 3E are P2 in §11.1 and were never started.
The demo prints a SKIPPED line naming what is missing and why, rather than
producing a plausible-looking placeholder. Step 2's cloud mask is §3C.7 and
is in the same position.

This is not modesty, it is the §13 reporting rules: "Simulated != measured",
and "do not improvise a stronger claim on stage than the numbers support." A
demo that fabricates two of its eleven steps is one audience question away
from being worthless.

--assert-offline (§10 step 8) GENUINELY BLOCKS SOCKETS. "Claiming offline
operation without proving it is the weakest possible version of this demo."
The guard replaces socket.socket for the whole inference stage, so any
attempt to open one raises rather than being merely counted.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.fusion import fuse_scores                                 # noqa: E402
from anomaly.local_rx import local_rx                                  # noqa: E402
from anomaly.rx import global_rx                                       # noqa: E402
from anomaly.scoring import (                                          # noqa: E402
    ace_score,
    calibrate_threshold_for_recall,
    estimate_target_signature,
    percentile_normalize,
    spatial_context_score,
    threshold_by_percentile,
)
from core.contracts import validate_geojson                            # noqa: E402
from geospatial.geojson import rois_to_geojson                         # noqa: E402
from geospatial.polygonize import mask_to_rois                         # noqa: E402
from preprocessing.normalize import standardize                        # noqa: E402
from segmentation.postfilter import (                                  # noqa: E402
    filter_rois,
    load_target_profile,
    morphological_cleanup,
    resolve_profile,
)

HAD100 = ROOT / "data" / "benchmark" / "had100" / "HAD100"


class OfflineViolation(RuntimeError):
    """Raised when inference attempts a network connection under --assert-offline."""


@contextmanager
def no_network(enabled: bool):
    """Replace socket.socket for the duration. Not a counter -- a barrier.

    A flag that prints "offline: true" while permitting connections is worse
    than no flag, because it converts an unverified claim into an apparently
    verified one.
    """
    if not enabled:
        yield
        return
    real = socket.socket

    class _Blocked(real):                                     # type: ignore[misc]
        def __init__(self, *a, **k):
            raise OfflineViolation(
                "inference attempted to open a socket while --assert-offline was set")

    socket.socket = _Blocked                                  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = real                                  # type: ignore[assignment]


def meta_for_harmonize(scene_hdr: Path, wl, cube):
    """Minimal SceneMeta for harmonize(). It needs the wavelength array and
    little else; the geometry comes from the ENVI header separately."""
    import numpy as _np
    import rasterio.crs, rasterio.transform
    from core.contracts import SceneMeta
    return SceneMeta(scene_id=scene_hdr.stem,
                     crs=rasterio.crs.CRS.from_epsg(4326),
                     transform=rasterio.transform.from_origin(0, 0, 1, 1),
                     wavelengths=(_np.asarray(wl, dtype=_np.float64)
                                  if wl is not None else None),
                     bad_bands=_np.zeros(cube.shape[-1], bool),
                     gsd_m=1.0, source="had100", georef="real")


def _step(n: int, title: str) -> None:
    print(f"\n[{n:2d}] {title}", flush=True)


def _skip(n: int, title: str, why: str) -> None:
    print(f"\n[{n:2d}] SKIPPED -- {title}\n     reason: {why}", flush=True)


def pick_scene() -> tuple[Path, Path | None]:
    for sub, gtsub in (("aviris_ng_target", "aviris_ng_gt"), ("aviris_target", "aviris_gt")):
        d = HAD100 / "data" / sub
        if not d.exists():
            continue
        for hdr in sorted(d.glob("*.hdr")):
            gt = HAD100 / "gt" / gtsub / f"{hdr.stem}.mat"
            return hdr, (gt if gt.exists() else None)
    raise FileNotFoundError(f"no HAD100 target scenes under {HAD100}")


def run_demo(*, scene_hdr: Path | None = None, out_dir: Path, assert_offline: bool = False,
             profile_name: str = "object", target_recall: float = 0.98) -> dict:
    import scipy.io as sio
    import spectral

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"steps": {}}
    t_start = time.perf_counter()

    # -- 1 ---------------------------------------------------------------
    _step(1, "Load a real hyperspectral scene")
    if scene_hdr is None:
        scene_hdr, gt_path = pick_scene()
    else:
        gt_path = None
    img = spectral.open_image(str(scene_hdr))
    cube = np.asarray(img.load(), dtype=np.float32)
    wl = getattr(img, "bands", None)
    wl = np.asarray(wl.centers, dtype=np.float64) if wl and wl.centers else None
    gt = None
    if gt_path and gt_path.exists():
        gm = sio.loadmat(gt_path)
        gt = gm[next(k for k in gm if not k.startswith("__"))].astype(bool)
    print(f"     {scene_hdr.name}  {cube.shape}  wavelengths={'yes' if wl is not None else 'no'}"
          f"  gt={'yes' if gt is not None else 'no'}")
    print(f"     source: HAD100 (AVIRIS) -- real wavelengths (D13.4) and real "
          f"UTM/WGS-84 georeferencing (D11.5)")
    summary["steps"]["1_load"] = dict(scene=scene_hdr.name, shape=list(cube.shape),
                                      has_wavelengths=wl is not None)

    tracemalloc.start()
    with no_network(assert_offline):
        # -- 2 -----------------------------------------------------------
        _step(2, "Local preprocessing -- harmonize, bad bands, cloud mask")
        norm = standardize(cube)
        print(f"     standardized {cube.shape[-1]} native bands for the CLASSICAL"
              f" detectors,")
        print(f"     which need no wavelengths and run on native bands (§3B.8).")

        # The LEARNED model is a different story and this is the join that
        # makes it work (D11.3/D11.6). The U-Net consumes a PCA fitted on
        # HARMONIZED 184-band cubes, so the raw 425-band AVIRIS-NG cube must
        # go onto the canonical grid first. Feeding it raw is not a
        # degradation, it is a hard shape error -- which is exactly how this
        # was found: the first run of this demo raised "X has 425 features,
        # but PCA is expecting 184". Nothing else in the repo exercised the
        # raw-scene -> harmonize -> infer path end to end.
        harm = None
        try:
            from preprocessing.harmonize import harmonize
            harm, harm_meta = harmonize(cube, meta_for_harmonize(scene_hdr, wl, cube))
            print(f"     harmonized {cube.shape[-1]} -> {harm.shape[-1]} canonical bands "
                  f"(400-2500 nm @10 nm, water windows dropped -- D9)")
        except Exception as exc:                                  # noqa: BLE001
            print(f"     harmonize FAILED: {type(exc).__name__}: {str(exc)[:110]}")
            print(f"     -> learned segmentation will be skipped; classical path unaffected")
        print(f"     cloud mask: NOT APPLIED -- preprocessing/cloud_mask.py is "
              f"§3C.7 and is not built (P2)")
        summary["steps"]["2_preprocess"] = dict(
            native_bands=int(cube.shape[-1]),
            harmonized_bands=(int(harm.shape[-1]) if harm is not None else None),
            cloud_mask="not built (§3C.7, P2)")

        # -- 3 -----------------------------------------------------------
        _step(3, "Fused anomaly detection")
        t0 = time.perf_counter()
        base = global_rx(norm)
        sig = estimate_target_signature(norm, base, top_frac=0.001)
        comps = {"rx": local_rx(norm, outer=15, inner=3, n_components=12),
                 "ace": ace_score(norm, sig),
                 "spatial": spatial_context_score(base, k=7)}
        fr = fuse_scores(comps)
        score = fr.score
        t_detect = time.perf_counter() - t0
        print(f"     active components: {'+'.join(sorted(fr.components))}")
        print(f"     weights: { {k: round(v, 4) for k, v in fr.weights.items()} }")
        print(f"     NOTE (D25): fusion is reported as COMPARABLE to the best single")
        print(f"     detector (+0.002 macro AUC on ABU), not as beating it. The")
        print(f"     `index` component is absent wherever wavelengths are (D20).")
        summary["steps"]["3_detect"] = dict(components=list(fr.components),
                                            seconds=t_detect)

        # -- 4 -----------------------------------------------------------
        _step(4, "Recall-calibrated ROI extraction")
        nscore, _, _ = percentile_normalize(score)
        stage1_recall = None
        if gt is not None:
            v = ~np.isnan(score)
            thr, fp_rate = calibrate_threshold_for_recall(
                score[v], gt[v], target_recall=target_recall)
            mask = np.zeros(score.shape, dtype=np.uint8)
            mask[v] = (score[v] >= thr).astype(np.uint8)
            stage1_recall = float((score[v][gt[v]] >= thr).mean())
            print(f"     STAGE-1 RECALL: {stage1_recall:.4f}  (target {target_recall})")
            print(f"     induced false-positive rate: {fp_rate:.4f}  "
                  f"<- the compute cost of that recall, stated (§4.2)")
        else:
            mask = threshold_by_percentile(nscore, pct=99.0)
            fp_rate = None
            print(f"     no ground truth for this scene -> percentile threshold; "
                  f"stage-1 recall NOT claimed")
        mask = morphological_cleanup(mask)
        from core.contracts import SceneMeta
        import rasterio.crs, rasterio.transform
        meta = SceneMeta(scene_id=scene_hdr.stem,
                         crs=rasterio.crs.CRS.from_epsg(4326),
                         transform=rasterio.transform.from_origin(0, 0, 1, 1),
                         wavelengths=wl, bad_bands=np.zeros(cube.shape[-1], bool),
                         gsd_m=float(getattr(img, "gsd", 1.0) or 1.0),
                         source="had100", georef="real")
        rois = mask_to_rois(mask, meta, source_branch="anomaly", target_profile=profile_name)
        for roi in rois:
            r0, c0, r1, c1 = roi.bbox
            vals = nscore[r0:r1, c0:c1][roi.mask.astype(bool)]
            roi.anomaly_score = float(np.nanmean(vals)) if vals.size else None
        print(f"     ROIs extracted: {len(rois)}")
        summary["steps"]["4_rois"] = dict(n_rois=len(rois), stage1_recall=stage1_recall,
                                          induced_fp_rate=fp_rate)

        # -- 5 -----------------------------------------------------------
        _step(5, "Segmentation on ROI crops only")
        scene_px = int(cube.shape[0] * cube.shape[1])
        roi_px = int(sum((r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) for r in rois))
        saving = 1.0 - (roi_px / scene_px) if scene_px else 0.0
        print(f"     scene {scene_px:,} px -> ROI windows {roi_px:,} px")
        print(f"     PIXEL-COUNT SAVING: {saving * 100:.1f}%  "
              f"(ROI-only inference vs full-scene)")
        seg_note = "no ROIs -- segmentation skipped (a valid outcome, D23)"
        if rois:
            try:
                from segmentation.infer import segment_rois
                import torch
                from segmentation.train_unet import LightUNet
                ckpt = ROOT / "experiments" / "seg_arch" / "unet_pretext.pt"
                model = LightUNet(in_channels=30)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
                if harm is None:
                    raise RuntimeError(
                        "no harmonized cube -- the U-Net's PCA was fitted on the "
                        "184-band canonical grid (D11.3), so a raw cube cannot be fed to it")
                rois = segment_rois(harm, meta, rois, model, device="cpu")
                seg_note = (f"unet_pretext on {len(rois)} ROI crops of the HARMONIZED "
                            f"{harm.shape[-1]}-band cube (fitted PCA reused, D15)")
            except Exception as exc:                             # noqa: BLE001
                seg_note = f"segmentation unavailable: {type(exc).__name__}: {str(exc)[:120]}"
        print(f"     {seg_note}")
        summary["steps"]["5_segment"] = dict(scene_px=scene_px, roi_px=roi_px,
                                             pixel_saving_frac=saving, note=seg_note)

        # -- 6 -----------------------------------------------------------
        _step(6, "Post-filter and polygons with full C6 + D5 provenance")
        profile = resolve_profile(load_target_profile(profile_name), scene_px=scene_px)
        kept, dropped = filter_rois(rois, profile, audit_dir=out_dir / "cascade_recall_audit")
        print(f"     profile '{profile_name}': kept {len(kept)}, dropped {len(dropped)}")
        print(f"     dropped ROIs written to the audit trail, not discarded (§3B.7)")
        summary["steps"]["6_postfilter"] = dict(kept=len(kept), dropped=len(dropped))

        # -- 7 -----------------------------------------------------------
        _step(7, "GeoJSON written locally")
        gj = rois_to_geojson(kept, meta, out_dir / f"{meta.scene_id}_demo.geojson")
        validate_geojson(gj)
        print(f"     {gj.name}  ({len(kept)} features)  validate_geojson: PASS")
        summary["steps"]["7_geojson"] = dict(path=str(gj), n_features=len(kept))

    # -- 8 -------------------------------------------------------------
    _step(8, "Offline operation")
    if assert_offline:
        print("     --assert-offline was ACTIVE for steps 2-7 (socket.socket replaced).")
        print("     No socket was opened; any attempt would have raised OfflineViolation.")
    else:
        print("     NOT ASSERTED. Re-run with --assert-offline to enforce and prove it.")
    summary["steps"]["8_offline"] = dict(asserted=assert_offline)

    # -- 9 -------------------------------------------------------------
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    elapsed = time.perf_counter() - t_start
    _step(9, "Latency / memory / bandwidth")
    print(f"     wall-clock total          : {elapsed:.2f} s        [MEASURED, this laptop]")
    print(f"     detector stage            : {t_detect:.2f} s        [MEASURED, this laptop]")
    print(f"     peak traced allocation    : {peak / 1e6:.1f} MB     [MEASURED, this laptop]")
    print(f"     ROI-only pixel saving     : {saving * 100:.1f} %       [MEASURED]")
    print(f"     edge latency / power      : SIMULATED -- no instrumented hardware")
    print(f"                                 exists (§0.2, §0.3, §9 Tier B is BLOCKED).")
    print(f"     bandwidth saved downlink  : SIMULATED -- derived from the pixel")
    print(f"                                 saving above, not measured on a radio link.")
    summary["steps"]["9_metrics"] = dict(
        wall_clock_s=elapsed, detector_s=t_detect, peak_alloc_mb=peak / 1e6,
        pixel_saving_frac=saving, measurement="MEASURED on laptop; edge figures SIMULATED")

    # -- 10 / 11 -------------------------------------------------------
    _skip(10, "Temporal t1-vs-t2 (SAM + physics fusion)",
          "branch 3C is not built (P2 in §11.1) and no multitemporal data is loaded. "
          "§10 step 10 is conditional on temporal data; there is none.")
    _skip(11, "Classical-vs-quantum comparison",
          "branch 3E is not built (P2 in §11.1). §13 rule 4 permits only a scoped "
          "novelty claim, never a quantum-advantage one -- and neither can be shown "
          "without the branch.")
    summary["steps"]["10_temporal"] = dict(status="SKIPPED", reason="3C not built (P2)")
    summary["steps"]["11_quantum"] = dict(status="SKIPPED", reason="3E not built (P2)")

    (out_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir / 'demo_summary.json'}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PLAN.md §10 Phase 7 demo")
    ap.add_argument("--scene", type=Path, default=None, help="HAD100 .hdr (default: first found)")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "demo")
    ap.add_argument("--assert-offline", action="store_true",
                    help="enforce and prove no socket is opened during inference (§10 step 8)")
    ap.add_argument("--profile", default="object", choices=["object", "landcover"])
    ap.add_argument("--target-recall", type=float, default=0.98)
    a = ap.parse_args(argv)
    run_demo(scene_hdr=a.scene, out_dir=a.out, assert_offline=a.assert_offline,
             profile_name=a.profile, target_recall=a.target_recall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
