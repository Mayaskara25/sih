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
Step 11 (classical-vs-quantum) is **not wired** here -- branch 3E is built
(plan.md D27) but its comparison results are owned by the quantum branch;
the demo prints a SKIPPED line naming what is missing rather than producing
a plausible-looking placeholder.

Step 10 (temporal t1-vs-t2) RUNS, but only as a **SYNTHETIC-PAIRS**
construction: there is no real bi-temporal hyperspectral pair in data/
(single-epoch benchmarks; EnMAP download blocked, O11). The demo derives
t2 from the loaded real scene by known-shift misregistration + co-register,
target implantation, and an illumination gain, then runs the §3C signal
stack on it. Per §13 reporting rules every number it prints is labelled
SYNTHETIC-PAIRS and must be quoted as such.

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
    # load_scene, NOT spectral.open_image directly: it delegates the ENVI
    # `map info` parse to GDAL (D14.2, after a hand-rolled parser was found
    # silently dropping real rotation present on every HAD100 header), and it
    # returns a SceneMeta whose transform and crs are the REAL ones. An
    # earlier version of this demo built its own SceneMeta with
    # from_origin(0,0,1,1) while setting georef="real" -- a fabricated affine
    # wearing a "real" label, which is precisely the mislabel D2 exists to
    # prevent and would have made any QGIS check meaningless.
    from preprocessing.raster_loader import load_scene
    cube, meta = load_scene(scene_hdr, source="had100")
    wl = meta.wavelengths
    if wl is not None:
        wl = np.asarray(wl, dtype=np.float64)
        if wl.max() < 100:                       # some ENVI headers use micrometres
            wl = wl * 1000.0
    gt = None
    if gt_path and gt_path.exists():
        gm = sio.loadmat(gt_path)
        gt = gm[next(k for k in gm if not k.startswith("__"))].astype(bool)
    print(f"     {scene_hdr.name}  {cube.shape}  wavelengths={'yes' if wl is not None else 'no'}"
          f"  gt={'yes' if gt is not None else 'no'}")
    print(f"     CRS {meta.crs.to_string() if meta.crs else 'NONE'}   georef={meta.georef}")
    print(f"     transform read from the ENVI header via GDAL (D14.2), not fabricated")
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
            harm, harm_meta = harmonize(cube, meta)
            print(f"     harmonized {cube.shape[-1]} -> {harm.shape[-1]} canonical bands "
                  f"(400-2500 nm @10 nm, water windows dropped -- D9)")
        except Exception as exc:                                  # noqa: BLE001
            print(f"     harmonize FAILED: {type(exc).__name__}: {str(exc)[:110]}")
            print(f"     -> learned segmentation will be skipped; classical path unaffected")
        cloud_note = "unavailable"
        try:
            from preprocessing.cloud_mask import cloud_shadow_mask
            cld = cloud_shadow_mask(cube, meta)
            clear_fraction = float((cld == 0).mean())
            print(f"     cloud mask (§3C.7): applied -- {clear_fraction * 100:.1f}% "
                  f"of pixels classed clear (spectral-threshold path)")
            cloud_note = f"applied ({clear_fraction * 100:.1f}% clear)"
        except ValueError as exc:
            print(f"     cloud mask (§3C.7): not applicable here -- {str(exc)[:90]}")
        summary["steps"]["2_preprocess"] = dict(
            native_bands=int(cube.shape[-1]),
            harmonized_bands=(int(harm.shape[-1]) if harm is not None else None),
            cloud_mask=cloud_note)

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

        # Also write the C2 score-raster pair, so the GeoJSON can be checked
        # AGAINST something in QGIS. Polygons alone on a basemap show only
        # that they landed plausibly; overlaid on the score raster they show
        # whether they landed on the pixels the detector actually fired on,
        # which is the affine-plumbing question §2.10 is really asking.
        from preprocessing.raster_loader import save_score_raster
        raw_p, norm_p = save_score_raster(score, meta, out_dir / f"{meta.scene_id}_anom",
                                          method="fused")
        print(f"     {norm_p.name}  (score raster, same CRS -- load both in QGIS)")
        summary["steps"]["7_geojson"] = dict(path=str(gj), n_features=len(kept),
                                             score_raster=str(norm_p))

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

    # -- 10 ------------------------------------------------------------
    _step(10, "Temporal t1-vs-t2 (SAM + physics fusion) [SYNTHETIC-PAIRS]")
    try:
        from change_detection.physics_fusion import (
            difference_structure, fuse_change_signals)
        from change_detection.spectral_angle import spectral_angle
        from preprocessing.harmonize import reduce_bands
        from preprocessing.registration import coregister_subpixel
        from segmentation.synth import implant_targets

        side = min(96, cube.shape[0], cube.shape[1])
        r0 = (cube.shape[0] - side) // 2
        c0 = (cube.shape[1] - side) // 2
        small = np.ascontiguousarray(cube[r0:r0 + side, c0:c0 + side])
        if small.shape[-1] > 30:
            small, _tf10 = reduce_bands(small, n_components=30)
        t1d = small.astype(np.float32)

        shifted = np.roll(t1d, shift=(2, -1), axis=(0, 1))
        aligned, reg10 = coregister_subpixel(t1d, shifted, meta, meta)

        rng10 = np.random.default_rng(0)
        spectra10 = t1d[rng10.integers(0, side, 8), rng10.integers(0, side, 8)]
        t2d, cmask, _impl = implant_targets(aligned, spectra10,
                                            n_targets=3, seed=0)
        changed10 = cmask.astype(bool)

        sam10 = spectral_angle(t1d, t2d)
        fused10 = fuse_change_signals(
            sam10, difference_structure(t1d, t2d),
            np.zeros(t1d.shape[:2], dtype=np.uint8))
        from anomaly.scoring import rank_normalize as _rn
        nfused = _rn(fused10)
        nsam = _rn(sam10)
        m_chg = float(np.nanmean(nfused[changed10]))
        m_bg = float(np.nanmean(nfused[~changed10]))
        print(f"     pair built from THIS scene: known shift co-registered to "
              f"{reg10['rmse_px']:.2f} px residual;")
        print(f"     3 targets implanted into t2 only -- NO real second epoch "
              f"exists (O11).")
        print(f"     mean fused score  changed px: {m_chg:.4f}   "
              f"background px: {m_bg:.4f}")
        print(f"     every figure on this line is SYNTHETIC-PAIRS and must be "
              f"quoted as such.")
        summary["steps"]["10_temporal"] = dict(
            status="RAN [SYNTHETIC-PAIRS]",
            registration_rmse_px=reg10["rmse_px"],
            fused_mean_changed=m_chg, fused_mean_background=m_bg,
            sam_mean_changed=float(np.nanmean(nsam[changed10])),
            n_implants=3)
    except Exception as exc:                                      # noqa: BLE001
        print(f"     FAILED: {type(exc).__name__}: {str(exc)[:110]}")
        print("     -> reported as failed, not silently skipped.")
        summary["steps"]["10_temporal"] = dict(status="FAILED",
                                               error=f"{type(exc).__name__}: "
                                                     f"{str(exc)[:110]}")

    # -- 11 ------------------------------------------------------------
    _skip(11, "Classical-vs-quantum comparison",
          "branch 3E is built (plan.md D27) but its comparison results are "
          "owned by the quantum branch; §13 rule 4 permits only a scoped "
          "novelty claim, never a quantum-advantage one. Not wired here.")
    summary["steps"]["11_quantum"] = dict(status="SKIPPED",
                                          reason="results owned by quantum branch")

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
