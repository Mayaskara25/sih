#!/usr/bin/env python3
"""PLAN.md Phase 5 Level 1 -- the classical-detector benchmark (§3A.10, §8).

THIS IS A BUG-FINDING INSTRUMENT AS MUCH AS A REPORT. It is the first thing
in this project that runs every detector against every labelled scene, and
the five most recent real defects (D22, D22.1, D22.2, D23, D24) were all
found by integration while a 279-test suite stayed green. A detector that
crashes, returns NaN, or produces something implausible is therefore a
FINDING: the row is kept and marked, never dropped and never swallowed by a
bare `except`. `--strict` turns any such row into a non-zero exit.

POOLING IS NOT A FREE CHOICE (§3A.10). ABU's anomaly density spans 0.084% to
2.72%, a 32x range (D13.2), so a pixel-weighted pool is effectively a report
on `abu-urban-4` and `abu-beach-2` and says almost nothing about
`abu-beach-1`. Scene-macro-average is PRIMARY; pixel-micro-average is
secondary and always carries the label `micro`. An unlabelled "pooled"
figure is banned, so this script never emits one.

STANDARDIZATION IS RECORDED, NOT ASSUMED (D22). `run_pipeline` standardizes
before the detector; calling a detector directly on native radiance does
not, and that difference crashed `global_rx` on 3 of 13 ABU scenes. Two
detectors compared under different input scalings are not comparable, so the
choice is a column in the output, not an implementation detail.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.crd import crd                                          # noqa: E402
from anomaly.fusion import fuse_scores                               # noqa: E402
from anomaly.kernel_rx import kernel_rx                              # noqa: E402
from anomaly.local_rx import local_rx                                # noqa: E402
from anomaly.rx import global_rx                                     # noqa: E402
from anomaly.scoring import (                                        # noqa: E402
    ace_score,
    calibrate_threshold_for_recall,
    estimate_target_signature,
    spatial_context_score,
)
from preprocessing.normalize import standardize                      # noqa: E402

OUT_DIR = ROOT / "experiments" / "rx_vs_ae"
AUDIT_DIR = ROOT / "experiments" / "cascade_recall_audit"
ABU = ROOT / "data" / "benchmark" / "abu"
HYDICE = ROOT / "data" / "benchmark" / "hydice_urban_anomaly" / "HYDICE-urban.mat"
HAD100 = ROOT / "data" / "benchmark" / "had100" / "HAD100"

# §3A.9's weights are the DEFAULTS, not the tuned ones. D25 measured that the
# tuned weights beat rx alone by +0.0018 macro AUC -- noise -- and that the
# optimizer assigns ACE exactly 0.0. Fusion is reported here as a component,
# not as a winner; see D25 for what may and may not be claimed.
FUSION_WEIGHTS = None   # None -> fuse_scores' own defaults, renormalized (D20)


@dataclass
class Row:
    scene: str
    dataset: str
    detector: str
    standardized: bool
    n_px: int
    n_anom: int
    anom_frac: float
    roc_auc: float | None = None
    pr_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    runtime_s: float | None = None
    peak_rss_mb: float | None = None
    components: str = ""
    status: str = "ok"
    note: str = ""


# ---------------------------------------------------------------- loaders --

def _abu_scenes() -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    out = []
    for p in sorted(ABU.glob("abu-*.mat")):
        m = sio.loadmat(p)
        out.append((p.stem, "abu", m["data"].astype(np.float32), m["map"].astype(bool)))
    return out


def _hydice_scenes():
    if not HYDICE.exists():
        return []
    m = sio.loadmat(HYDICE)
    return [("hydice_urban_anomaly", "hydice",
             m["data"].astype(np.float32), m["map"].astype(bool))]


def _had100_scenes(limit: int | None):
    """HAD100 target patches. ENVI .dat/.hdr cubes with .mat ground truth.

    O8/D13.4: HAD100 is the ONLY benchmark shipping real wavelengths, so it
    is the only one on which the 4-component fusion the plan specifies can
    even run, and the only legal scoring set for learned models. Classical
    detectors run here too, on native bands.
    """
    import spectral
    out = []
    for sub, gtsub in (("aviris_ng_target", "aviris_ng_gt"), ("aviris_target", "aviris_gt")):
        ddir, gdir = HAD100 / "data" / sub, HAD100 / "gt" / gtsub
        if not ddir.exists():
            continue
        for hdr in sorted(ddir.glob("*.hdr")):
            gt_path = gdir / f"{hdr.stem}.mat"
            if not gt_path.exists():
                continue
            cube = np.asarray(spectral.open_image(str(hdr)).load(), dtype=np.float32)
            gm = sio.loadmat(gt_path)
            key = next(k for k in gm if not k.startswith("__"))
            out.append((hdr.stem, "had100", cube, gm[key].astype(bool)))
            if limit and len(out) >= limit:
                return out
    return out


# -------------------------------------------------------------- detectors --

def _fused(cube):
    base = global_rx(cube)
    sig = estimate_target_signature(cube, base, top_frac=0.001)
    comps = {"rx": local_rx(cube, **_local_rx_params(cube)),
             "ace": ace_score(cube, sig),
             "spatial": spatial_context_score(base, k=7)}
    r = fuse_scores(comps, FUSION_WEIGHTS)
    return r.score, "+".join(sorted(r.components))


def _local_rx_params(cube) -> dict:
    """§3A.2 sets local_rx's window PER DATASET and forbids hardcoding it:
    HAD100's unpacked 64x64 patches want `outer=15, inner=3, n_components=12`,
    and the plan is explicit that this must NOT be carried to larger scenes --
    "at 120x120 it needlessly starves the annulus."

    Measured, because the first run of this harness did carry it over: with
    the 64x64 params on ABU's 100x100/150x150 scenes, local_rx scored a
    scene-macro ROC-AUC of 0.9269 and came LAST of five detectors, below
    global_rx, which is implausible for a detector that beat global RX by
    0.14 AUC on abu-airport-1 in isolation. The plan's warning is correct and
    the cost of ignoring it is a benchmark that ranks the detectors wrongly.
    """
    h, w = cube.shape[:2]
    if min(h, w) <= 64:
        return dict(outer=15, inner=3, n_components=12)
    return dict(outer=21, inner=5, n_components=20)


DETECTORS = {
    "global_rx": lambda c: (global_rx(c), ""),
    "local_rx": lambda c: (local_rx(c, **_local_rx_params(c)), ""),
    "kernel_rx": lambda c: (kernel_rx(c, n_background=1000, seed=0), ""),
    "crd": lambda c: (crd(c), ""),
    "fused": _fused,
}

# §11.1 marks these P3 "do not start". §3A.10's table lists them, so they are
# emitted as DEFERRED rows rather than silently dropped -- a missing row reads
# as an oversight, a marked one reads as a decision (§14 keeps deferrals
# visible for the same reason).
DEFERRED = {
    "kpca_autoencoder": "P3 (§11.1) -- not started; §3A.6",
    "deep_detector": "P3 (§11.1) -- not started; §3A.7, scoped down per D8",
    "unet_implanted_lib": "blocked: SPECTRA_POOLS['lib'] unfetchable, both sources human-gated (D21)",
    "unet_lodo_abu": "suspended pending O9 -- ABU ships no wavelengths (D19)",
    "unet_lodo_hyd": "suspended pending O9 -- HYDICE ships no wavelengths (D19)",
    "unet_all_real": "suspended pending O9 -- training input needs harmonizable real spectra (D19)",
}


def _metrics(score, gt, row: Row) -> None:
    valid = ~np.isnan(score)
    if not valid.any():
        row.status = "all_nan"
        row.note = "detector returned NaN everywhere -- FINDING, not a skip"
        return
    y, s = gt[valid], score[valid]
    if y.sum() == 0 or y.all():
        row.status = "degenerate_labels"
        row.note = f"{int(y.sum())} positives of {y.size} valid px"
        return
    row.roc_auc = float(roc_auc_score(y, s))
    row.pr_auc = float(average_precision_score(y, s))
    # Operating point: the recall-first cascade threshold (§4.2), because
    # precision/recall/F1 are meaningless without stating one.
    thr, _fp = calibrate_threshold_for_recall(s, y, target_recall=0.98)
    pred = s >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    row.precision = tp / (tp + fp) if tp + fp else 0.0
    row.recall = tp / (tp + fn) if tp + fn else 0.0
    row.f1 = (2 * row.precision * row.recall / (row.precision + row.recall)
              if row.precision + row.recall else 0.0)
    if np.isnan(score).any():
        row.note = f"{int(np.isnan(score).sum())} NaN px excluded"


def run_one(name, dataset, cube, gt, det_name, standardize_first) -> Row:
    row = Row(scene=name, dataset=dataset, detector=det_name,
              standardized=standardize_first, n_px=int(gt.size),
              n_anom=int(gt.sum()), anom_frac=float(gt.mean()))
    x = standardize(cube) if standardize_first else cube
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        score, comps = DETECTORS[det_name](x)
        row.components = comps
    except Exception as exc:                                    # noqa: BLE001
        row.status = f"FAILED:{type(exc).__name__}"
        row.note = str(exc)[:200]
        row.runtime_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        row.peak_rss_mb = peak / 1e6
        return row
    row.runtime_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    row.peak_rss_mb = peak / 1e6
    _metrics(score, gt, row)
    if row.status == "ok":
        _audit_misses(name, dataset, det_name, score, gt)
    return row


def _audit_misses(scene, dataset, det, score, gt) -> None:
    """§8 Level 1: 'every real labelled target the stage-1 detector fails to
    flag is logged individually with its scene, pixel extent, mean score, and
    the margin by which it missed the threshold.' A miss that is invisible
    cannot be argued about."""
    from scipy import ndimage
    valid = ~np.isnan(score)
    y, s = gt[valid], score[valid]
    if y.sum() == 0:
        return
    thr, fp_rate = calibrate_threshold_for_recall(s, y, target_recall=0.98)
    flagged = np.zeros_like(gt, dtype=bool)
    flagged[valid] = s >= thr
    labels, n = ndimage.label(gt)
    misses = []
    for i in range(1, n + 1):
        comp = labels == i
        if flagged[comp].any():
            continue                       # component partially caught
        cs = score[comp]
        rr, cc = np.where(comp)
        misses.append(dict(
            component=i, n_px=int(comp.sum()),
            extent=[int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1],
            mean_score=float(np.nanmean(cs)), max_score=float(np.nanmax(cs)),
            threshold=float(thr), margin=float(thr - np.nanmax(cs)),
        ))
    if not misses:
        return
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / f"{dataset}__{scene}__{det}.json").write_text(json.dumps(dict(
        scene=scene, dataset=dataset, detector=det, target_recall=0.98,
        threshold=float(thr), induced_fp_rate=float(fp_rate),
        n_components_total=int(n), n_components_missed=len(misses),
        missed=misses), indent=2))


# ----------------------------------------------------------------- pooling --

def pool(df: pd.DataFrame) -> pd.DataFrame:
    """PRIMARY scene-macro (each scene equal weight) and SECONDARY pixel-micro
    (anomaly-pixel weighted), both explicitly labelled -- §3A.10."""
    ok = df[df.status == "ok"]
    out = []
    for (ds, det), g in ok.groupby(["dataset", "detector"]):
        w = g.n_anom.to_numpy(dtype=float)
        rec = dict(dataset=ds, detector=det, n_scenes=len(g),
                   n_failed=int((df[(df.dataset == ds) & (df.detector == det)].status != "ok").sum()))
        for m in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
            v = g[m].to_numpy(dtype=float)
            rec[f"{m}_MACRO"] = float(np.mean(v))
            rec[f"{m}_micro"] = float(np.average(v, weights=w)) if w.sum() else float("nan")
        rec["runtime_s_MACRO"] = float(g.runtime_s.mean())
        rec["peak_rss_mb_MACRO"] = float(g.peak_rss_mb.mean())
        out.append(rec)
    return pd.DataFrame(out).sort_values(["dataset", "roc_auc_MACRO"], ascending=[True, False])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="abu,hydice",
                    help="comma list of abu,hydice,had100")
    ap.add_argument("--had100-limit", type=int, default=20)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--no-standardize", action="store_true",
                    help="call detectors on NATIVE scale (D22: global_rx used to crash)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any row failed")
    args = ap.parse_args(argv)

    want = set(args.datasets.split(","))
    scenes = []
    if "abu" in want:
        scenes += _abu_scenes()
    if "hydice" in want:
        scenes += _hydice_scenes()
    if "had100" in want:
        scenes += _had100_scenes(args.had100_limit)
    dets = [d for d in args.detectors.split(",") if d in DETECTORS]
    std = not args.no_standardize
    print(f"{len(scenes)} scenes x {len(dets)} detectors, standardized={std}\n", flush=True)

    rows = []
    for name, ds, cube, gt in scenes:
        for d in dets:
            r = run_one(name, ds, cube, gt, d, std)
            rows.append(r)
            flag = "" if r.status == "ok" else f"  <-- {r.status}"
            auc = f"{r.roc_auc:.4f}" if r.roc_auc is not None else "  --  "
            print(f"  {ds:8s} {name:28s} {d:10s} auc {auc} {r.runtime_s:6.2f}s{flag}", flush=True)

    df = pd.DataFrame([vars(r) for r in rows])
    for k, why in DEFERRED.items():
        df.loc[len(df)] = dict(scene="(all)", dataset="-", detector=k, standardized=std,
                               n_px=0, n_anom=0, anom_frac=0.0, status="DEFERRED", note=why,
                               components="")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    pooled = pool(df)
    pooled.to_csv(OUT_DIR / "results_pooled.csv", index=False)

    print("\n=== SCENE-MACRO (PRIMARY) vs pixel-micro (secondary) ===")
    print(pooled[["dataset", "detector", "n_scenes", "n_failed",
                  "roc_auc_MACRO", "roc_auc_micro", "pr_auc_MACRO", "pr_auc_micro"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    failed = df[df.status.str.startswith("FAILED") | (df.status == "all_nan")]
    if len(failed):
        print(f"\n*** {len(failed)} FAILED ROWS -- these are findings, not skips ***")
        for _, r in failed.iterrows():
            print(f"    {r.dataset}/{r.scene} {r.detector}: {r.status} {r.note[:90]}")
    print(f"\nwrote {OUT_DIR/'results.csv'} and results_pooled.csv")
    if AUDIT_DIR.exists():
        print(f"cascade recall audit: {len(list(AUDIT_DIR.glob('*.json')))} scene/detector files")
    return 1 if (args.strict and len(failed)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
