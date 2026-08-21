"""PLAN.md §2.9 -- the Phase 2 walking-skeleton pipeline.

Stages, each behind a config-selected implementation so a Phase 4 swap is a
config edit and not a rewrite:
    load -> drop_bad_bands -> standardize -> detector -> percentile_normalize
    -> threshold -> morphology -> connected components -> mask_to_rois
    -> rois_to_polygons -> geojson
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import rasterio

from anomaly.crd import crd
from anomaly.fusion import fuse_scores
from anomaly.kernel_rx import kernel_rx
from anomaly.local_rx import local_rx
from anomaly.rx import global_rx
from anomaly.scoring import (
    ace_score,
    estimate_target_signature,
    percentile_normalize,
    spatial_context_score,
    threshold_by_percentile,
)
from core.contracts import (
    validate_geojson,
    validate_mask,
    validate_roi,
    validate_scene,
    validate_score_raster,
)
from geospatial.geojson import rois_to_geojson
from geospatial.polygonize import mask_to_rois
from preprocessing.normalize import drop_bad_bands, l2_normalize, standardize
from preprocessing.raster_loader import load_scene, save_score_raster
from segmentation.postfilter import morphological_cleanup


def _fused(cube: np.ndarray, **params) -> np.ndarray:
    """PLAN.md §3A.9 fusion, wired as an ordinary cube->score detector.

    Bootstraps ACE's target signature from global RX, per §3A.8 (unsupervised
    anomaly detection has no target, so one is estimated from the base
    detector). The `index` component is deliberately ABSENT here: it needs
    per-band wavelengths, and D20 records that most benchmark scenes ship
    none. `fuse_scores` renormalizes the remaining weights and reports which
    components were active -- see D20 for why a missing component must not be
    zero-filled instead.
    """
    base = global_rx(cube)
    sig = estimate_target_signature(cube, base, top_frac=params.get("top_frac", 0.001))
    components = {
        "rx": local_rx(cube, **params.get("local_rx", {})),
        "ace": ace_score(cube, sig),
        "spatial": spatial_context_score(base, k=params.get("spatial_k", 7)),
    }
    return fuse_scores(components).score


# Stage implementations resolved BY NAME so a Phase 4 swap is a config edit
# (PLAN.md §4.1). Every entry here takes a [H,W,B] cube and returns an [H,W]
# score, per the interface pinned in CONTRIBUTING.md.
DETECTORS = {
    "global_rx": global_rx,
    "local_rx": local_rx,
    "kernel_rx": kernel_rx,
    "crd": crd,
    "fused": _fused,
}

# streaming_rx is DELIBERATELY NOT in DETECTORS. It takes a scene PATH, not a
# cube, because its whole purpose is never materializing the cube (§3A.5) --
# so it cannot satisfy the cube->score contract the registry resolves against.
# Named here rather than silently omitted, so that "why is streaming_rx
# missing?" has an answer at the point where someone will ask it. A caller
# wanting it must invoke anomaly.streaming_rx.streaming_rx directly; the
# Phase 5 benchmark harness special-cases it for the same reason.
PATH_DETECTORS = ("streaming_rx",)

NORMALIZERS = {"standardize": standardize, "l2": l2_normalize}


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = ["numpy", "scipy", "rasterio", "geopandas", "shapely", "pyproj", "scikit-image"]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            pass
    return out


def run_pipeline(*, scene: Path, source: str, detector: str, threshold_pct: float,
                  profile: str, out_dir: Path, normalize_method: str = "standardize",
                  detector_params: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def stage(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[name] = time.perf_counter() - t0
        return result

    cube, meta = stage("load", load_scene, scene, source=source)
    cube, meta = stage("drop_bad_bands", drop_bad_bands, cube, meta)
    norm_cube = stage("standardize", NORMALIZERS[normalize_method], cube)
    validate_scene(norm_cube, meta)

    if detector in PATH_DETECTORS:
        raise ValueError(
            f"{detector!r} takes a scene path, not a cube, and is not part of the "
            f"cube->score registry -- see PATH_DETECTORS. Available: {sorted(DETECTORS)}")
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector {detector!r}; available: {sorted(DETECTORS)}")
    # Per-dataset detector params come from config, never hardcoded (§3A.2:
    # HAD100's 64x64 patches need outer=15/inner=3/n_components=12, while
    # raw scenes up to 120x120 should keep the wider default annulus).
    score = stage("detector", DETECTORS[detector], norm_cube, **(detector_params or {}))
    norm_score, _v_lo, _v_hi = stage("percentile_normalize", percentile_normalize, score)
    mask = stage("threshold", threshold_by_percentile, norm_score, pct=threshold_pct)
    mask = stage("morphology", morphological_cleanup, mask)
    validate_mask(mask)

    rois = stage("mask_to_rois", mask_to_rois, mask, meta,
                 source_branch="anomaly", target_profile=profile)
    for roi in rois:
        r0, c0, r1, c1 = roi.bbox
        vals = norm_score[r0:r1, c0:c1][roi.mask.astype(bool)]
        roi.anomaly_score = float(np.nanmean(vals)) if vals.size else None
        validate_roi(roi)

    base = out_dir / f"{meta.scene_id}_anom"
    raw_path, norm_path = stage("save_score_raster", save_score_raster, score, meta, base,
                                 method=detector)
    validate_score_raster(norm_path)

    mask_path = out_dir / f"{meta.scene_id}_mask.tif"
    with rasterio.open(
        mask_path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
        count=1, dtype="uint8", crs=meta.crs, transform=meta.transform,
    ) as ds:
        ds.write(mask, 1)

    geojson_path = out_dir / f"{meta.scene_id}_rois.geojson"
    stage("geojson", rois_to_geojson, rois, meta, geojson_path)
    validate_geojson(geojson_path)

    manifest = dict(
        scene=str(scene), source=source, detector=detector, threshold_pct=threshold_pct,
        profile=profile, normalize_method=normalize_method,
        detector_params=detector_params or {},
        git_sha=_git_sha(), package_versions=_package_versions(),
        timings_s=timings, n_rois=len(rois),
        outputs=dict(anom_raw=str(raw_path), anom_norm=str(norm_path),
                     mask=str(mask_path), geojson=str(geojson_path)),
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--source", required=True)
    p.add_argument("--detector", default="global_rx", choices=sorted(DETECTORS))
    p.add_argument("--detector-params", default=None,
                   help="JSON dict of detector kwargs, e.g. '{\"outer\": 15}'")
    p.add_argument("--threshold-pct", type=float, default=99.0)
    p.add_argument("--profile", default="object", choices=["object", "landcover"])
    p.add_argument("--normalize-method", default="standardize", choices=sorted(NORMALIZERS))
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    manifest = run_pipeline(
        scene=args.scene, source=args.source, detector=args.detector,
        threshold_pct=args.threshold_pct, profile=args.profile,
        out_dir=args.out, normalize_method=args.normalize_method,
        detector_params=json.loads(args.detector_params) if args.detector_params else None,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
