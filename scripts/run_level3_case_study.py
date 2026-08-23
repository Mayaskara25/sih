#!/usr/bin/env python3
"""Phase 5 Level 3 -- Sentinel-2 multitemporal case study (PLAN.md §8 Level 3).

Site: Noida International Airport (Jewar), India. 4 Sentinel-2 L2A products
over one AOI (T43RGM, 300x300 px @ 20 m), all fetched and verified by a prior
agent -- see docs/validation.md and docs/sentinel2_verified.json. This script
does NOT fetch, does NOT re-verify band counts, and does NOT touch
core/cdse_s3.py. It runs the processing pipeline PLAN.md §8 calls for:

    fetch stack [done already] -> cloud mask -> co-register -> TemporalBaseline
    -> SAM + physics fusion -> ROIs -> GeoJSON -> QGIS

using only the existing modules in preprocessing/, change_detection/,
segmentation/ and geospatial/ -- no detection science is reimplemented here.

REPORTING CONSTRAINT (non-negotiable, PLAN.md §8 / Roadmap §1.9 & §9.7):
this script and everything it writes report OBSERVED PHYSICAL CHANGE ONLY --
dates, areas, index values. No causal claim, no attribution, no inference
about intent, no economic/political/developmental commentary.

--- Design decisions made BEFORE running, not after seeing results ---

1. SENSOR CONFOUND (task brief, restated): the three S2B products and the
   one S2A product have band centres differing by up to 16.7 nm (B12).
   PRIMARY_DATES is therefore S2B-only (2020-10-16, 2022-03-30, 2026-06-17).
   The 2024-10-30 S2A date is processed too, but only into a clearly
   SECONDARY / CROSSSENSOR-labelled product tree, never merged into the
   primary interval outputs or the accept-criteria numbers.

2. BOA rescaling: DN -> reflectance via (DN + BOA_ADD_OFFSET) /
   BOA_QUANTIFICATION_VALUE, both read PER-PRODUCT from
   docs/sentinel2_verified.json (NOT hardcoded, despite all four products
   here sharing the same numeric values -- the four processing baselines
   differ and a fifth product could differ too). Negative reflectances are
   NOT clipped (legitimate over water, per SCL class 6 pixels in this AOI).

3. Threshold percentile for the change mask: 95.0, NOT Level 2's 99.0.
   Chosen because the "landcover" profile (configs/target_profile.yaml) has
   no max_area_px cap and a far lower min_solidity (0.05 vs 0.15) than
   "object" -- it is explicitly built for large, sprawling regions, not
   compact objects, so a 99th-percentile "top 1%" threshold works against
   the profile's own design intent. This choice is fixed here, before any
   pipeline run, and is not revisited based on how much area it flags.

4. spectral_index_score() (anomaly/scoring.py) is NOT used for the raw
   index rasters below. Its swir1 lookup target is 1650 nm (Landsat-style)
   with a fixed 15 nm tolerance; Sentinel-2's real B11 centre is ~1610-1614
   nm, 36-40 nm away -- select_band() raises ValueError (confirmed
   empirically while building this script). This is a genuine,
   previously-unexercised limitation of that code path against real
   Sentinel-2 data (it was built for/tested on ABU/HAD100's dense
   hyperspectral sampling), not a workaround chosen to dodge a bad number.
   Index VALUES are instead computed here by reusing the exact formulas in
   anomaly.scoring._INDEX_DEFINITIONS (not reimplemented), with bands
   selected by direct position against the verified, fixed, identical
   6-band order (B02,B03,B04,B8A,B11,B12) documented in
   docs/sentinel2_verified.json for all four products -- safe here
   specifically because that order is verified constant across this
   dataset, unlike the general cross-sensor case D9 guards against.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.scoring import (  # noqa: E402
    _INDEX_DEFINITIONS,
    percentile_normalize,
    threshold_by_percentile,
)
from change_detection.physics_fusion import difference_structure, fuse_change_signals
from change_detection.spectral_angle import spectral_angle
from change_detection.temporal_baseline import TemporalBaseline
from change_detection.temporal_difference import magnitude_difference
from core.contracts import (
    validate_geojson,
    validate_mask,
    validate_roi,
    validate_scene,
    validate_score_raster,
)
from geospatial.geojson import _resolve_timestamp, rois_to_geojson
from geospatial.polygonize import mask_to_rois
from preprocessing.cloud_mask import cloud_shadow_mask
from preprocessing.normalize import drop_bad_bands
from preprocessing.raster_loader import load_scene, load_sentinel2_scl, save_score_raster
from preprocessing.registration import RegistrationFailure, coregister_subpixel
from segmentation.postfilter import filter_rois, morphological_cleanup, resolve_profile

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "sentinel2" / "jewar_airport"
OUT_DIR = ROOT / "experiments" / "phase5_level3"
VERIFIED_JSON = ROOT / "docs" / "sentinel2_verified.json"

THRESHOLD_PCT = 95.0          # see module docstring point 3
PATCH = 7                      # difference_structure window, matches run_change_arms.py default
BAND_ORDER = ("B02", "B03", "B04", "B8A", "B11", "B12")
S2_ROLE_POSITION = {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir1": 4, "swir2": 5}
LANDCOVER_INDICES = ("ndvi", "ndwi", "nbr", "bsi")

# All 4 products, chronological. "primary": True marks the S2B-only series.
PRODUCTS = [
    dict(tag="2020-10-16_S2B", date="2020-10-16", sensor="S2B", primary=True,
         stack=DATA_DIR / "S2B_MSIL2A_20201016T052819_N0500_R105_T43RGM_20230413T162324.SAFE_stack.tif"),
    dict(tag="2022-03-30_S2B", date="2022-03-30", sensor="S2B", primary=True,
         stack=DATA_DIR / "S2B_MSIL2A_20220330T052639_N0510_R105_T43RGM_20240522T115955.SAFE_stack.tif"),
    dict(tag="2024-10-30_S2A_SECONDARY", date="2024-10-30", sensor="S2A", primary=False,
         stack=DATA_DIR / "S2A_MSIL2A_20241030T052941_N0511_R105_T43RGM_20241030T093050.SAFE_stack.tif"),
    dict(tag="2026-06-17_S2B", date="2026-06-17", sensor="S2B", primary=True,
         stack=DATA_DIR / "S2B_MSIL2A_20260617T052649_N0512_R105_T43RGM_20260617T104156.SAFE_stack.tif"),
]

PRIMARY_INTERVALS = [(0, 1), (1, 3)]          # 2020->2022, 2022->2026 (skips secondary index 2)
SECONDARY_INTERVALS = [(1, 2), (2, 3)]        # 2022(S2B)->2024(S2A), 2024(S2A)->2026(S2B): cross-sensor
ALL_CONSECUTIVE = [(0, 1), (1, 2), (2, 3)]    # for registration verification, chronological


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
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


def _load_verified_facts() -> dict:
    return json.loads(VERIFIED_JSON.read_text())


def _boa_params(product_name: str, facts: dict) -> tuple[float, float, str]:
    """Per-product (BOA_ADD_OFFSET, BOA_QUANTIFICATION_VALUE, processing_baseline),
    read from docs/sentinel2_verified.json -- never hardcoded (task rule 1)."""
    for p in facts["products"]:
        if p["product_name"] == product_name:
            offsets = p["boa_add_offset_distinct"]
            if len(offsets) != 1:
                raise ValueError(f"{product_name}: boa_add_offset not uniform: {offsets}")
            return float(offsets[0]), float(p["boa_quantification_value"]), p["processing_baseline"]
    raise ValueError(f"{product_name}: not found in {VERIFIED_JSON}")


def _to_reflectance(cube_dn: np.ndarray, offset: float, quant: float) -> np.ndarray:
    """(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE. NOT clipped -- negative
    reflectance over water is legitimate (task rule 1)."""
    return ((cube_dn.astype(np.float64) + offset) / quant).astype(np.float32)


def _compute_indices(cube_refl: np.ndarray) -> dict[str, np.ndarray]:
    """ndvi/ndwi/nbr/bsi via the EXACT anomaly.scoring._INDEX_DEFINITIONS
    formulas, band-selected by verified fixed position (see module docstring
    point 4) rather than select_band()'s default-tolerance lookup."""
    roles = {name: cube_refl[..., pos].astype(np.float64) for name, pos in S2_ROLE_POSITION.items()}
    valid = ~np.any(np.isnan(cube_refl), axis=-1)
    out = {}
    for name in LANDCOVER_INDICES:
        needed, formula = _INDEX_DEFINITIONS[name]
        bands = {n: roles[n] for n in needed}
        with np.errstate(divide="ignore", invalid="ignore"):
            val = formula(bands).astype(np.float32)
        out[name] = np.where(valid, val, np.nan).astype(np.float32)
    return out


def _write_single_band(path: Path, arr: np.ndarray, meta, *, tags: dict | None = None) -> None:
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype="float32" if arr.dtype != np.uint8 else "uint8",
                   crs=meta.crs, transform=meta.transform,
                   nodata=(np.nan if arr.dtype != np.uint8 else None))
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype(profile["dtype"]), 1)
        if tags:
            ds.update_tags(**tags)


def _summary(arr: np.ndarray) -> dict:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return dict(n_valid=0, mean=None, median=None, p95=None, min=None, max=None)
    return dict(n_valid=int(v.size), mean=float(np.mean(v)), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), min=float(np.min(v)), max=float(np.max(v)))


def load_date(entry: dict, facts: dict, timings: dict) -> dict:
    tag = entry["tag"]
    t0 = time.perf_counter()
    cube_dn, meta = load_scene(entry["stack"], source="sentinel2")
    validate_scene(cube_dn, meta)
    cube_dn, meta = drop_bad_bands(cube_dn, meta)
    scl = load_sentinel2_scl(entry["stack"])
    cloud = cloud_shadow_mask(cube_dn, meta, scl=scl)
    validate_mask(cloud)

    product_name = entry["stack"].name.removesuffix("_stack.tif")
    offset, quant, baseline = _boa_params(product_name, facts)
    cube_refl = _to_reflectance(cube_dn, offset, quant)
    indices = _compute_indices(cube_refl)
    timings[f"load_{tag}"] = time.perf_counter() - t0

    n_neg = {}
    for i, b in enumerate(BAND_ORDER):
        band = cube_refl[..., i]
        finite = band[np.isfinite(band)]
        n_neg[b] = int((finite < 0).sum()) if finite.size else 0

    stats = dict(
        product_name=product_name, scene_id=meta.scene_id, acquired=meta.acquired,
        sensor=entry["sensor"], processing_baseline=baseline,
        boa_add_offset=offset, boa_quantification_value=quant,
        wavelengths_nm=meta.wavelengths.tolist(),
        cloud_fraction=float(cloud.mean()),
        n_negative_reflectance_px_per_band=n_neg,
        reflectance_range_per_band={
            BAND_ORDER[i]: [float(np.nanmin(cube_refl[..., i])), float(np.nanmax(cube_refl[..., i]))]
            for i in range(len(BAND_ORDER))
        },
        index_summary={name: _summary(arr) for name, arr in indices.items()},
    )
    return dict(tag=tag, meta=meta, cube_dn=cube_dn, cube_refl=cube_refl, cloud=cloud,
                indices=indices, stats=stats)


def run_interval(t1: dict, t2: dict, out_dir: Path, *, label: str, timings: dict,
                  reg_report: dict | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta1, meta2 = t1["meta"], t2["meta"]
    refl1, refl2 = t1["cube_refl"], t2["cube_refl"]

    t0 = time.perf_counter()
    sam = spectral_angle(refl1, refl2)
    structure = difference_structure(refl1, refl2, patch=PATCH)
    magdiff = magnitude_difference(refl1, refl2)
    cloud_union = np.maximum(t1["cloud"], t2["cloud"]).astype(np.uint8)
    validate_mask(cloud_union)
    fused = fuse_change_signals(sam, structure, cloud_union)
    timings[f"change_signals_{label}"] = time.perf_counter() - t0

    norm, v_lo, v_hi = percentile_normalize(fused)
    mask = threshold_by_percentile(norm, pct=THRESHOLD_PCT)
    mask = morphological_cleanup(mask)
    validate_mask(mask)

    rois = mask_to_rois(mask, meta2, source_branch="change", target_profile="landcover")
    h, w = mask.shape
    profile = resolve_profile("landcover", scene_px=h * w)
    kept, dropped = filter_rois(rois, profile, audit_dir=out_dir / "dropped_rois_audit")
    clear_map = (1.0 - cloud_union.astype(np.float32))
    for roi in kept:
        r0, c0, r1, c1 = roi.bbox
        m = roi.mask.astype(bool)
        vals = norm[r0:r1, c0:c1][m]
        roi.change_score = float(np.nanmean(vals)) if vals.size else None
        roi.clear_fraction = float(np.nanmean(clear_map[r0:r1, c0:c1][m])) if vals.size else None
        validate_roi(roi)

    base = out_dir / f"{label}_change"
    raw_path, norm_path = save_score_raster(fused, meta2, base, method="sam_physics_fusion")
    validate_score_raster(norm_path)
    for p in (raw_path, norm_path):
        with rasterio.open(p, "r+") as ds:
            ds.update_tags(T1_SCENE_ID=meta1.scene_id, T2_SCENE_ID=meta2.scene_id,
                            REG_RMSE_PX=(reg_report.get("rmse_px", "REGISTRATION_FAILED")
                                         if reg_report else "NOT_RUN"))

    mask_path = out_dir / f"{label}_change_mask.tif"
    _write_single_band(mask_path, mask, meta2)

    geojson_path = out_dir / f"{label}_change_rois.geojson"
    rois_to_geojson(kept, meta2, geojson_path)
    validate_geojson(geojson_path)

    expected_ts = _resolve_timestamp(None, meta2)
    written = json.loads(geojson_path.read_text())
    bad_ts = [f["properties"]["timestamp"] for f in written["features"]
              if f["properties"]["timestamp"] != expected_ts]
    if bad_ts:
        raise AssertionError(
            f"{label}: {len(bad_ts)} feature(s) carry a timestamp != meta_t2.acquired "
            f"({expected_ts!r}): {bad_ts[:3]}")
    areas_m2 = [f["properties"]["area"] for f in written["features"]]

    # NDVI delta vs fused score: does physics fusion track raw seasonal/veg
    # magnitude, or add discriminating structure beyond it? Spearman rank
    # correlation over valid, non-cloud pixels (both series NaN-matched).
    ndvi_delta = np.abs(t2["indices"]["ndvi"].astype(np.float64) - t1["indices"]["ndvi"].astype(np.float64))
    valid = np.isfinite(fused) & np.isfinite(ndvi_delta)
    if valid.sum() > 2:
        rho, pval = scipy_stats.spearmanr(fused[valid], ndvi_delta[valid])
    else:
        rho, pval = float("nan"), float("nan")

    return dict(
        label=label, t1_scene=meta1.scene_id, t2_scene=meta2.scene_id,
        t1_date=t1["stats"]["acquired"], t2_date=t2["stats"]["acquired"],
        sam_summary=_summary(sam), magnitude_difference_summary=_summary(magdiff),
        fused_change_summary=_summary(fused),
        percentile_normalize=dict(p_lo=1.0, p_hi=99.9, v_lo=v_lo, v_hi=v_hi),
        threshold_pct=THRESHOLD_PCT,
        n_rois_raw=len(rois), n_rois_kept=len(kept), n_rois_dropped=len(dropped),
        kept_roi_area_m2=areas_m2, kept_roi_area_ha_total=sum(areas_m2) / 10_000.0,
        cloud_union_fraction=float(cloud_union.mean()),
        spearman_fused_vs_abs_ndvi_delta=dict(rho=float(rho), p=float(pval), n=int(valid.sum())),
        registration=reg_report,
        outputs=dict(change_raw=str(raw_path), change_norm=str(norm_path),
                     change_mask=str(mask_path), geojson=str(geojson_path)),
    )


def main() -> int:
    facts = _load_verified_facts()
    timings: dict[str, float] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== loading 4 products ===")
    loaded = [load_date(p, facts, timings) for p in PRODUCTS]
    for d in loaded:
        print(f"  {d['tag']}: acquired={d['stats']['acquired']} "
              f"cloud_frac={d['stats']['cloud_fraction']:.6f} "
              f"baseline={d['stats']['processing_baseline']}")

    dates_dir = OUT_DIR / "dates"
    dates_dir.mkdir(parents=True, exist_ok=True)
    for d in loaded:
        _write_single_band(dates_dir / f"{d['tag']}_cloud_mask.tif", d["cloud"], d["meta"])
        _write_single_band(dates_dir / f"{d['tag']}_ndvi.tif", d["indices"]["ndvi"], d["meta"])
        (dates_dir / f"{d['tag']}_stats.json").write_text(json.dumps(d["stats"], indent=2))

    print("=== co-registration verification (grid already identical -- measuring residual) ===")
    reg_reports: dict[tuple[int, int], dict] = {}
    reg_dir = OUT_DIR / "registration"
    reg_dir.mkdir(parents=True, exist_ok=True)
    pairs_to_check = sorted(set(ALL_CONSECUTIVE) | set(PRIMARY_INTERVALS) | set(SECONDARY_INTERVALS))
    for (i, j) in pairs_to_check:
        t1, t2 = loaded[i], loaded[j]
        t0 = time.perf_counter()
        try:
            _, report = coregister_subpixel(t1["cube_refl"], t2["cube_refl"], t1["meta"], t2["meta"])
            report["status"] = "ok"
        except RegistrationFailure as exc:
            report = {"status": "RegistrationFailure", "message": str(exc)}
        timings[f"coregister_{t1['tag']}_{t2['tag']}"] = time.perf_counter() - t0
        reg_reports[(i, j)] = report
        (reg_dir / f"{t1['tag']}__to__{t2['tag']}.json").write_text(json.dumps(report, indent=2))
        print(f"  {t1['tag']} -> {t2['tag']}: "
              f"rmse_px={report.get('rmse_px')} status={report.get('status')}")

    print("=== primary intervals (S2B-only) ===")
    primary_dir = OUT_DIR / "intervals"
    primary_results = []
    for (i, j) in PRIMARY_INTERVALS:
        t1, t2 = loaded[i], loaded[j]
        label = f"{t1['stats']['acquired'][:10]}_to_{t2['stats']['acquired'][:10]}"
        res = run_interval(t1, t2, primary_dir, label=label, timings=timings,
                            reg_report=reg_reports.get((i, j)))
        primary_results.append(res)
        print(f"  {label}: {res['n_rois_kept']} ROIs kept / {res['n_rois_raw']} raw, "
              f"{res['kept_roi_area_ha_total']:.2f} ha")

    print("=== secondary, cross-sensor intervals (S2A involved -- NOT accept-criteria numbers) ===")
    secondary_dir = OUT_DIR / "secondary_cross_sensor"
    secondary_results = []
    for (i, j) in SECONDARY_INTERVALS:
        t1, t2 = loaded[i], loaded[j]
        label = f"SECONDARY_CROSSSENSOR_{t1['stats']['acquired'][:10]}_{t1['stats']['sensor']}_to_" \
                f"{t2['stats']['acquired'][:10]}_{t2['stats']['sensor']}"
        res = run_interval(t1, t2, secondary_dir, label=label, timings=timings,
                            reg_report=reg_reports.get((i, j)))
        secondary_results.append(res)
        print(f"  {label}: {res['n_rois_kept']} ROIs kept, {res['kept_roi_area_ha_total']:.2f} ha")

    # Sensor-confound quantification in numbers: same-sensor vs cross-sensor SAM.
    same_sensor_sam = [r["sam_summary"]["mean"] for r in primary_results]
    cross_sensor_sam = [r["sam_summary"]["mean"] for r in secondary_results]
    sensor_confound = dict(
        same_sensor_pairs=[r["label"] for r in primary_results],
        same_sensor_sam_mean=same_sensor_sam,
        cross_sensor_pairs=[r["label"] for r in secondary_results],
        cross_sensor_sam_mean=cross_sensor_sam,
        wavelength_gap_nm=dict(
            B11=abs(facts["products"][0]["wavelengths_nm"][4] - facts["products"][2]["wavelengths_nm"][4]),
            B12=abs(facts["products"][0]["wavelengths_nm"][5] - facts["products"][2]["wavelengths_nm"][5]),
        ),
    )
    print(f"  same-sensor mean SAM: {same_sensor_sam}")
    print(f"  cross-sensor mean SAM: {cross_sensor_sam}")

    print("=== TemporalBaseline arm (n=2 baseline: 2020,2022 -> score 2026) ===")
    tb_dir = OUT_DIR / "temporal_baseline"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb = TemporalBaseline(window=2)
    tb.add_epoch(loaded[0]["cube_refl"])
    tb.add_epoch(loaded[1]["cube_refl"])
    med, mad = tb.baseline()
    z = tb.change_score(loaded[3]["cube_refl"])
    z_band_mean = np.nanmean(z, axis=-1).astype(np.float32)
    _write_single_band(tb_dir / "baseline_median_bandmean.tif", np.nanmean(med, axis=-1), loaded[0]["meta"])
    _write_single_band(tb_dir / "baseline_mad_bandmean.tif", np.nanmean(mad, axis=-1), loaded[0]["meta"])
    _write_single_band(tb_dir / "zscore_2026_vs_2020_2022_baseline.tif", z_band_mean, loaded[0]["meta"])
    tb_stats = dict(
        n_baseline_epochs=2, baseline_dates=["2020-10-16", "2022-03-30"], scored_date="2026-06-17",
        note="MAD over 2 points is |a-b|/2 -- a coarse, low-power baseline, not a robust "
             "multi-epoch statistic; reported as such, not as a strong detector.",
        z_bandmean_summary=_summary(z_band_mean),
        frac_px_z_gt_3=float(np.nanmean(z_band_mean > 3.0)),
    )
    (tb_dir / "temporal_baseline_stats.json").write_text(json.dumps(tb_stats, indent=2))
    print(f"  z-bandmean summary: {tb_stats['z_bandmean_summary']}")

    manifest = dict(
        git_sha=_git_sha(), package_versions=_package_versions(),
        run_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        aoi="Noida International Airport (Jewar), India, MGRS T43RGM",
        threshold_pct=THRESHOLD_PCT, patch=PATCH, target_profile="landcover",
        primary_dates=[d["stats"]["acquired"] for d, p in zip(loaded, PRODUCTS) if p["primary"]],
        secondary_dates=[d["stats"]["acquired"] for d, p in zip(loaded, PRODUCTS) if not p["primary"]],
        per_date_stats=[d["stats"] for d in loaded],
        registration=[dict(pair=f"{loaded[i]['tag']}->{loaded[j]['tag']}", **reg_reports[(i, j)])
                      for (i, j) in pairs_to_check],
        primary_intervals=primary_results,
        secondary_cross_sensor_intervals=secondary_results,
        sensor_confound_quantification=sensor_confound,
        temporal_baseline=tb_stats,
        timings_s=timings,
    )
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {OUT_DIR / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
