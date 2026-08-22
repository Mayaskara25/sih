#!/usr/bin/env python3
"""Verify the EnMAP L2A products on local disk against what PLAN.md O11/D16/D32
depend on. Companion to verify_had100.py / verify_benchmarks.py. Same contract:
reads only, parses every file, asserts what the plan depends on, exits non-zero
on drift, prints a clear per-scene report.

Written because PLAN.md O11 called Phase 5 Level 2 "blocked" by a DLR
entitlement denial while eight complete EnMAP L2A products (~3.6 GB, 40 files)
sat in data/raw/enmap/ the whole time -- D16 (2026-08-21) had already opened
one of them for metadata alone. This script is the first pass that opens
every one of the eight, reads the actual pixel cubes (not just metadata), and
records per-scene facts rather than assuming they generalize from one file.

Checks, per scene:
  - SPECTRAL_IMAGE_COG.TIF dimensions, band count, dtype, CRS, geotransform,
    nodata value, WGS84 bounds
  - METADATA.XML(.XML) sidecar: acquisition start time, cloud/haze/cirrus
    cover, orthorectification residual/RMSE, the wavelength array (count,
    min, max, strict monotonicity -- FAILS per D11.4 on a NaN-poisoned or
    non-monotonic axis), and GainOfBand/OffsetOfBand (the reflectance scale
    factor) as actually found, uniformity checked across all 224 bands
  - fraction of valid (non-nodata) pixels, from band 1 at full resolution
  - EVERY band's nodata fraction (decimated 4x read, all 224 bands, not a
    3-band spot check -- a 3-band spot check was tried first and missed a
    real, systematic 5-band block entirely by chance; see below) -- flags
    any band that is nodata for effectively 100% of the scene, which is a
    DIFFERENT failure from the border nodata every band has: it silently
    zeroes the pipeline's valid-pixel set globally (D32) unless the loader
    marks it bad_bands, because standardize()'s per-band mean is NaN for
    that band at every pixel, and global_rx's any-band-NaN validity check
    then excludes every pixel in the scene, not just that band
  - harmonize() coverage against the canonical grid (D9/D16), reusing
    preprocessing.harmonize directly rather than reimplementing it

Cross-scene invariants asserted: band count, dtype, nodata value, GSD, and
the wavelength grid (byte-identical, per D16) are constant across all 8
scenes. CRS is NOT asserted uniform (it varies: EPSG:32642/32643/32644,
UTM zone follows scene longitude) -- reported, not enforced. The fully-nodata
band set is reported, not asserted, for the same reason.

Reads only. Writes docs/enmap_verified.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
ENMAP_DIR = ROOT / "data" / "raw" / "enmap"
SPECTRAL_SUFFIX = "-SPECTRAL_IMAGE_COG.TIF"

sys.path.insert(0, str(ROOT))
from preprocessing.harmonize import CANONICAL_WL, coverage_ok, water_mask  # noqa: E402


def find_metadata(spectral_path: Path) -> Path:
    """Both filename forms are on disk (D-note): try -METADATA.XML.XML first,
    then -METADATA.XML. Raise if neither exists."""
    stem = spectral_path.name[: -len(SPECTRAL_SUFFIX)]
    for suffix in ("-METADATA.XML.XML", "-METADATA.XML"):
        cand = spectral_path.with_name(stem + suffix)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{spectral_path.name}: no METADATA.XML(.XML) sidecar found")


def parse_metadata(path: Path) -> dict:
    root = ET.parse(path).getroot()

    entries = []
    for band in root.iter("bandID"):
        num = band.get("number")
        wl = band.findtext("wavelengthCenterOfBand")
        fwhm = band.findtext("FWHMOfBand")
        gain = band.findtext("GainOfBand")
        offset = band.findtext("OffsetOfBand")
        if num is None or wl is None:
            continue
        entries.append(dict(
            number=int(num), wl=float(wl),
            fwhm=float(fwhm) if fwhm is not None else None,
            gain=float(gain) if gain is not None else None,
            offset=float(offset) if offset is not None else None,
        ))
    entries.sort(key=lambda e: e["number"])
    wl = np.array([e["wl"] for e in entries], dtype=np.float64)
    gains = sorted({e["gain"] for e in entries if e["gain"] is not None})
    offsets = sorted({e["offset"] for e in entries if e["offset"] is not None})

    return dict(
        n_bands_in_metadata=len(entries),
        wavelengths=wl,
        wl_min=float(wl.min()) if wl.size else None,
        wl_max=float(wl.max()) if wl.size else None,
        strictly_ascending=bool(np.all(np.diff(wl) > 0)) if wl.size > 1 else None,
        all_finite=bool(np.all(np.isfinite(wl))) if wl.size else None,
        median_step_nm=float(np.median(np.diff(wl))) if wl.size > 1 else None,
        gains_distinct=gains,
        offsets_distinct=offsets,
        gain_uniform=(len(gains) == 1),
        offset_uniform=(len(offsets) == 1),
        background_value=_findtext_num(root, ".//backgroundValue"),
        start_time=root.findtext(".//startTime"),
        stop_time=root.findtext(".//stopTime"),
        cloud_cover_pct=_findtext_num(root, ".//cloudCover"),
        haze_cover_pct=_findtext_num(root, ".//hazeCover"),
        cirrus_cover_pct=_findtext_num(root, ".//cirrusCover"),
        snow_cover_pct=_findtext_num(root, ".//snowCover"),
        # No unit attribute present on these elements in the schema as shipped
        # (checked directly) -- reported as raw numbers, unit NOT asserted.
        ortho_rmse=_findtext_num(root, ".//orthoRMSE"),
        ortho_rmse_x=_findtext_num(root, ".//orthoRMSE_x"),
        ortho_rmse_y=_findtext_num(root, ".//orthoRMSE_y"),
        swir_selected=root.findtext(".//SWIRAOrSWIRBSelected"),
        n_vnir_bands=_findtext_num(root, ".//numberOfVNIRBands"),
        n_swir_bands=_findtext_num(root, ".//numberOfSWIRBands"),
    )


def _findtext_num(root, path):
    t = root.findtext(path)
    if t is None:
        return None
    try:
        return float(t) if "." in t else int(t)
    except ValueError:
        return t


def verify_scene(spectral_path: Path) -> dict:
    scene_id = spectral_path.name[: -len(SPECTRAL_SUFFIX)]
    metadata_path = find_metadata(spectral_path)
    md = parse_metadata(metadata_path)

    with rasterio.open(spectral_path) as ds:
        h, w, b = ds.height, ds.width, ds.count
        dtype = ds.dtypes[0]
        crs = str(ds.crs)
        transform = list(ds.transform)[:6]
        nodata = ds.nodata
        bounds = ds.bounds

        b1 = ds.read(1)
        m1 = (b1 == nodata) if nodata is not None else np.zeros_like(b1, dtype=bool)
        valid_fraction = float((~m1).mean())

        # Per-band nodata fraction, EVERY band -- not a 3-band spot check.
        # A 3-band check (band 1 / mid / last) was tried first and looked
        # clean; it silently missed a real, systematic 5-band block near the
        # middle of the spectrum (see below) because none of the 3 sampled
        # indices happened to land inside it. Read decimated (every 4th row
        # and column, via GDAL overview-aware resampling) -- ~1/16 the bytes
        # of a full-resolution read, fast enough to check all `b` bands, and
        # a band that is ACTUALLY 100% nodata is still ~100% nodata at 4x
        # decimation (verified: matches the full-resolution count exactly
        # for this failure mode, which is spatially total, not sparse).
        out_h, out_w = max(h // 4, 1), max(w // 4, 1)
        decimated = ds.read(out_shape=(b, out_h, out_w),
                            resampling=rasterio.enums.Resampling.nearest)
        per_band_nodata_frac = ((decimated == nodata).mean(axis=(1, 2))
                                if nodata is not None else np.zeros(b))
        fully_nodata_bands = [i + 1 for i, f in enumerate(per_band_nodata_frac) if f > 0.999]

    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = tf.transform(bounds.left, bounds.bottom)
    lon1, lat1 = tf.transform(bounds.right, bounds.top)
    wgs84_bounds = dict(
        min_lon=min(lon0, lon1), max_lon=max(lon0, lon1),
        min_lat=min(lat0, lat1), max_lat=max(lat0, lat1),
    )

    wl = md.pop("wavelengths")
    wl32 = wl.astype(np.float32)

    # coverage_ok reused directly from preprocessing.harmonize -- not
    # reimplemented. Reports D16's finding independently, per-scene.
    retained_wl = CANONICAL_WL[~water_mask(CANONICAL_WL)]
    covered = coverage_ok(wl32, retained_wl) if wl32.size else False
    n_uncovered = 0
    uncovered_list: list[float] = []
    if wl32.size:
        wl_sorted = np.sort(wl32.astype(np.float64))
        diffs = np.diff(wl_sorted)
        tol = float(np.median(diffs)) if diffs.size else np.inf
        idx = np.searchsorted(wl_sorted, retained_wl.astype(np.float64))
        idx = np.clip(idx, 1, len(wl_sorted) - 1)
        left = wl_sorted[idx - 1]
        right = wl_sorted[idx]
        nearest = np.minimum(np.abs(retained_wl.astype(np.float64) - left),
                              np.abs(retained_wl.astype(np.float64) - right))
        bad = nearest > tol
        n_uncovered = int(bad.sum())
        uncovered_list = [float(x) for x in retained_wl[bad]]

    wl_sha256 = hashlib.sha256(wl32.tobytes()).hexdigest() if wl32.size else None

    return dict(
        scene_id=scene_id,
        spectral_file=spectral_path.name,
        metadata_file=metadata_path.name,
        dims_hw=[h, w],
        band_count=b,
        dtype=dtype,
        crs=crs,
        transform=transform,
        gsd_m=abs(transform[0]),
        nodata=nodata,
        wgs84_bounds=wgs84_bounds,
        valid_fraction=valid_fraction,
        fully_nodata_bands_decimated=fully_nodata_bands,
        n_fully_nodata_bands=len(fully_nodata_bands),
        wavelength_count=int(wl32.size),
        wavelength_min_nm=md["wl_min"],
        wavelength_max_nm=md["wl_max"],
        wavelength_strictly_ascending=md["strictly_ascending"],
        wavelength_all_finite=md["all_finite"],
        wavelength_median_step_nm=md["median_step_nm"],
        wavelength_sha256=wl_sha256,
        gain_of_band_distinct=md["gains_distinct"],
        gain_of_band_uniform=md["gain_uniform"],
        offset_of_band_distinct=md["offsets_distinct"],
        offset_of_band_uniform=md["offset_uniform"],
        acquisition_start=md["start_time"],
        acquisition_stop=md["stop_time"],
        cloud_cover_pct=md["cloud_cover_pct"],
        haze_cover_pct=md["haze_cover_pct"],
        cirrus_cover_pct=md["cirrus_cover_pct"],
        snow_cover_pct=md["snow_cover_pct"],
        ortho_rmse=md["ortho_rmse"],
        ortho_rmse_x=md["ortho_rmse_x"],
        ortho_rmse_y=md["ortho_rmse_y"],
        ortho_rmse_unit_in_xml="NOT PRESENT -- no unit attribute on this element; "
                                "reported as a raw number, unit unverified from the file",
        swir_selected=md["swir_selected"],
        metadata_background_value=md["background_value"],
        harmonize_coverage_ok=covered,
        harmonize_n_canonical_bands_uncovered=n_uncovered,
        harmonize_uncovered_wavelengths_nm=uncovered_list,
    )


def main() -> int:
    if not ENMAP_DIR.exists():
        print(f"MISSING: {ENMAP_DIR}", file=sys.stderr)
        return 2

    spectral_files = sorted(ENMAP_DIR.glob(f"*{SPECTRAL_SUFFIX}"))
    if not spectral_files:
        print(f"No *{SPECTRAL_SUFFIX} files in {ENMAP_DIR}", file=sys.stderr)
        return 2

    fail: list[str] = []
    scenes: list[dict] = []

    print(f"=== EnMAP L2A -- {len(spectral_files)} products in {ENMAP_DIR} ===\n")
    for f in spectral_files:
        try:
            rec = verify_scene(f)
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole run
            fail.append(f"{f.name}: FAILED TO VERIFY -- {exc}")
            print(f"  {f.name}: FAILED -- {exc}")
            continue
        scenes.append(rec)
        print(f"  {rec['scene_id'][:38]:38} {rec['dims_hw'][0]}x{rec['dims_hw'][1]}x"
              f"{rec['band_count']} {rec['dtype']:6} {rec['crs']:12} "
              f"valid={rec['valid_fraction']:.3f} cloud={rec['cloud_cover_pct']!s:>4} "
              f"wl=[{rec['wavelength_min_nm']:.2f},{rec['wavelength_max_nm']:.2f}] "
              f"ascending={rec['wavelength_strictly_ascending']} "
              f"coverage_ok={rec['harmonize_coverage_ok']} "
              f"fully_nodata_bands={rec['fully_nodata_bands_decimated']}")

        if rec["band_count"] != 224:
            fail.append(f"{rec['scene_id']}: band_count={rec['band_count']}, expected 224")
        if rec["dtype"] != "int16":
            fail.append(f"{rec['scene_id']}: dtype={rec['dtype']}, expected int16")
        if rec["nodata"] != -32768.0:
            fail.append(f"{rec['scene_id']}: nodata={rec['nodata']}, expected -32768.0")
        if abs(rec["gsd_m"] - 30.0) > 1e-6:
            fail.append(f"{rec['scene_id']}: gsd_m={rec['gsd_m']}, expected 30.0")
        if not rec["wavelength_strictly_ascending"]:
            fail.append(f"{rec['scene_id']}: wavelength axis NOT strictly ascending -- "
                        "D11.4 poison, must not reach harmonize()")
        if not rec["wavelength_all_finite"]:
            fail.append(f"{rec['scene_id']}: wavelength axis contains non-finite values")
        if rec["wavelength_count"] != 224:
            fail.append(f"{rec['scene_id']}: wavelength_count={rec['wavelength_count']}, "
                        "expected 224 (must equal band_count)")
        if not rec["gain_of_band_uniform"]:
            fail.append(f"{rec['scene_id']}: GainOfBand is NOT uniform across bands: "
                        f"{rec['gain_of_band_distinct']}")
        if not rec["fully_nodata_bands_decimated"]:
            pass  # normal: most scenes may have zero fully-nodata bands

    print()

    # ---------- cross-scene invariants ----------
    out: dict = {"scenes": scenes, "n_scenes": len(scenes)}
    if scenes:
        band_counts = {s["band_count"] for s in scenes}
        dtypes = {s["dtype"] for s in scenes}
        nodatas = {s["nodata"] for s in scenes}
        gsds = {round(s["gsd_m"], 6) for s in scenes}
        crss = sorted({s["crs"] for s in scenes})
        wl_shas = {s["wavelength_sha256"] for s in scenes}
        gains = sorted({g for s in scenes for g in s["gain_of_band_distinct"]})
        coverage_oks = {s["harmonize_coverage_ok"] for s in scenes}
        swir = sorted({s["swir_selected"] for s in scenes})
        fully_nodata_sets = {tuple(s["fully_nodata_bands_decimated"]) for s in scenes}

        out["cross_scene"] = dict(
            band_count_uniform=(len(band_counts) == 1),
            band_counts=sorted(band_counts),
            dtype_uniform=(len(dtypes) == 1),
            dtypes=sorted(dtypes),
            nodata_uniform=(len(nodatas) == 1),
            nodatas=sorted(nodatas),
            gsd_uniform=(len(gsds) == 1),
            gsds=sorted(gsds),
            crs_uniform=(len(crss) == 1),
            crs_values=crss,
            wavelength_grid_byte_identical_across_scenes=(len(wl_shas) == 1),
            wavelength_grid_sha256_values=sorted(wl_shas),
            gain_of_band_values_seen=gains,
            harmonize_coverage_ok_uniform=(len(coverage_oks) == 1),
            harmonize_coverage_ok_values=sorted(str(c) for c in coverage_oks),
            swir_selected_values=swir,
            mean_valid_fraction=float(np.mean([s["valid_fraction"] for s in scenes])),
            min_valid_fraction=float(np.min([s["valid_fraction"] for s in scenes])),
            max_valid_fraction=float(np.max([s["valid_fraction"] for s in scenes])),
            fully_nodata_bands_uniform_across_scenes=(len(fully_nodata_sets) == 1),
            fully_nodata_bands_sets_seen=[list(s) for s in fully_nodata_sets],
        )

        print("=== cross-scene invariants ===")
        print(f"  band_count uniform: {out['cross_scene']['band_count_uniform']} "
              f"{out['cross_scene']['band_counts']}")
        print(f"  dtype uniform     : {out['cross_scene']['dtype_uniform']} "
              f"{out['cross_scene']['dtypes']}")
        print(f"  nodata uniform    : {out['cross_scene']['nodata_uniform']} "
              f"{out['cross_scene']['nodatas']}")
        print(f"  gsd_m uniform     : {out['cross_scene']['gsd_uniform']} "
              f"{out['cross_scene']['gsds']}")
        print(f"  CRS  (NOT asserted uniform -- follows UTM zone by scene longitude): "
              f"{out['cross_scene']['crs_values']}")
        print(f"  wavelength grid byte-identical across all scenes (sha256): "
              f"{out['cross_scene']['wavelength_grid_byte_identical_across_scenes']}")
        print(f"  GainOfBand values seen (all bands, all scenes): {gains}")
        print(f"  harmonize.coverage_ok uniform: "
              f"{out['cross_scene']['harmonize_coverage_ok_uniform']} "
              f"= {out['cross_scene']['harmonize_coverage_ok_values']}")
        print(f"  SWIR detector selected: {swir}")
        print(f"  valid_fraction: mean={out['cross_scene']['mean_valid_fraction']:.3f} "
              f"min={out['cross_scene']['min_valid_fraction']:.3f} "
              f"max={out['cross_scene']['max_valid_fraction']:.3f}")
        print(f"  fully-nodata bands (>99.9% nodata, decimated scan) uniform across scenes: "
              f"{out['cross_scene']['fully_nodata_bands_uniform_across_scenes']} "
              f"= {out['cross_scene']['fully_nodata_bands_sets_seen']}")

        if len(band_counts) != 1:
            fail.append(f"band_count NOT uniform across scenes: {band_counts}")
        if len(dtypes) != 1:
            fail.append(f"dtype NOT uniform across scenes: {dtypes}")
        if len(nodatas) != 1:
            fail.append(f"nodata NOT uniform across scenes: {nodatas}")
        if len(gsds) != 1:
            fail.append(f"gsd_m NOT uniform across scenes: {gsds}")
        if len(wl_shas) != 1:
            fail.append(f"wavelength grid differs across scenes (sha256): {wl_shas}")
        if len(coverage_oks) != 1:
            fail.append(f"harmonize.coverage_ok differs across scenes: {coverage_oks}")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "enmap_verified.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {ROOT / 'docs' / 'enmap_verified.json'}")

    if fail:
        print("\nINVARIANTS VIOLATED / VERIFICATION FAILURES:", file=sys.stderr)
        for f_ in fail:
            print(f"  - {f_}", file=sys.stderr)
        return 1
    print("all documented invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
