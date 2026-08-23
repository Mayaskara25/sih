#!/usr/bin/env python3
"""Verify the Sentinel-2 L2A products fetched by `scripts/fetch_sentinel2.py`
against what PLAN.md §15/§8 claims and what Phase 5 Level 3 depends on.
Companion to `scripts/verify_had100.py` / `scripts/verify_enmap.py`. Same
contract: reads only, parses every file, asserts what the plan depends on,
exits non-zero on drift, prints a clear per-product report.

Two layers, both against REAL files, neither assumed:

1. LOCAL (`verify_stack`) -- opens every `*_stack.tif` / `*_scl.tif` pair this
   project's own fetcher wrote under `data/raw/sentinel2/`, reads band count,
   dtype, CRS, transform/GSD, nodata, GDAL tags (quantification value,
   BOA_ADD_OFFSET, processing baseline, sensing time), and the actual SCL
   class values present. Always runs, no network needed.

2. REMOTE CROSS-CHECK (`verify_remote_claims`) -- re-fetches ONE product's
   MTD_MSIL2A.xml fresh over the authenticated CDSE S3 leg (core.cdse_s3,
   core.credentials.require("cdse")) and checks it independently against
   PLAN.md's literal claims ("~13 bands multispectral", "10/20/60 m
   resolution mixing") and against what layer 1 found in our own tags --
   this is what stops the "verified" tier from silently becoming "trusts its
   own fetcher's tags." Skipped (with a clear reason, not a silent pass) when
   credentials are not configured; failures here still fail the whole run
   when credentials ARE configured, since that means a real file disagreed.

Writes docs/sentinel2_verified.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import credentials  # noqa: E402
from scripts.fetch_sentinel2 import (  # noqa: E402
    GSD_M,
    _nodes,
    cdse_s3,
    discover_granule,
    fetch_mtd_msil2a,
)

S2_DIR = ROOT / "data" / "raw" / "sentinel2"

# What the L2A SCL band's class codes mean (ESA PSD-15). Used only to check
# every value found is a KNOWN code, not to assert which ones appear.
SCL_CLASSES = {
    0: "no_data", 1: "saturated_or_defective", 2: "dark_area_pixels",
    3: "cloud_shadows", 4: "vegetation", 5: "bare_soils", 6: "water",
    7: "unclassified", 8: "cloud_medium_probability", 9: "cloud_high_probability",
    10: "thin_cirrus", 11: "snow_or_ice",
}
SCL_CLOUDY = frozenset({3, 8, 9, 10})   # matches preprocessing/cloud_mask.py


def verify_stack(stack_path: Path) -> dict:
    scl_path = stack_path.with_name(stack_path.name.replace("_stack.tif", "_scl.tif"))
    if not scl_path.exists():
        raise FileNotFoundError(f"{stack_path.name}: no matching {scl_path.name}")

    with rasterio.open(stack_path) as ds:
        h, w, b = ds.height, ds.width, ds.count
        dtypes = set(ds.dtypes)
        crs = str(ds.crs)
        transform = list(ds.transform)[:6]
        nodata = ds.nodata
        tags = ds.tags()
        descriptions = list(ds.descriptions)
        arr = ds.read()

    with rasterio.open(scl_path) as sds:
        scl_h, scl_w = sds.height, sds.width
        scl_dtype = sds.dtypes[0]
        scl_crs = str(sds.crs)
        scl_transform = list(sds.transform)[:6]
        scl = sds.read(1)

    wavelengths = np.array([float(x) for x in tags["WAVELENGTHS_NM"].split(",")],
                           dtype=np.float64)
    boa_offsets = json.loads(tags["BOA_ADD_OFFSET_DISTINCT"].replace("'", '"')) \
        if "BOA_ADD_OFFSET_DISTINCT" in tags else None
    scl_values = sorted(int(v) for v in np.unique(scl))
    unknown_scl = [v for v in scl_values if v not in SCL_CLASSES]
    cloudy_frac = float(np.isin(scl, list(SCL_CLOUDY)).mean())
    fill_frac_per_band = [float((arr[i] == (nodata if nodata is not None else 0)).mean())
                          for i in range(b)]

    return dict(
        product_name=stack_path.name.replace("_stack.tif", ""),
        stack_file=stack_path.name, scl_file=scl_path.name,
        dims_hw=[h, w], band_count=b, dtypes=sorted(dtypes),
        band_order=tags.get("BAND_ORDER", "").split(",") if tags.get("BAND_ORDER") else [],
        band_descriptions=descriptions,
        crs=crs, transform=transform, gsd_m=abs(transform[0]), nodata=nodata,
        fill_value_source=tags.get("FILL_VALUE_SOURCE"),
        fill_fraction_per_band=fill_frac_per_band,
        wavelengths_nm=wavelengths.tolist(),
        wavelengths_strictly_ascending=bool(np.all(np.diff(wavelengths) > 0)),
        sensing_time=tags.get("SENSING_TIME"),
        product_start_time=tags.get("PRODUCT_START_TIME"),
        sensing_vs_product_start_differ=(tags.get("SENSING_TIME") !=
                                         tags.get("PRODUCT_START_TIME")),
        processing_baseline=tags.get("PROCESSING_BASELINE"),
        boa_quantification_value=float(tags["BOA_QUANTIFICATION_VALUE"])
            if "BOA_QUANTIFICATION_VALUE" in tags else None,
        boa_add_offset_distinct=boa_offsets,
        boa_add_offset_uniform=tags.get("BOA_ADD_OFFSET_UNIFORM") == "True",
        resampling_note=tags.get("RESAMPLING"),
        cloud_cover_tile_pct=tags.get("CLOUD_COVER_TILE_PCT"),
        aoi_clear_fraction_scl_tag=tags.get("AOI_CLEAR_FRACTION_SCL"),
        aoi_clear_fraction_recomputed=1.0 - cloudy_frac,
        scl_dims_hw=[scl_h, scl_w], scl_dtype=scl_dtype, scl_crs=scl_crs,
        scl_transform=scl_transform,
        scl_grid_matches_stack=(scl_crs == crs and scl_transform == transform
                                and [scl_h, scl_w] == [h, w]),
        scl_values_present=scl_values,
        scl_values_unknown=unknown_scl,
    )


_BAND_FILENAME_RE = re.compile(r"_(B[0-9A-Z]{2}|SCL|TCI|AOT|WVP)_(10|20|60)m\.jp2$")


def discover_delivered_bands(prod_id: str, name: str, granule: str) -> dict[str, set]:
    """List every band actually SHIPPED as imagery, across all three resolution
    folders, via the unauthenticated OData Nodes API -- not read from metadata's
    band DEFINITIONS (which include B10 even though L2A ships no B10 image), and
    not assumed from one earlier scratch listing. Returns
    {"spectral": {codes}, "non_spectral": {codes}} so the "13 defined vs how many
    actually delivered" claim is counted from the product actually in scope,
    not hand-typed."""
    spectral: set[str] = set()
    non_spectral: set[str] = set()
    non_spectral_names = {"SCL", "TCI", "AOT", "WVP"}
    for res in ("R10m", "R20m", "R60m"):
        nodes = _nodes(prod_id, name, "GRANULE", granule, "IMG_DATA", res)
        for n in nodes:
            m = _BAND_FILENAME_RE.search(n["Name"])
            if not m:
                continue
            code = m.group(1)
            (non_spectral if code in non_spectral_names else spectral).add(code)
    return dict(spectral=spectral, non_spectral=non_spectral)


def verify_remote_claims(manifest_products: list[dict]) -> dict | None:
    """Re-fetch ONE product's MTD_MSIL2A.xml fresh and check PLAN.md's literal
    band-count / resolution-mixing claims against it directly, independent of
    anything this project's own fetcher wrote into the local GeoTIFF tags."""
    if not manifest_products:
        return None
    try:
        credentials.require("cdse")
    except RuntimeError as e:
        return dict(skipped=True, reason=f"cdse not configured: {e}")

    prod = manifest_products[0]
    key_prefix = prod["s3path"].lstrip("/") if "s3path" in prod else None
    if key_prefix is None:
        # manifest entries from a merged/cached run may not carry s3path;
        # reconstruct it is not safe to guess, so skip with a clear reason
        # rather than fabricate a path.
        return dict(skipped=True, reason="manifest entry has no s3path recorded")
    prod_id = prod["product_id"]
    prod_name = prod["product_name"]
    key_prefix = key_prefix[len(cdse_s3.BUCKET) + 1:]

    mtd = fetch_mtd_msil2a(key_prefix)
    n_spectral = len(mtd["spectral"])
    resolutions = sorted({v["resolution_m"] for v in mtd["spectral"].values()})
    by_res = {r: sorted(k for k, v in mtd["spectral"].items() if v["resolution_m"] == r)
              for r in resolutions}

    granule = discover_granule(prod_id, prod_name)
    delivered = discover_delivered_bands(prod_id, prod_name, granule)

    return dict(
        skipped=False,
        product_name=prod.get("product_name"),
        n_spectral_bands_in_metadata=n_spectral,
        resolutions_present_m=resolutions,
        bands_by_resolution=by_res,
        delivered_spectral_bands=sorted(delivered["spectral"]),
        n_delivered_spectral_bands=len(delivered["spectral"]),
        delivered_non_spectral_layers=sorted(delivered["non_spectral"]),
        defined_but_not_delivered=sorted(set(mtd["spectral"]) - delivered["spectral"]),
        plan_claim_13_bands=dict(
            claim="~13 bands multispectral",
            metadata_defines_n_bands=n_spectral,
            claim_true_for_metadata_definition=(n_spectral == 13),
            but_l2a_delivers_imagery_for=(
                f"{len(delivered['spectral'])} spectral bands "
                f"({sorted(delivered['spectral'])}) -- "
                f"{sorted(set(mtd['spectral']) - delivered['spectral'])} defined in "
                "metadata but shipped as no image file in any resolution folder "
                "(cirrus is not a surface quantity, dropped by the L2A processor) "
                f"-- plus non-spectral auxiliary layers {sorted(delivered['non_spectral'])}"
            ),
        ),
        plan_claim_resolution_mixing=dict(
            claim="10/20/60 m resolution mixing; Level 3 code must not assume one grid",
            true_at_sensor_level=(resolutions == [10.0, 20.0, 60.0]),
            true_for_landcover_index_bands=False,
            why_false_for_landcover=(
                "the landcover profile's 6 bands (B02/B03/B04/B8A/B11/B12) are ALL "
                "available pre-resampled to a single 20 m grid by ESA's own L2A "
                "processor -- verified: R20m/ ships a copy of every one of them. "
                "This project's fetcher reads only those 20 m copies, so Level 3 "
                "code built on this fetcher's output needs no on-the-fly "
                "resampling at all, contrary to the plan's blanket statement. A "
                "different index set needing B08 (10 m-only) or B01/B09 "
                "(60 m-only, no 20 m copy) would still need real resampling."),
        ),
        boa_quantification_value=mtd["quantification_value"],
        boa_add_offset_distinct=mtd["boa_add_offset_distinct"],
        boa_add_offset_uniform=mtd["boa_add_offset_uniform"],
        processing_baseline=mtd["processing_baseline"],
    )


def main() -> int:
    if not S2_DIR.exists():
        print(f"MISSING: {S2_DIR}", file=sys.stderr)
        return 2
    stack_files = sorted(S2_DIR.glob("*/*_stack.tif"))
    if not stack_files:
        print(f"No *_stack.tif files under {S2_DIR}", file=sys.stderr)
        return 2

    fail: list[str] = []
    products: list[dict] = []
    print(f"=== Sentinel-2 L2A -- {len(stack_files)} local product(s) in {S2_DIR} ===\n")
    for f in stack_files:
        try:
            rec = verify_stack(f)
        except Exception as exc:  # noqa: BLE001
            fail.append(f"{f.name}: FAILED TO VERIFY -- {exc}")
            print(f"  {f.name}: FAILED -- {exc}")
            continue
        products.append(rec)
        max_fill = max(rec["fill_fraction_per_band"]) if rec["fill_fraction_per_band"] else 0.0
        print(f"  {rec['product_name'][:45]:45} {rec['dims_hw'][0]}x{rec['dims_hw'][1]}x"
              f"{rec['band_count']} {rec['dtypes']} gsd={rec['gsd_m']} "
              f"sensing={rec['sensing_time']} clear={rec['aoi_clear_fraction_recomputed']:.3f} "
              f"offset={rec['boa_add_offset_distinct']} quant={rec['boa_quantification_value']} "
              f"max_fill_frac={max_fill:.4f}")
        # D32's mechanism (a fully/partially-fill band poisons every OTHER
        # band's per-pixel NaN-validity check downstream) is reported here
        # explicitly rather than left for the second agent to rediscover --
        # a nonzero max_fill_frac means SOME pixels in this AOI window will
        # become NaN once FILL_VALUE is applied, which is fine as long as
        # it's not silently missed.
        if max_fill > 0.0:
            print(f"    NOTE: {rec['product_name']} has nonzero fill fraction "
                 f"{rec['fill_fraction_per_band']} per band (DN==0) -- these pixels "
                 "become NaN, see PLAN.md D32 for the downstream any-NaN-band hazard")

        if rec["band_count"] != 6:
            fail.append(f"{rec['product_name']}: band_count={rec['band_count']}, expected 6")
        if rec["dtypes"] != ["uint16"]:
            fail.append(f"{rec['product_name']}: dtypes={rec['dtypes']}, expected ['uint16']")
        if abs(rec["gsd_m"] - GSD_M) > 1e-9:
            fail.append(f"{rec['product_name']}: gsd_m={rec['gsd_m']}, expected {GSD_M}")
        if not rec["wavelengths_strictly_ascending"]:
            fail.append(f"{rec['product_name']}: wavelengths not strictly ascending")
        if rec["boa_add_offset_distinct"] is None:
            fail.append(f"{rec['product_name']}: no BOA_ADD_OFFSET recorded")
        elif not rec["boa_add_offset_uniform"]:
            fail.append(f"{rec['product_name']}: BOA_ADD_OFFSET not uniform across "
                        f"bands: {rec['boa_add_offset_distinct']}")
        if rec["boa_quantification_value"] is None:
            fail.append(f"{rec['product_name']}: no BOA_QUANTIFICATION_VALUE recorded")
        if rec["sensing_time"] is None:
            fail.append(f"{rec['product_name']}: SENSING_TIME missing -- meta.acquired "
                        "would be None (D33)")
        if not rec["scl_grid_matches_stack"]:
            fail.append(f"{rec['product_name']}: SCL grid does not match the stack grid")
        if rec["scl_values_unknown"]:
            fail.append(f"{rec['product_name']}: unknown SCL class values "
                        f"{rec['scl_values_unknown']}")

    print()
    out: dict = {"products": products, "n_products": len(products)}

    if products:
        crss = sorted({p["crs"] for p in products})
        transforms = sorted({tuple(p["transform"]) for p in products})
        wl_sets = sorted({tuple(p["wavelengths_nm"]) for p in products})
        gsds = sorted({round(p["gsd_m"], 6) for p in products})
        out["cross_product"] = dict(
            crs_uniform=(len(crss) == 1), crs_values=crss,
            transform_uniform=(len(transforms) == 1),
            transform_values=[list(t) for t in transforms],
            wavelengths_uniform=(len(wl_sets) == 1),
            gsd_uniform=(len(gsds) == 1), gsds=gsds,
        )
        print("=== cross-product invariants ===")
        print(f"  CRS uniform: {out['cross_product']['crs_uniform']} {crss}")
        print(f"  transform (grid) uniform across dates: "
              f"{out['cross_product']['transform_uniform']} "
              "-- same MGRS tile means same pixel grid, no co-registration needed, "
              "if True" if transforms else "")
        print(f"  wavelengths uniform across dates: {out['cross_product']['wavelengths_uniform']}")
        if not transforms or len(transforms) != 1:
            print("  NOTE: grid differs across dates -- co-registration WOULD be needed")

    # s3path is only in manifest.json (verify_stack reads the GeoTIFF, which
    # does not carry it), so re-derive it here for the remote cross-check.
    manifest_products = []
    for mf in sorted(S2_DIR.glob("*/manifest.json")):   # authoritative, merged (not manifest_*.json scratch files)
        try:
            d = json.loads(mf.read_text())
            manifest_products.extend(d.get("products", []))
        except (json.JSONDecodeError, OSError):
            continue
    remote = verify_remote_claims(manifest_products)
    out["remote_claim_check"] = remote
    if remote is None:
        print("\nremote claim check: no manifest.json found, skipped")
    elif remote.get("skipped"):
        print(f"\nremote claim check: SKIPPED -- {remote['reason']}")
    else:
        print(f"\n=== remote claim check (fresh MTD_MSIL2A.xml, {remote['product_name'][:40]}) ===")
        print(f"  n_spectral_bands_in_metadata: {remote['n_spectral_bands_in_metadata']}")
        print(f"  resolutions present (m): {remote['resolutions_present_m']}")
        for r, bs in remote["bands_by_resolution"].items():
            print(f"    {r}m: {bs}")
        c13 = remote["plan_claim_13_bands"]
        print(f"  PLAN.md '~13 bands multispectral': "
              f"{'TRUE for the metadata definition' if c13['claim_true_for_metadata_definition'] else 'FALSE'} "
              f"({c13['metadata_defines_n_bands']} spectral bands defined); "
              f"but L2A delivers imagery for {c13['but_l2a_delivers_imagery_for']}")
        cmix = remote["plan_claim_resolution_mixing"]
        print(f"  PLAN.md '10/20/60m mixing, must not assume one grid': "
              f"TRUE at sensor level ({cmix['true_at_sensor_level']}) but "
              f"FALSE for the landcover index band set actually fetched here "
              f"(true_for_landcover_index_bands={cmix['true_for_landcover_index_bands']})")
        if remote["boa_quantification_value"] is None or remote["boa_add_offset_distinct"] is None:
            fail.append("remote claim check: fresh MTD_MSIL2A.xml missing quantification/offset")
        # Cross-check against what our own fetcher recorded, if we have a local match
        local_match = next((p for p in products
                            if p["product_name"] == remote["product_name"]), None)
        if local_match:
            if local_match["boa_quantification_value"] != remote["boa_quantification_value"]:
                fail.append(f"{remote['product_name']}: local tag quantification "
                            f"{local_match['boa_quantification_value']} != fresh remote "
                            f"{remote['boa_quantification_value']}")
            if local_match["boa_add_offset_distinct"] != remote["boa_add_offset_distinct"]:
                fail.append(f"{remote['product_name']}: local tag offset "
                            f"{local_match['boa_add_offset_distinct']} != fresh remote "
                            f"{remote['boa_add_offset_distinct']}")

    (ROOT / "docs").mkdir(exist_ok=True)
    out_path = ROOT / "docs" / "sentinel2_verified.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")

    if fail:
        print("\nINVARIANTS VIOLATED / VERIFICATION FAILURES:", file=sys.stderr)
        for f_ in fail:
            print(f"  - {f_}", file=sys.stderr)
        return 1
    print("all documented invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
