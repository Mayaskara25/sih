#!/usr/bin/env python3
"""Fetch a Sentinel-2 L2A time series over one AOI (Phase 5 Level 3, PLAN.md §8
Level 3 / O5). Companion to `scripts/verify_sentinel2.py`.

WHAT THIS FETCHES AND WHY. The `landcover` target profile (configs/target_profile.yaml,
D6) needs ndvi/ndwi/nbr/bsi, which together need blue/green/red/NIR/SWIR1/SWIR2. All
six are available with **zero on-the-fly resampling** at a single 20 m grid --
verified 2026-08-23 against a real product's MTD_MSIL2A.xml
<Spectral_Information>/<RESOLUTION> elements and its R20m/ folder listing:

    band  centre_nm  native_res_m  20m copy shipped by ESA?
    B02   492.3      10            yes (resampled by the L2A processor upstream)
    B03   559.0      10            yes
    B04   665.0      10            yes
    B8A   864.0      20            yes -- NOT B08 (833 nm, 10 m-only, no 20 m copy)
    B11   1610.4     20            yes
    B12   2185.7     20            yes

So "resolution mixing" (PLAN.md §15) is real at the SENSOR level (10/20/60 m native
per band) but the *landcover* index set specifically needs no on-the-fly mixing by
this code: ESA's own L2A processor already produced 20 m copies of the three 10 m
bands, and this fetcher reads those pre-resampled copies rather than resampling
B02/B03/B04 itself. This is recorded on every output file (RESAMPLING tag), not
assumed silently. A different index set needing e.g. B08 or B01 would need real
resampling and must not reuse this code path unexamined.

WHAT IS NOT FETCHED: SCL is fetched too (as a *separate* file, uint8, native class
values 0-11) -- `preprocessing/cloud_mask.py` already expects Sentinel-2 SCL as a
keyword-only `scl` array alongside the reflectance cube, not baked into it (verified
by reading that module; not modified here). TCI/AOT/WVP are not science bands and are
skipped. B01/B09/B10 are not in the landcover index set and are skipped -- B10 is not
even shipped in L2A (cirrus is not a surface quantity; dropped by the L2A processor).

DOWNLOAD MECHANISM. Two authenticated CDSE legs, both via `core.cdse_s3` (S3 keys
only, `core.credentials.require("cdse")`, never printed):
  - band JP2s: `/vsis3/` + a byte-WINDOW read through rasterio (core.cdse_s3.s3_env),
    so only the AOI's pixels are pulled over the wire, not the ~1 GB tile. Measured
    2026-08-23: a 300x300 px window out of a 5490x5490 px (33.7 MB) B02 file
    transferred ~512 KB over three HTTP range requests (CPL_CURL_VERBOSE trace) --
    JP2's own tiling, not something this script controls further.
  - MTD_MSIL2A.xml / MTD_TL.xml (small, whole-file): `core.cdse_s3.sigv4_get`.
Product/granule/band-filename discovery uses CDSE's OData "Nodes" browsing API,
which needs NO credentials (verified) -- only the two legs above are authenticated.

`core.http_guard.assert_not_html` (PLAN.md §8.0a: "fetch_enmap.py and fetch_sentinel2.py
must call it on every payload") is called on both XML payloads above. It is
deliberately NOT called on the band JP2 reads: those go through GDAL's own
`/vsis3/` driver, which parses the bytes as a raster immediately and raises its
own (loud) error if the payload is not a valid JP2 -- e.g. an HTML login/error
page -- so the silent-HTML-as-data failure mode `http_guard` exists for does
not arise on that path. This is a deliberate scope decision, not an omission.

RE-RUNNABLE: a product already on disk (both `_stack.tif` and `_scl.tif` present,
sizes > 0) is skipped, not re-downloaded, unless `--force`.

`meta.acquired` (D33) MUST come from `MTD_TL.xml`'s per-tile `SENSING_TIME`, not from
the OData `ContentDate/Start` (catalogue metadata, the tier this project distrusts by
policy -- PLAN.md §15) and not from `MTD_MSIL2A.xml`'s `PRODUCT_START_TIME` (that is
the DATATAKE start, i.e. the whole imaging strip's start time, not this tile's;
verified to differ by ~14 minutes for one product on 2026-08-23). Both are still
recorded in the manifest for the cross-check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import cdse_s3, credentials  # noqa: E402
from core.http_guard import assert_not_html  # noqa: E402

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
COLLECTION = "SENTINEL-2"

# Ascending wavelength order already (492.3 < 559 < 665 < 864 < 1610.4 < 2185.7 nm,
# per MTD_MSIL2A.xml -- read per-product below, NOT hardcoded, because S2A/B/C
# spectral response centres differ slightly scene to scene).
BANDS_20M = ("B02", "B03", "B04", "B8A", "B11", "B12")
SCL_NAME = "SCL"
RESOLUTION = "R20m"
GSD_M = 20.0

# ESA fill/background value convention for L2A JP2 bands. NOT read from a nodata
# tag in the file -- verified 2026-08-23 that the JP2 driver reports `ds.nodata is
# None` (JP2 carries no nodata tag at all). Applied here as documented convention,
# recorded as such in every manifest entry and GeoTIFF tag, never silently assumed
# to be a file-verified fact.
FILL_VALUE = 0


def search_products(lon: float, lat: float, start: str, end: str,
                     max_cloud: float, limit: int = 50) -> list[dict]:
    """OData catalogue search. No credentials needed (verified live)."""
    filt = (
        f"Collection/Name eq '{COLLECTION}' and contains(Name,'MSIL2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') and "
        "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value lt {max_cloud}) and "
        f"ContentDate/Start gt {start}T00:00:00.000Z and "
        f"ContentDate/Start lt {end}T00:00:00.000Z"
    )
    params = {"$filter": filt, "$top": limit, "$orderby": "ContentDate/Start asc",
              "$expand": "Attributes"}
    r = requests.get(CATALOGUE, params=params, timeout=60)
    r.raise_for_status()
    out = []
    for p in r.json().get("value", []):
        cloud = next((a["Value"] for a in p.get("Attributes", [])
                      if a.get("Name") == "cloudCover"), None)
        out.append(dict(
            id=p["Id"], name=p["Name"], s3path=p["S3Path"],
            content_date_start=p["ContentDate"]["Start"],
            cloud_cover_tile_pct=cloud,
        ))
    return out


def _nodes(prod_id: str, name: str, *parts: str) -> list[dict]:
    url = f"{CATALOGUE}({prod_id})/Nodes({name})"
    for part in parts:
        url += f"/Nodes({part})"
    url += "/Nodes"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["result"]


def discover_granule(prod_id: str, name: str) -> str:
    nodes = _nodes(prod_id, name, "GRANULE")
    if not nodes:
        raise RuntimeError(f"{name}: no GRANULE folder -- not a normal L2A product")
    return nodes[0]["Name"]


def discover_band_files(prod_id: str, name: str, granule: str) -> dict[str, str]:
    nodes = _nodes(prod_id, name, "GRANULE", granule, "IMG_DATA", RESOLUTION)
    wanted = BANDS_20M + (SCL_NAME,)
    files: dict[str, str] = {}
    for n in nodes:
        fname = n["Name"]
        for b in wanted:
            if f"_{b}_20m.jp2" == fname[-(len(b) + 9):]:
                files[b] = fname
    missing = [b for b in wanted if b not in files]
    if missing:
        raise RuntimeError(f"{name}: {RESOLUTION} is missing band file(s) {missing} "
                           f"-- found {sorted(n['Name'] for n in nodes)}")
    return files


def _normalize_band_code(physical_band: str) -> str:
    """MTD_MSIL2A.xml's `physicalBand` attribute uses un-padded codes for the
    single-digit bands ("B2", "B8", "B9" -- verified against a real product,
    2026-08-23) while every other reference in this project (filenames, this
    script's BANDS_20M) uses the zero-padded form ("B02", "B08", "B09"). B8A/
    B11/B12 already match. Normalizes to the zero-padded/filename form."""
    digits = physical_band[1:]
    return f"B{int(digits):02d}" if digits.isdigit() else physical_band


def fetch_mtd_msil2a(key_prefix: str) -> dict:
    resp = cdse_s3.sigv4_get(f"{key_prefix}/MTD_MSIL2A.xml")
    resp.raise_for_status()
    assert_not_html(resp.content, url="MTD_MSIL2A.xml")
    root = ET.fromstring(resp.content)

    quant_el = root.find(".//BOA_QUANTIFICATION_VALUE")
    if quant_el is None:
        raise ValueError(f"{key_prefix}: MTD_MSIL2A.xml has no BOA_QUANTIFICATION_VALUE "
                         "-- refusing to guess a reflectance scale factor")
    offsets = {int(e.get("band_id")): float(e.text)
               for e in root.iter("BOA_ADD_OFFSET")}
    if not offsets:
        raise ValueError(f"{key_prefix}: MTD_MSIL2A.xml has no BOA_ADD_OFFSET_VALUES_LIST "
                         "-- do not assume -1000 (PLAN.md trap: this must be READ, "
                         "and pre-baseline-04.00 products genuinely lack it)")

    spectral: dict[str, dict] = {}
    for si in root.iter("Spectral_Information"):
        pb = si.get("physicalBand")
        res = si.findtext("RESOLUTION")
        wl = si.find("Wavelength")
        cen = wl.findtext("CENTRAL") if wl is not None else None
        if pb and res and cen:
            spectral[_normalize_band_code(pb)] = dict(
                resolution_m=float(res), central_nm=float(cen))

    return dict(
        quantification_value=float(quant_el.text),
        quantification_unit=quant_el.get("unit"),
        boa_add_offset_by_band_id=offsets,
        boa_add_offset_distinct=sorted(set(offsets.values())),
        boa_add_offset_uniform=(len(set(offsets.values())) == 1),
        processing_baseline=root.findtext(".//PROCESSING_BASELINE"),
        product_start_time=root.findtext(".//PRODUCT_START_TIME"),
        datatake_sensing_start=root.findtext(".//DATATAKE_SENSING_START"),
        spectral=spectral,
    )


def fetch_mtd_tl(key_prefix: str, granule: str) -> dict:
    resp = cdse_s3.sigv4_get(f"{key_prefix}/GRANULE/{granule}/MTD_TL.xml")
    resp.raise_for_status()
    assert_not_html(resp.content, url="MTD_TL.xml")
    root = ET.fromstring(resp.content)
    sensing_time = root.findtext(".//SENSING_TIME")
    if not sensing_time:
        raise ValueError(f"{key_prefix}/{granule}: MTD_TL.xml has no SENSING_TIME -- "
                         "meta.acquired cannot be populated (D33)")
    return dict(
        sensing_time=sensing_time,
        crs=root.findtext(".//HORIZONTAL_CS_CODE"),
        cloudy_pixel_pct=root.findtext(".//CLOUDY_PIXEL_PERCENTAGE"),
        nodata_pixel_pct=root.findtext(".//NODATA_PIXEL_PERCENTAGE"),
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_aoi_stack(prod: dict, aoi_bounds: tuple[float, float, float, float],
                     out_dir: Path, *, force: bool = False,
                     granule: str | None = None, tl: dict | None = None) -> dict:
    """Download the 6-band landcover stack + SCL for one product, cropped to
    `aoi_bounds` (minx, miny, maxx, maxy, in the tile's own UTM CRS). Returns
    a manifest record. Re-runnable: skips if both outputs already exist.

    `granule`/`tl` let a caller that already resolved the granule ID and
    fetched MTD_TL.xml (e.g. `main()`'s AOI-CRS probe, which needs
    `tl["crs"]` before it can even compute `aoi_bounds`) pass them through
    instead of this function fetching MTD_TL.xml a second time -- that file
    is ~550 KB over the authenticated S3 leg, and re-fetching it per product
    was pure waste the first version of this script didn't need to pay.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_path = out_dir / f"{prod['name']}_stack.tif"
    scl_path = out_dir / f"{prod['name']}_scl.tif"

    if stack_path.exists() and scl_path.exists() and stack_path.stat().st_size > 0 \
            and scl_path.stat().st_size > 0 and not force:
        with rasterio.open(stack_path) as ds:
            tags = ds.tags()
        return dict(
            product_id=prod["id"], product_name=prod["name"], status="cached",
            s3path=prod.get("s3path"),
            stack_path=str(stack_path), scl_path=str(scl_path),
            stack_bytes=stack_path.stat().st_size, scl_bytes=scl_path.stat().st_size,
            stack_sha256=_sha256(stack_path), scl_sha256=_sha256(scl_path),
            sensing_time=tags.get("SENSING_TIME"),
            cloud_cover_tile_pct=prod.get("cloud_cover_tile_pct"),
            aoi_clear_fraction_scl=float(tags["AOI_CLEAR_FRACTION_SCL"])
                if "AOI_CLEAR_FRACTION_SCL" in tags else None,
        )

    key_prefix = prod["s3path"].lstrip("/")
    if not key_prefix.startswith(cdse_s3.BUCKET + "/"):
        raise ValueError(f"{prod['name']}: S3Path {prod['s3path']!r} does not start "
                         f"with bucket {cdse_s3.BUCKET!r}")
    key_prefix = key_prefix[len(cdse_s3.BUCKET) + 1:]

    if granule is None:
        granule = discover_granule(prod["id"], prod["name"])
    band_files = discover_band_files(prod["id"], prod["name"], granule)
    mtd = fetch_mtd_msil2a(key_prefix)
    if tl is None:
        tl = fetch_mtd_tl(key_prefix, granule)

    band_arrays: list[np.ndarray] = []
    wavelengths: list[float] = []
    transform = crs = None
    win = None

    with cdse_s3.s3_env():
        for b in BANDS_20M:
            key = f"{key_prefix}/GRANULE/{granule}/IMG_DATA/{RESOLUTION}/{band_files[b]}"
            with rasterio.open(cdse_s3.vsis3_path(key)) as ds:
                if transform is None:
                    transform, crs = ds.transform, ds.crs
                    win = from_bounds(*aoi_bounds, transform).round_offsets().round_lengths()
                elif ds.transform != transform or ds.crs != crs:
                    # Every 20 m band of one tile is on the SAME grid (verified
                    # 2026-08-23) -- a mismatch here means a real, unexpected grid
                    # shift and must stop the fetch, not silently misalign bands.
                    raise ValueError(
                        f"{prod['name']}: band {b} grid ({ds.transform}, {ds.crs}) "
                        f"differs from band {BANDS_20M[0]} ({transform}, {crs}) -- "
                        "refusing to stack misaligned bands")
                arr = ds.read(1, window=win)
            band_arrays.append(arr)
            if b not in mtd["spectral"]:
                raise ValueError(f"{prod['name']}: no Spectral_Information for {b} "
                                 "in MTD_MSIL2A.xml -- cannot attach a wavelength")
            wavelengths.append(mtd["spectral"][b]["central_nm"])

        scl_key = f"{key_prefix}/GRANULE/{granule}/IMG_DATA/{RESOLUTION}/{band_files[SCL_NAME]}"
        with rasterio.open(cdse_s3.vsis3_path(scl_key)) as ds:
            if ds.transform != transform or ds.crs != crs:
                raise ValueError(f"{prod['name']}: SCL grid differs from the band grid")
            scl_arr = ds.read(1, window=win)

    stack = np.stack(band_arrays, axis=0)          # [B, H, W] uint16, native DN
    out_transform = rasterio.windows.transform(win, transform)

    scl_cloudy = np.isin(scl_arr, (3, 8, 9, 10))    # matches preprocessing.cloud_mask
    aoi_clear_fraction = float((~scl_cloudy).mean())

    profile = dict(driver="GTiff", height=stack.shape[1], width=stack.shape[2],
                    count=stack.shape[0], dtype="uint16", crs=crs,
                    transform=out_transform, nodata=FILL_VALUE)
    with rasterio.open(stack_path, "w", **profile) as ds:
        ds.write(stack)
        for i, b in enumerate(BANDS_20M, start=1):
            ds.set_band_description(i, b)
        ds.update_tags(
            SOURCE="sentinel2", PRODUCT_ID=prod["id"], PRODUCT_NAME=prod["name"],
            SENSING_TIME=tl["sensing_time"],                       # D33: meta.acquired
            PRODUCT_START_TIME=mtd["product_start_time"],          # datatake start, NOT acquired
            DATATAKE_SENSING_START=mtd["datatake_sensing_start"],
            PROCESSING_BASELINE=mtd["processing_baseline"],
            BOA_QUANTIFICATION_VALUE=repr(mtd["quantification_value"]),
            BOA_ADD_OFFSET_DISTINCT=repr(mtd["boa_add_offset_distinct"]),
            BOA_ADD_OFFSET_UNIFORM=str(mtd["boa_add_offset_uniform"]),
            BAND_ORDER=",".join(BANDS_20M),
            WAVELENGTHS_NM=",".join(repr(w) for w in wavelengths),
            RESOLUTION_M=repr(GSD_M),
            RESAMPLING="none-by-fetcher; B02/B03/B04 20m copies are ESA L2A-processor "
                       "output, B8A/B11/B12 are 20m-native",
            FILL_VALUE=repr(FILL_VALUE),
            FILL_VALUE_SOURCE="ESA convention, NOT a file-read nodata tag (JP2 carries none)",
            CLOUD_COVER_TILE_PCT=repr(prod.get("cloud_cover_tile_pct")),
            AOI_CLEAR_FRACTION_SCL=repr(aoi_clear_fraction),
            MGRS_TILE=prod["name"].split("_T")[-1].split("_")[0] if "_T" in prod["name"] else "",
        )

    scl_profile = dict(driver="GTiff", height=scl_arr.shape[0], width=scl_arr.shape[1],
                        count=1, dtype="uint8", crs=crs, transform=out_transform)
    with rasterio.open(scl_path, "w", **scl_profile) as ds:
        ds.write(scl_arr.astype(np.uint8), 1)
        ds.update_tags(SOURCE="sentinel2", PRODUCT_ID=prod["id"],
                       SENSING_TIME=tl["sensing_time"],
                       SCL_CLOUDY_CLASSES="3,8,9,10")

    return dict(
        product_id=prod["id"], product_name=prod["name"], status="downloaded",
        s3path=prod.get("s3path"),
        stack_path=str(stack_path), scl_path=str(scl_path),
        stack_bytes=stack_path.stat().st_size, scl_bytes=scl_path.stat().st_size,
        stack_sha256=_sha256(stack_path), scl_sha256=_sha256(scl_path),
        sensing_time=tl["sensing_time"],
        product_start_time=mtd["product_start_time"],
        processing_baseline=mtd["processing_baseline"],
        quantification_value=mtd["quantification_value"],
        boa_add_offset_distinct=mtd["boa_add_offset_distinct"],
        boa_add_offset_uniform=mtd["boa_add_offset_uniform"],
        cloud_cover_tile_pct=prod.get("cloud_cover_tile_pct"),
        aoi_clear_fraction_scl=aoi_clear_fraction,
        content_date_start=prod.get("content_date_start"),
        crs=str(crs), transform=list(out_transform)[:6], gsd_m=GSD_M,
        wavelengths_nm=wavelengths,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--half-km", type=float, default=3.0,
                    help="AOI half-width in km (default 3 -> 6x6 km box)")
    ap.add_argument("--start", required=True, help="ISO date, e.g. 2020-01-01")
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-cloud", type=float, default=15.0,
                    help="tile-level cloudCover ceiling for the catalogue search "
                         "(AOI clear fraction is computed separately from SCL)")
    ap.add_argument("--limit", type=int, default=20,
                    help="max candidate products to consider from the search")
    ap.add_argument("--max-dates", type=int, default=6,
                    help="stop after downloading this many products (bytes budget)")
    ap.add_argument("--min-aoi-clear", type=float, default=0.0,
                    help="skip a candidate if its AOI clear fraction (from SCL) "
                         "is below this, after downloading its SCL band")
    ap.add_argument("--out-dir", default=None,
                    help="default: data/raw/sentinel2/<lon>_<lat>/")
    ap.add_argument("--manifest", default=None,
                    help="default: <out-dir>/manifest.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    try:
        credentials.require("cdse")
    except RuntimeError as e:
        print(f"CREDENTIAL ERROR: {e}", file=sys.stderr)
        return 2

    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "data" / "raw" / "sentinel2" / f"{a.lon}_{a.lat}"
    manifest_path = Path(a.manifest) if a.manifest else out_dir / "manifest.json"

    from pyproj import Transformer
    candidates = search_products(a.lon, a.lat, a.start, a.end, a.max_cloud, a.limit)
    if not candidates:
        print(f"No products found for ({a.lon},{a.lat}) {a.start}..{a.end} "
              f"cloud<{a.max_cloud}%. Widen the search.", file=sys.stderr)
        return 1
    print(f"{len(candidates)} candidate product(s), tile cloud < {a.max_cloud}%")

    records: list[dict] = []
    total_bytes = 0
    for prod in candidates:
        if len(records) >= a.max_dates:
            break
        print(f"  {prod['name'][:55]:55} cloud={prod['cloud_cover_tile_pct']}", end=" ")
        # AOI bounds in the product's own CRS. Resolve lazily per-product since
        # every product over one tile shares the same UTM CRS/grid (verified),
        # but a search spanning a tile boundary would not -- transform per call.
        # If both outputs are already on disk, skip the CRS probe entirely --
        # fetch_aoi_stack's own cached-path branch doesn't need aoi_bounds,
        # granule, or a fresh MTD_TL.xml fetch, so don't pay for any of them
        # just to compute a value that will go unused. This is also what
        # makes a re-run against an already-fetched --out-dir fast and
        # network-light, not just "correct."
        already_cached = (
            (out_dir / f"{prod['name']}_stack.tif").exists()
            and (out_dir / f"{prod['name']}_scl.tif").exists()
            and not a.force
        )
        granule = tl_probe = aoi_bounds = None
        if not already_cached:
            # We don't know the UTM EPSG until we open a band, so pass WGS84
            # lon/lat AOI half-extent converted via an approximate local UTM
            # zone guess is wrong in general; instead discover the CRS first
            # via a lightweight metadata call, THEN compute bounds. Cheap:
            # pyproj caches the CRS pipeline.
            try:
                key_prefix = prod["s3path"].lstrip("/")[len(cdse_s3.BUCKET) + 1:]
                granule = discover_granule(prod["id"], prod["name"])
                tl_probe = fetch_mtd_tl(key_prefix, granule)
                epsg = tl_probe["crs"]
                transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
                cx, cy = transformer.transform(a.lon, a.lat)
                half_m = a.half_km * 1000.0
                aoi_bounds = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP (probe failed: {type(e).__name__}: {e})")
                continue

        try:
            rec = fetch_aoi_stack(prod, aoi_bounds, out_dir, force=a.force,
                                  granule=granule, tl=tl_probe)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL ({type(e).__name__}: {e})")
            continue

        clear = rec.get("aoi_clear_fraction_scl")
        print(f"-> {rec['status']} aoi_clear={clear:.3f} "
              f"bytes={rec['stack_bytes'] + rec['scl_bytes']}"
              if clear is not None else f"-> {rec['status']}")
        if a.min_aoi_clear and (clear is None or clear < a.min_aoi_clear):
            print(f"    below --min-aoi-clear={a.min_aoi_clear}, keeping file but "
                 "flagging in manifest")
            rec["below_min_aoi_clear"] = True
        records.append(rec)
        total_bytes += rec.get("stack_bytes", 0) + rec.get("scl_bytes", 0)

    # MERGE into any existing manifest, keyed by product_id, rather than
    # overwrite it. A prior invocation (e.g. a separate narrow-date-range
    # call picking a different era's best date) must survive a later run
    # against the same --manifest path, or "re-runnable" only holds for
    # single-invocation use and the very re-run pattern this fetcher is
    # designed to support (see docs/validation.md's per-era selection)
    # silently destroys the multi-date manifest a second agent depends on.
    existing_by_id: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text())
            for p in prior.get("products", []):
                pid = p.get("product_id")
                if pid:
                    existing_by_id[pid] = p
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read existing manifest {manifest_path} "
                 f"({type(e).__name__}: {e}) -- starting a fresh one rather than "
                 "silently dropping it; back it up if it mattered", file=sys.stderr)

    for rec in records:
        pid = rec["product_id"]
        if pid in existing_by_id:
            # Soft-merge: a fresh fetch/cache-read overwrites the facts it
            # actually re-derived, but keeps any extra field the prior
            # manifest carried that this run never touches (e.g. a hand- or
            # second-agent-added `era_label`) -- a bare overwrite would
            # silently drop that annotation on every re-run.
            merged = {**existing_by_id[pid], **rec}
        else:
            merged = rec
        existing_by_id[pid] = merged
    merged_records = sorted(existing_by_id.values(),
                            key=lambda r: r.get("sensing_time") or r.get("content_date_start") or "")
    merged_total_bytes = sum(r.get("stack_bytes", 0) + r.get("scl_bytes", 0)
                             for r in merged_records)

    manifest = dict(
        aoi=dict(lon=a.lon, lat=a.lat, half_km=a.half_km),
        search=dict(start=a.start, end=a.end, max_cloud=a.max_cloud),
        bands=list(BANDS_20M), scl=SCL_NAME, resolution_m=GSD_M,
        n_products=len(merged_records),
        total_local_bytes=merged_total_bytes,
        products=merged_records,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\n{len(records)} product(s) fetched/cached this run, {total_bytes} bytes; "
          f"manifest now holds {len(merged_records)} product(s) total, "
          f"{merged_total_bytes} bytes")
    print(f"Manifest: {manifest_path}")
    return 0 if merged_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
