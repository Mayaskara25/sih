#!/usr/bin/env python3
"""Pick EnMAP L2A scenes by LOCATION, not by browsing date folders.

`download.geoservice.dlr.de/ENMAP/files/L2A/` is laid out YYYY/MM/DD/datatake/NN,
so finding a scene over a given place by clicking is hopeless -- there are 13 017
L2A scenes over India alone. The STAC API is searchable by bounding box and needs
NO credentials (verified 2026-08-21), so search here and download in the browser.

    python scripts/find_enmap_scenes.py --bbox 68,6,98,36 --max-cloud 10 --limit 20

Writes docs/enmap_candidates.json with one entry per scene: id, date, cloud, EPSG,
declared dtype, and every asset URL. Paste the `image` URL into the browser tab
that is logged in to the EnMAP Access Service.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

SEARCH = "https://geoservice.dlr.de/eoc/ogc/stac/v1/search"
COLLECTION = "ENMAP_HSI_L2A"

# What Phase 5 actually needs per scene. `image` is the 224-band cube; `metadata`
# carries the wavelength table (the open question in 8.0); the quality masks feed
# the cloud/shadow handling in 9.4. The vnir/swir/thumbnail assets are browse
# overviews and are NOT science data -- downloading all 13 assets triples the
# bytes for nothing.
WANTED = ("image", "metadata", "quality_classes", "quality_cloud", "defective_pixel_mask")


def _datatake(scene_id: str) -> str:
    m = re.search(r"DT\d+", scene_id)
    return m.group() if m else scene_id


def _num(v, *, default):
    """Coerce a STAC property that SHOULD be numeric but, verified 2026-08-23,
    genuinely is not always (eo:cloud_cover/eo:snow_cover arrive as int on some
    features and as a numeric string on others from the same search). `None`
    (property absent) returns `default`; a value that is neither None nor
    coercible to float is returned AS-IS rather than silently dropped, so a
    genuinely unexpected shape surfaces downstream instead of vanishing here."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def search(bbox: str, start: str | None, end: str | None,
           max_cloud: float, limit: int, per_datatake: int,
           assets: tuple[str, ...], page_cap: int = 40) -> list[dict]:
    params = {"collections": COLLECTION, "bbox": bbox, "limit": 100}
    if start or end:
        params["datetime"] = f"{start or '..'}/{end or '..'}"
    url, out, pages = SEARCH, [], 0
    seen_dt: dict[str, int] = {}
    while url and len(out) < limit and pages < page_cap:
        r = requests.get(url, params=params if pages == 0 else None, timeout=90)
        r.raise_for_status()
        doc = r.json()
        for f in doc.get("features", []):
            p = f["properties"]
            # DLR's live STAC API serializes eo:cloud_cover (and eo:snow_cover)
            # inconsistently -- int on some features, a numeric STRING on others,
            # verified 2026-08-23 by paging the raw search response for this same
            # bbox (both types seen within the first 5 pages, not tied to a
            # particular page or date range). An un-coerced `> max_cloud` compare
            # raises TypeError on the string-typed features rather than filtering
            # them, so every numeric catalogue field read here is coerced via
            # `_num` -- never assumed to already be numeric just because the STAC
            # spec says it should be.
            cloud = _num(p.get("eo:cloud_cover"), default=101.0)
            if cloud > max_cloud:
                continue
            # One overpass produces consecutive tiles seconds apart. They are
            # near-duplicates, and letting several through would put correlated
            # scenes on both sides of the scene-level split in 3B -- the exact
            # leakage the plan is built to prevent.
            dt = _datatake(f["id"])
            if seen_dt.get(dt, 0) >= per_datatake:
                continue
            seen_dt[dt] = seen_dt.get(dt, 0) + 1
            out.append({
                "id": f["id"],
                "date": p.get("datetime", "")[:19],
                "cloud": cloud,
                "snow": _num(p.get("eo:snow_cover"), default=None),
                "epsg": p.get("proj:epsg"),
                # Declared by the catalogue -- still UNVERIFIED against the file.
                # PLAN.md 8.0 requires opening the product to confirm it.
                "declared_dtype": p.get("data_type"),
                "sun_elevation": p.get("view:sun_elevation"),
                "datatake": dt,
                "assets": {k: v["href"] for k, v in f["assets"].items()
                           if k in assets},
            })
            if len(out) >= limit:
                break
        url = next((l["href"] for l in doc.get("links", []) if l.get("rel") == "next"), None)
        pages += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", default="68,6,98,36", help="minlon,minlat,maxlon,maxlat (default: India)")
    ap.add_argument("--start", help="ISO date, e.g. 2025-01-01")
    ap.add_argument("--end", help="ISO date")
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--per-datatake", type=int, default=1,
                    help="max scenes from one overpass (default 1; raise only deliberately)")
    ap.add_argument("--all-assets", action="store_true",
                    help="emit all 13 assets instead of the 5 Phase 5 needs")
    ap.add_argument("--out", default="docs/enmap_candidates.json")
    a = ap.parse_args()

    assets = () if a.all_assets else WANTED
    scenes = search(a.bbox, a.start, a.end, a.max_cloud, a.limit,
                    a.per_datatake, assets or tuple())
    if a.all_assets:
        scenes = search(a.bbox, a.start, a.end, a.max_cloud, a.limit,
                        a.per_datatake, tuple(WANTED) + ("vnir", "swir", "thumbnail",
                        "quality_cloud_shadow", "quality_haze", "quality_cirrus",
                        "quality_snow", "quality_testflags"))
    if not scenes:
        print(f"No scenes under {a.max_cloud}% cloud in bbox {a.bbox}. Widen the filter.")
        return 1

    print(f"{len(scenes)} scene(s), cloud <= {a.max_cloud}%\n")
    print(f"{'date':20} {'cloud':>5} {'snow':>5} {'epsg':>6}  {'datatake':14} id")
    for s in scenes:
        print(f"{s['date']:20} {s['cloud']:>5} {s['snow']:>5} {s['epsg']:>6}  "
              f"{s['datatake']:14} {s['id'][:40]}")
    dts = {s["datatake"] for s in scenes}
    print(f"\n{len(scenes)} scene(s) from {len(dts)} distinct overpass(es); "
          f"{len(scenes[0]['assets'])} asset(s) each.")
    if len(dts) < len(scenes):
        print("WARNING: some scenes share an overpass and are near-duplicates.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenes, indent=2))
    print(f"\nWrote {out}  ({len(scenes)} scenes, all asset URLs included)")
    print("Download the 'image' asset of each in a browser logged in to the")
    print("EnMAP Access Service, then verify with scripts/verify_phase5_datasets.py.")
    print("Catalogue metadata is NOT verification -- open the files (PLAN.md 8.0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
