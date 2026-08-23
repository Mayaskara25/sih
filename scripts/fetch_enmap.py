#!/usr/bin/env python3
"""Fetch EnMAP L2A products from DLR, re-runnably, with a manifest.

WHY THIS EXISTS. `scripts/verify_access.py::dlr()` proves the DLR download leg
works end to end (verified 2026-08-23). But nothing in this repo could turn
that into a scene on disk: of the 20 products in `data/raw/enmap/`, 8 were
fetched by hand following `docs/enmap_handover.md` and 12 by a script that was
never committed. A fresh clone cannot reproduce any of them. This script closes
that gap, and its `--reconcile` mode retroactively records the 20 that already
exist so they stop being unreproducible artifacts.

REUSE, NOT REIMPLEMENTATION.
  - Auth: `scripts.verify_access._cas_login` (imported, not re-derived) drives
    the CAS credential form; the CAS entry URL is the same constant
    (`scripts.verify_access.CAS`) verify_access.py itself uses.
  - Search: `scripts.find_enmap_scenes.search` does the STAC bbox query
    (no credentials needed); this module does not touch the STAC query logic.
  - Magic-byte / HTML-wall checking: `core.http_guard.assert_magic` /
    `assert_not_html` -- the same functions verify_access.py uses to tell a
    real TIFF from DLR's 200-status HTML login/error page (O11).
  - Sidecar naming: `scripts.verify_enmap.find_metadata` -- the SAME
    try-both-spellings lookup verify_enmap.py and
    preprocessing/raster_loader.py already use for
    `-METADATA.XML` vs `-METADATA.XML.XML` (both genuinely occur; measured
    2026-08-23 across the 20 local scenes: 13 single-suffix, 7 double-suffix
    -- not the "1 of 8 / 7 of 8" split the original handoff note describes,
    because that note was about the original 8, and 12 more scenes have
    landed in the directory since. Recorded here as a documentation/disk
    contradiction, not silently reconciled).

BOUNDED BY DEFAULT. Each `image` asset is a ~400-500 MB cube (measured via
HEAD, 2026-08-23: 458 234 018 B for one scene). `--limit` (default 5) caps how
many NEW scenes this run will download; scenes already complete on disk don't
count against it. Before touching the network for real bytes, the projected
total (from HEAD Content-Length, in the same authenticated session used for
the real GETs) is printed and confirmed -- `--yes` skips the prompt, and a
non-interactive session without `--yes` refuses rather than hanging or
guessing. `shutil.disk_usage` on `--out-dir` is checked against
`--min-free-gb` before any GET. `thumbnail`/`vnir`/`swir` browse assets are
never requested (find_enmap_scenes.WANTED already excludes them, and this
script's own asset key list is a subset of that). Quality masks
(quality_classes/quality_cloud/defective_pixel_mask) are opt-in via
`--with-quality-masks`; the default fetches only the cube and its metadata
sidecar.

TRUNCATION. A cut-off BigTIFF still starts with the right 4 magic bytes
(`II+\0`), so `assert_magic` alone cannot catch a truncated download. Every
download also compares bytes actually written against the response's
Content-Length header (present on every DLR asset probed 2026-08-23) and
refuses the file on any mismatch -- see `_download_asset`. Downloads land in
`<name>.part` (built by string concatenation, never `Path.with_suffix`, so a
half-written `*-SPECTRAL_IMAGE_COG.TIF.part` never matches
`verify_enmap.py`'s `*-SPECTRAL_IMAGE_COG.TIF` glob) and are renamed only
after both checks pass.

THREE FAILURE MODES, kept separate exactly as `verify_access.py::dlr()` does
(O11: collapsing them into "download failed" cost a day):
  (a) DLR_USERNAME/DLR_PASSWORD not set          -> credentials.require() raises
                                                      before any network call.
  (b) credentials present but REJECTED, or the
      account is not activated for this CAS
      service                                    -> the bare CAS login (no
                                                      service named) returns no
                                                      ticket cookie.
  (c) authenticated, but a specific asset returns
      non-data (HTML wall / error page)           -> AssetEntitlementError,
                                                      raised from the SAME
                                                      assert_magic/assert_not_html
                                                      check verify_access.py
                                                      uses for O11. Recorded
                                                      per-asset; does not abort
                                                      the whole run.

MANIFEST: `docs/enmap_fetch_manifest.json` (not `data/raw/enmap/`, because
`data/` is gitignored -- a manifest that cannot be committed cannot be the
thing that makes the local scenes reproducible for a fresh clone). Merged by
product id via `merge_manifest`, itself merging per-asset so a `--reconcile`
run that only sees `image`+`metadata` locally does not erase a prior run's
quality-mask records, or vice versa. No credential material is ever written
to it.

    python scripts/fetch_enmap.py --limit 2 --max-cloud 5 --yes
    python scripts/fetch_enmap.py --reconcile

Companion: `scripts/verify_enmap.py` opens every local product and checks the
pixel data -- run it after fetching; this script does not duplicate its checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import credentials  # noqa: E402
from core.http_guard import assert_magic, assert_not_html  # noqa: E402
from scripts.verify_access import CAS, _cas_login  # noqa: E402
from scripts.find_enmap_scenes import COLLECTION, _datatake, _num, search as stac_search  # noqa: E402
from scripts.verify_enmap import SPECTRAL_SUFFIX, find_metadata  # noqa: E402

STAC_ITEM_URL = f"https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/{COLLECTION}/items/{{id}}"

# What this fetcher will ever request. A strict subset of find_enmap_scenes.WANTED
# (which is itself a subset of the 13 assets STAC advertises) -- thumbnail/vnir/swir
# are browse overviews, never science data, and thumbnail is JPEG despite sitting
# beside a dozen GeoTIFFs (verified in core/http_guard.py's MAGIC table).
CORE_ASSETS = ("image", "metadata")
QUALITY_ASSETS = ("quality_classes", "quality_cloud", "defective_pixel_mask")

DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "enmap"
DEFAULT_MANIFEST = ROOT / "docs" / "enmap_fetch_manifest.json"


class AssetEntitlementError(RuntimeError):
    """Authenticated, but the asset did not return real data (O11-shaped)."""


class TruncatedDownloadError(RuntimeError):
    """Bytes written did not match the server's declared Content-Length."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _asset_filename(url: str) -> str:
    """The filename this fetcher will save an asset as: exactly what the URL
    gives (DLR sends no Content-Disposition -- verified 2026-08-23 against a
    live metadata GET), never normalised. This is why a freshly downloaded
    metadata sidecar always lands as `-METADATA.XML` (the STAC href's own
    suffix) -- the `-METADATA.XML.XML` form seen on 7 of the 20 local scenes
    came from something other than this code path and is not reproduced."""
    return unquote(Path(urlparse(url).path).name)


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def _existing_metadata_path(spectral_path: Path) -> Path | None:
    """Both spellings, reusing verify_enmap.find_metadata rather than a third
    copy of the try-both logic. Works even if spectral_path itself does not
    exist yet -- find_metadata only derives the stem from the name."""
    try:
        p = find_metadata(spectral_path)
    except FileNotFoundError:
        return None
    return p if p.exists() and p.stat().st_size > 0 else None


def _local_path_for(out_dir: Path, scene_id: str, key: str, url: str) -> Path:
    if key == "metadata":
        spectral_path = out_dir / f"{scene_id}{SPECTRAL_SUFFIX}"
        existing = _existing_metadata_path(spectral_path)
        if existing is not None:
            return existing
    return out_dir / _asset_filename(url)


def _asset_is_present(out_dir: Path, scene_id: str, key: str, url: str) -> bool:
    p = _local_path_for(out_dir, scene_id, key, url)
    return p.exists() and p.stat().st_size > 0


def _download_asset(session: requests.Session, url: str, dest_path: Path, kind: str) -> dict:
    """Stream `url` to `dest_path.name + '.part'`, verify magic bytes AND
    exact byte count (Content-Length, when the server sends one) before
    accepting, then rename. Never accepts on magic bytes alone -- a truncated
    BigTIFF still starts with `II+\\0`."""
    tmp_path = dest_path.parent / (dest_path.name + ".part")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    r = session.get(url, timeout=(30, 300), stream=True)
    r.raise_for_status()
    content_length_hdr = r.headers.get("content-length")
    content_length = int(content_length_hdr) if content_length_hdr is not None else None

    written = 0
    head_buf = bytearray()
    with tmp_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            if len(head_buf) < 512:
                head_buf.extend(chunk[: 512 - len(head_buf)])
            f.write(chunk)
            written += len(chunk)

    try:
        if kind == "tiff":
            assert_magic(bytes(head_buf), "tiff", url=url)
        else:
            assert_not_html(bytes(head_buf), url=url)
    except ValueError as e:
        # Deliberately NOT deleted -- left as `<name>.part` for diagnosis, same
        # as a truncation failure below. A real re-run will overwrite it.
        raise AssetEntitlementError(str(e)) from e

    if content_length is not None and written != content_length:
        raise TruncatedDownloadError(
            f"{dest_path.name}: wrote {written} B but Content-Length declared "
            f"{content_length} B -- refusing to accept a truncated download "
            f"(left as {tmp_path.name})")

    tmp_path.rename(dest_path)
    return dict(url=url, filename=dest_path.name, bytes=dest_path.stat().st_size,
                sha256=_sha256(dest_path), status="downloaded",
                content_length_header=content_length)


def merge_manifest(manifest_path: Path, new_records: list[dict]) -> dict:
    """Merge `new_records` into the manifest at `manifest_path` by product id,
    writing the result and returning it. A record already on disk survives a
    run that never touches its id; within one record, `assets` is ALSO
    soft-merged per-key so a `--reconcile` run that only sees image+metadata
    locally does not erase a prior download run's quality-mask entries (or
    the reverse)."""
    existing_by_id: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text())
            for p in prior.get("products", []):
                pid = p.get("id")
                if pid:
                    existing_by_id[pid] = p
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read existing manifest {manifest_path} "
                  f"({type(e).__name__}: {e}) -- starting a fresh one rather than "
                  "silently dropping it; back it up if it mattered", file=sys.stderr)

    for rec in new_records:
        pid = rec["id"]
        if pid in existing_by_id:
            merged_assets = {**existing_by_id[pid].get("assets", {}), **rec.get("assets", {})}
            merged = {**existing_by_id[pid], **rec, "assets": merged_assets}
        else:
            merged = rec
        existing_by_id[pid] = merged

    merged_records = sorted(existing_by_id.values(), key=lambda r: r.get("date") or "")
    total_bytes = sum(
        sum((a.get("bytes") or 0) for a in r.get("assets", {}).values())
        for r in merged_records
    )
    manifest = dict(n_products=len(merged_records), total_local_bytes=total_bytes,
                     products=merged_records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _asset_record_from_path(path: Path, url: str | None, *, status: str) -> dict:
    return dict(url=url, filename=path.name, bytes=path.stat().st_size,
                sha256=_sha256(path), status=status)


def _run_reconcile(out_dir: Path, manifest_path: Path) -> int:
    """Index *-SPECTRAL_IMAGE_COG.TIF products already on disk into the
    manifest, without downloading anything. Looks up each scene's STAC record
    by id (no credentials needed) for catalogue fields and true asset URLs;
    if that lookup fails (404, or a network problem), the scene is STILL
    recorded -- bytes/sha256/filenames come from disk regardless, only the
    catalogue fields go null. Skipping on a failed lookup would defeat the
    purpose: the scenes most likely to 404 (reprocessed/superseded/withdrawn)
    are exactly the oldest hand-fetched ones this mode exists to capture."""
    spectral_files = sorted(out_dir.glob(f"*{SPECTRAL_SUFFIX}"))
    if not spectral_files:
        print(f"No *{SPECTRAL_SUFFIX} files in {out_dir} to reconcile", file=sys.stderr)
        return 1

    now = _utcnow_iso()
    records: list[dict] = []
    for sp in spectral_files:
        scene_id = sp.name[: -len(SPECTRAL_SUFFIX)]
        local_paths: dict[str, Path] = {"image": sp}
        meta_path = _existing_metadata_path(sp)
        if meta_path is not None:
            local_paths["metadata"] = meta_path
        else:
            print(f"  {scene_id}: WARNING no METADATA.XML(.XML) sidecar found on disk")
        for key, suffix in (("quality_classes", "-QL_QUALITY_CLASSES_COG.TIF"),
                             ("quality_cloud", "-QL_QUALITY_CLOUD_COG.TIF"),
                             ("defective_pixel_mask", "-QL_PIXELMASK_COG.TIF")):
            p = out_dir / f"{scene_id}{suffix}"
            if p.exists() and p.stat().st_size > 0:
                local_paths[key] = p

        catalog = None
        try:
            r = requests.get(STAC_ITEM_URL.format(id=scene_id), timeout=30)
            if r.status_code == 200:
                catalog = r.json()
            elif r.status_code == 404:
                print(f"  {scene_id}: STAC lookup 404 (superseded/withdrawn/renamed?) "
                      "-- recording local facts only, catalogue fields null")
            else:
                print(f"  {scene_id}: STAC lookup HTTP {r.status_code} -- "
                      "recording local facts only, catalogue fields null")
        except requests.exceptions.RequestException as e:
            print(f"  {scene_id}: STAC lookup failed ({type(e).__name__}: {e}) -- "
                  "recording local facts only, catalogue fields null")

        asset_urls = (catalog or {}).get("assets", {})
        assets_out = {
            key: _asset_record_from_path(
                path, asset_urls.get(key, {}).get("href"), status="reconciled")
            for key, path in local_paths.items()
        }
        props = (catalog or {}).get("properties", {})
        rec = dict(
            id=scene_id,
            date=(props.get("datetime") or "")[:19] or None,
            # Reuses find_enmap_scenes._num: DLR's STAC API serializes these
            # fields inconsistently (int on some features, numeric string on
            # others -- verified 2026-08-23, see find_enmap_scenes.search).
            # This reconcile path reads the raw item directly rather than
            # through search(), so it needs the same coercion independently.
            cloud=_num(props.get("eo:cloud_cover"), default=None),
            snow=_num(props.get("eo:snow_cover"), default=None),
            epsg=props.get("proj:epsg"),
            datatake=_datatake(scene_id),
            assets=assets_out,
            reconciled_utc=now,
            search_params=None,
            catalog_lookup_ok=catalog is not None,
        )
        records.append(rec)
        print(f"  {scene_id}: {len(assets_out)} local asset(s) indexed, "
              f"catalog_lookup_ok={catalog is not None}")

    manifest = merge_manifest(manifest_path, records)
    print(f"\nReconciled {len(records)} scene(s) from {out_dir}.")
    print(f"Manifest now holds {manifest['n_products']} product(s), "
          f"{_human_bytes(manifest['total_local_bytes'])} total.")
    print(f"Manifest: {manifest_path}")
    print("Run `.venv/bin/python scripts/verify_enmap.py` to validate the local files.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", default="68,6,98,36", help="minlon,minlat,maxlon,maxlat (default: India)")
    ap.add_argument("--start", help="ISO date, e.g. 2026-01-01")
    ap.add_argument("--end", help="ISO date")
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument("--per-datatake", type=int, default=1,
                     help="max scenes from one overpass (default 1; near-duplicates otherwise)")
    ap.add_argument("--search-limit", type=int, default=40,
                     help="max STAC candidates to consider (default 40)")
    ap.add_argument("--limit", type=int, default=5,
                     help="max NEW scenes to download this run (default 5; bounded on "
                          "purpose -- each cube is ~400-500 MB). Scenes already complete "
                          "on disk do not count against this.")
    ap.add_argument("--with-quality-masks", action="store_true",
                     help="also fetch quality_classes/quality_cloud/defective_pixel_mask "
                          "(opt-in; default fetches only the cube + metadata sidecar)")
    ap.add_argument("--out-dir", default=None, help=f"default: {DEFAULT_OUT_DIR}")
    ap.add_argument("--manifest", default=None, help=f"default: {DEFAULT_MANIFEST}")
    ap.add_argument("--yes", action="store_true", help="skip the download confirmation prompt")
    ap.add_argument("--force", action="store_true", help="re-download even if already present")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                     help="refuse to start if free disk space after the projected "
                          "download would fall below this many GB (default 5)")
    ap.add_argument("--reconcile", action="store_true",
                     help="index *-SPECTRAL_IMAGE_COG.TIF products already on disk into "
                          "the manifest, without downloading anything")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else DEFAULT_OUT_DIR
    manifest_path = Path(a.manifest) if a.manifest else DEFAULT_MANIFEST
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.reconcile:
        return _run_reconcile(out_dir, manifest_path)

    # (a) missing credential variable -- fails before any network call, names
    # the variable, never a value.
    try:
        creds = credentials.require("dlr")
    except RuntimeError as e:
        print(f"CREDENTIAL ERROR: {e}", file=sys.stderr)
        return 2

    assets_wanted = list(CORE_ASSETS) + (list(QUALITY_ASSETS) if a.with_quality_masks else [])
    print(f"Searching DLR STAC ({COLLECTION}) bbox={a.bbox} cloud<={a.max_cloud}% "
          f"assets={assets_wanted} ...")
    candidates = stac_search(a.bbox, a.start, a.end, a.max_cloud, a.search_limit,
                              a.per_datatake, tuple(assets_wanted))
    if not candidates:
        print(f"No scenes under {a.max_cloud}% cloud in bbox {a.bbox}. Widen the search.",
              file=sys.stderr)
        return 1
    print(f"{len(candidates)} candidate scene(s) from the catalogue search.\n")

    search_params = dict(bbox=a.bbox, start=a.start, end=a.end, max_cloud=a.max_cloud,
                          per_datatake=a.per_datatake, assets=assets_wanted)

    cached_records: list[dict] = []
    to_fetch: list[tuple[dict, dict[str, str]]] = []
    new_count = 0
    for scene in candidates:
        missing: dict[str, str] = {}
        present: dict[str, str] = {}
        for key in assets_wanted:
            url = scene["assets"].get(key)
            if url is None:
                print(f"  WARNING {scene['id']}: asset key {key!r} missing from this "
                      "scene's STAC response -- skipping that asset for this scene")
                continue
            if not a.force and _asset_is_present(out_dir, scene["id"], key, url):
                present[key] = url
            else:
                missing[key] = url

        if not missing:
            asset_results = {
                key: _asset_record_from_path(
                    _local_path_for(out_dir, scene["id"], key, url), url, status="cached")
                for key, url in present.items()
            }
            cached_records.append(dict(
                id=scene["id"], date=scene.get("date"), cloud=scene.get("cloud"),
                snow=scene.get("snow"), epsg=scene.get("epsg"),
                declared_dtype=scene.get("declared_dtype"),
                sun_elevation=scene.get("sun_elevation"), datatake=scene.get("datatake"),
                assets=asset_results, fetched_utc=_utcnow_iso(), search_params=search_params,
            ))
            continue

        if new_count >= a.limit:
            print(f"  SKIP {scene['id'][:45]:45} (--limit={a.limit} reached; "
                  f"{len(missing)} new asset(s) would be fetched)")
            continue
        new_count += 1
        to_fetch.append((scene, {"missing": missing, "present": present}))

    if cached_records:
        print(f"{len(cached_records)} scene(s) already complete on disk (recorded, "
              "not re-downloaded).")

    if not to_fetch:
        manifest = merge_manifest(manifest_path, cached_records)
        print(f"\nNothing new to download. Manifest holds {manifest['n_products']} "
              f"product(s), {_human_bytes(manifest['total_local_bytes'])}.")
        print(f"Manifest: {manifest_path}")
        return 0

    # (b) credentials present but rejected / account not activated for this
    # CAS service -- the bare login (no service named) carries no ticket.
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) sih-fetch-enmap/1.0"
    print(f"CAS login ({CAS}) ...", end=" ", flush=True)
    code, tickets = _cas_login(session, CAS, creds)
    print(f"HTTP {code}  tickets={tickets or '-'}")
    if not tickets:
        print("FAIL -- the credentials themselves are rejected, or the account is not "
              "activated. Fix DLR_USERNAME/DLR_PASSWORD, or finish account activation "
              "(docs/enmap_handover.md sections 1-3). Nothing was downloaded.",
              file=sys.stderr)
        return 3

    # Projected size, via HEAD in the SAME authenticated session used for the
    # real GETs -- printed and confirmed before any real bytes move.
    print(f"\nSizing {sum(len(m['missing']) for _, m in to_fetch)} new asset(s) "
          f"across {len(to_fetch)} scene(s) ...")
    per_scene_bytes: dict[str, int] = {}
    unsized: list[str] = []
    total_bytes = 0
    for scene, m in to_fetch:
        sb = 0
        for key, url in m["missing"].items():
            try:
                hr = session.head(url, timeout=60, allow_redirects=True)
                n = int(hr.headers["content-length"]) if hr.status_code == 200 and \
                    "content-length" in hr.headers else None
            except (requests.exceptions.RequestException, KeyError, ValueError):
                n = None
            if n is None:
                unsized.append(f"{scene['id']}:{key}")
            else:
                sb += n
        per_scene_bytes[scene["id"]] = sb
        total_bytes += sb
        print(f"  {scene['id'][:45]:45} cloud={scene.get('cloud')!s:>5}  "
              f"{len(m['missing'])} asset(s)  ~{_human_bytes(sb)}")

    print(f"\nProjected total: ~{_human_bytes(total_bytes)} across {len(to_fetch)} scene(s)"
          + (f"  ({len(unsized)} asset(s) had no Content-Length -- excluded from total)"
             if unsized else ""))

    free_bytes = shutil.disk_usage(out_dir).free
    headroom_needed = a.min_free_gb * 1e9
    if free_bytes - total_bytes < headroom_needed:
        print(f"REFUSING: {_human_bytes(free_bytes)} free at {out_dir}, projected download "
              f"~{_human_bytes(total_bytes)} would leave less than --min-free-gb="
              f"{a.min_free_gb} GB free. Free up space or lower --limit.", file=sys.stderr)
        return 1

    if not a.yes:
        if not sys.stdin.isatty():
            print("Refusing to download without --yes in a non-interactive session.",
                  file=sys.stderr)
            return 1
        resp = input(f"Proceed downloading ~{_human_bytes(total_bytes)} "
                      f"({len(to_fetch)} scene(s))? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("Aborted, nothing downloaded.")
            return 1

    records: list[dict] = []
    run_had_failure = False
    for scene, m in to_fetch:
        print(f"\n{scene['id']}")
        asset_results: dict[str, dict] = {
            key: _asset_record_from_path(
                _local_path_for(out_dir, scene["id"], key, url), url, status="cached")
            for key, url in m["present"].items()
        }
        for key, url in m["missing"].items():
            dest_path = out_dir / _asset_filename(url)
            kind = "tiff" if key != "metadata" else "xml"
            print(f"  [{key}] GET {url.rsplit('/', 1)[-1][:60]} ...", end=" ", flush=True)
            try:
                res = _download_asset(session, url, dest_path, kind)
                print(f"OK  {_human_bytes(res['bytes'])}  sha256={res['sha256'][:12]}...")
            except AssetEntitlementError as e:
                print("FAIL (entitlement)")
                print(f"    Authenticated, but this asset did not return real data: {e}")
                print("    This is an ENTITLEMENT problem, not a credential one -- see "
                      "verify_access.py::dlr() / PLAN.md O11. The account may need the "
                      "EnMAP Access Service role even though it logs in fine.")
                res = dict(url=url, filename=None, bytes=None, sha256=None,
                           status="failed_entitlement", error=str(e))
                run_had_failure = True
            except TruncatedDownloadError as e:
                print("FAIL (truncated)")
                print(f"    {e}")
                res = dict(url=url, filename=None, bytes=None, sha256=None,
                           status="failed_truncated", error=str(e))
                run_had_failure = True
            except requests.exceptions.RequestException as e:
                print("FAIL (network)")
                print(f"    {type(e).__name__}: {e}")
                res = dict(url=url, filename=None, bytes=None, sha256=None,
                           status="failed_network", error=f"{type(e).__name__}: {e}")
                run_had_failure = True
            asset_results[key] = res

        records.append(dict(
            id=scene["id"], date=scene.get("date"), cloud=scene.get("cloud"),
            snow=scene.get("snow"), epsg=scene.get("epsg"),
            declared_dtype=scene.get("declared_dtype"),
            sun_elevation=scene.get("sun_elevation"), datatake=scene.get("datatake"),
            assets=asset_results, fetched_utc=_utcnow_iso(), search_params=search_params,
        ))

    manifest = merge_manifest(manifest_path, records + cached_records)
    downloaded_bytes = sum(
        (a_["bytes"] or 0) for r in records for a_ in r["assets"].values()
        if a_["status"] == "downloaded")
    print(f"\n{len(to_fetch)} scene(s) attempted this run, "
          f"{_human_bytes(downloaded_bytes)} newly downloaded; manifest holds "
          f"{manifest['n_products']} product(s), {_human_bytes(manifest['total_local_bytes'])} total.")
    print(f"Manifest: {manifest_path}")
    print("Run `.venv/bin/python scripts/verify_enmap.py` to validate all local files.")
    return 1 if run_had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
