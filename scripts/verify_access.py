#!/usr/bin/env python3
"""Prove that the configured credentials can actually RETRIEVE data.

`check_credentials.py` answers "is a value present". This answers the question
that matters: does the value work, and does the wall actually open. It is the
discharge test for PLAN.md O11 (EnMAP entitlement) and the first action of 8.0.

Prints hostnames, status codes and byte counts. Never prints a credential.
Exit 0 = every configured service returned real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from urllib.parse import urljoin

import requests

from core import credentials
from core.http_guard import assert_magic
from core import cdse_s3

STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A/items?limit=1"
CAS = "https://sso.eoc.dlr.de/eoc/auth/login"

CDSE_CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
# Any recent, real Sentinel-2 L2A product over India works -- resolved live rather
# than pinned to one scene ID, so this probe never goes stale as CDSE's archive moves.
CDSE_PROBE_FILTER = (
    "Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL2A') and "
    "OData.CSC.Intersects(area=geography'SRID=4326;POINT(77.61 28.17)')"
)


def _login_form(html: str) -> tuple[str, dict[str, str]]:
    """Return (action, fields) for the form that actually carries the password.

    The EOC login page serves TWO post forms: the credential form
    (username/password/execution/submitBtn, no _csrf) and a separate SSO-button
    form (_csrf/client_name/submitButton) with its OWN, different `execution`
    token. Scraping hidden inputs with a page-wide regex silently splices the
    two and CAS answers 401 -- which reads exactly like a bad password.
    Always scope the scrape to the form containing `name="password"`.
    """
    for m in re.finditer(r"(?is)<form\b(.*?)</form>", html):
        body = m.group(1)
        if 'name="password"' not in body:
            continue
        attrs = re.match(r"(?is)([^>]*)>", body).group(1)
        action = (re.search(r'action="([^"]*)"', attrs) or (None, ""))[1]
        fields = {}
        for inp in re.finditer(r"(?is)<(?:input|button)\b([^>]*)>", body):
            a = inp.group(1)
            n = re.search(r'name="([^"]+)"', a)
            if not n:
                continue
            v = re.search(r'value="([^"]*)"', a)
            fields[n.group(1)] = v.group(1) if v else ""
        return action, fields
    return "", {}


def _cas_login(s: requests.Session, entry_url: str, creds: dict) -> tuple[int, list[str]]:
    """Drive the CAS credential form at `entry_url`. Returns (status, ticket_cookies)."""
    land = s.get(entry_url, timeout=60)
    action, fields = _login_form(land.text)
    if not fields.get("execution"):
        raise RuntimeError("no credential form at " + entry_url)
    fields["username"] = creds["DLR_USERNAME"]
    fields["password"] = creds["DLR_PASSWORD"]
    fields.setdefault("_eventId", "submit")
    r = s.post(urljoin(land.url, action), data=fields, timeout=90,
               headers={"Referer": land.url, "Origin": "https://sso.eoc.dlr.de"})
    tickets = [k for k in s.cookies.keys()
               if k.upper().startswith(("TGC", "JSESSIONID", "CAS"))]
    return r.status_code, tickets


def dlr() -> bool:
    """Separate three outcomes that all look alike from the outside.

    CAS authorises PER SERVICE. A login naming a service the account may not use
    returns 401 -- byte-identical in effect to a wrong password. So test the
    credentials first with no service at all, and only then with the EnMAP
    download service. Verified 2026-08-21: bare login 200 + TGC, EnMAP login 401.
    """
    creds = credentials.require("dlr")          # values never printed
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) sih-verify/1.0"

    print("  STAC search (no auth) ...", end=" ", flush=True)
    r = s.get(STAC, timeout=60)
    item = r.json()["features"][0]
    print(f"HTTP {r.status_code}  item {item['id'][:38]}…")

    print("  CAS login, no service ...", end=" ", flush=True)
    code, tickets = _cas_login(s, CAS, creds)
    print(f"HTTP {code}  tickets={tickets or '-'}")
    if not tickets:
        print("  FAIL — the credentials themselves are rejected.")
        print("  Fix DLR_USERNAME / DLR_PASSWORD, or finish account activation.")
        print("  The page reports its reason in JavaScript; check " + CAS)
        return False
    print("  → credentials and activation are GOOD.")

    # Probe a small QUALITY mask, not `image` (a full ~1 GB cube) and not
    # `thumbnail` (which is JPEG, despite sitting beside a dozen GeoTIFFs).
    key = "quality_cloud" if "quality_cloud" in item["assets"] else "vnir"
    url = item["assets"][key]["href"]

    print(f"  asset GET [{key}] ...", end=" ", flush=True)
    r = s.get(url, timeout=180)
    print(f"HTTP {r.status_code}  {len(r.content)} B  {r.headers.get('content-type','?')}")
    try:
        assert_magic(r.content, "tiff", url=url)
    except ValueError:
        print("  FAIL — authenticated, but the EnMAP asset did not return data.")
        print("  This is O11 and it is an ENTITLEMENT problem, not a credential one:")
        print("  the same account logs in fine with no service named, and DLR's own")
        print("  403 body says 'insufficient privileges to download this dataset'.")
        print("  Remedy: EnMAP archive access needs a role assignment via the")
        print("  Instrument Planning Portal (planning.enmap.org — unreachable as of")
        print("  2026-08-21). Contact erdbeobachtung@dlr.de. Meanwhile Phase 5 L2")
        print("  falls back to AVIRIS-NG for the whole background pool (PLAN.md O11).")
        return False
    print("  PASS — real TIFF bytes. O11 discharged.")
    return True


def cdse() -> bool:
    """CDSE (Sentinel-2), verified live 2026-08-23. Distinguishes the three
    outcomes O10 warns get confused for each other:

      (a) missing variable    -- credentials.require() raises before any network
                                  call; the caller's except block reports it and
                                  names the variable, never a value.
      (b) present but REJECTED (expired/revoked/wrong key) -- the S3 endpoint
                                  answers HTTP 403 with an S3-style XML
                                  <Error><Code>...</Code></Error> body
                                  (InvalidAccessKeyId / SignatureDoesNotMatch /
                                  AccessDenied / ExpiredToken all land here).
                                  O10: a caller-chosen S3 key expiry means this
                                  looks, from a glance at 'it failed', exactly
                                  like (c) -- the XML body is what tells them
                                  apart, not the fact that something failed.
      (c) genuine service/network problem -- a connection error, timeout, or
                                  HTTP 5xx from the S3 endpoint itself.
    """
    print("  catalogue search (no auth) ...", end=" ", flush=True)
    r = requests.get(CDSE_CATALOGUE, params={"$filter": CDSE_PROBE_FILTER, "$top": 1,
                                             "$orderby": "ContentDate/Start desc"},
                      timeout=60)
    r.raise_for_status()
    items = r.json().get("value", [])
    if not items:
        print("no products found — cannot probe download leg")
        return False
    prod = items[0]
    s3path = prod["S3Path"].lstrip("/")          # "eodata/Sentinel-2/MSI/L2A/..."
    key = s3path[len(cdse_s3.BUCKET) + 1:]        # strip the leading "eodata/"
    print(f"HTTP {r.status_code}  {prod['Name'][:45]}…")

    try:
        creds = credentials.require("cdse")       # (a) — raises + names the variable
    except RuntimeError as e:
        print(f"  FAIL (a) missing variable: {e}")
        return False
    del creds  # never touched again; require() above is only to fail fast and clearly

    # Smallest reliably-present file in the product: the manifest.
    print(f"  S3 GET [manifest.safe] ...", end=" ", flush=True)
    try:
        resp = cdse_s3.sigv4_get(f"{key}/manifest.safe")
    except requests.exceptions.RequestException as e:
        print(f"FAIL (c) service/network problem: {type(e).__name__}: {e}")
        return False

    print(f"HTTP {resp.status_code}  {len(resp.content)} B")
    if resp.status_code in (401, 403):
        print(f"  FAIL (b) present but REJECTED by the S3 endpoint:")
        print(f"    {resp.text[:300]}")
        print("  This is an S3 key problem (expired/revoked/wrong), not a service")
        print("  outage — regenerate CDSE_S3_ACCESS_KEY/CDSE_S3_SECRET_KEY at")
        print("  https://eodata-s3keysmanager.dataspace.copernicus.eu/")
        return False
    if resp.status_code >= 500:
        print(f"  FAIL (c) genuine service problem: HTTP {resp.status_code}")
        return False
    if resp.status_code != 200:
        print(f"  FAIL unexpected status {resp.status_code}: {resp.text[:300]}")
        return False
    if not resp.content.startswith(b"<?xml"):
        print("  FAIL — HTTP 200 but payload is not the expected XML manifest "
              f"(leading bytes {resp.content[:16]!r}) — see core/http_guard.py")
        return False
    print("  PASS — real, authenticated S3 bytes. O10 discharged for this key.")

    # Also prove the windowed-raster path (/vsis3/ + rasterio), the one the
    # fetcher actually uses for band data, not just the small-file GET above.
    # The granule folder name and band filenames are not derivable from the
    # product name, so discover them via the unauthenticated OData Nodes
    # browsing API (verified separately to need no credentials at all) rather
    # than guessing a path.
    import rasterio

    prod_id, name = prod["Id"], prod["Name"]
    nbase = f"{CDSE_CATALOGUE}({prod_id})/Nodes({name})/Nodes(GRANULE)/Nodes"
    gran = requests.get(nbase, timeout=60).json()["result"][0]["Name"]
    band_nodes = requests.get(
        f"{CDSE_CATALOGUE}({prod_id})/Nodes({name})/Nodes(GRANULE)/Nodes({gran})"
        f"/Nodes(IMG_DATA)/Nodes(R60m)/Nodes", timeout=60).json()["result"]
    band_file = next(n["Name"] for n in band_nodes if "_B01_" in n["Name"])
    band_key = f"{key}/GRANULE/{gran}/IMG_DATA/R60m/{band_file}"

    print(f"  /vsis3/ windowed raster open [{band_file}] ...", end=" ", flush=True)
    with cdse_s3.s3_env():
        with rasterio.open(cdse_s3.vsis3_path(band_key)) as ds:
            win = rasterio.windows.Window(0, 0, 16, 16)
            arr = ds.read(1, window=win)
    print(f"OK  {ds.width}x{ds.height} {ds.dtypes[0]} {ds.crs}  "
          f"window read {arr.shape} min/max=[{int(arr.min())},{int(arr.max())}]")
    return True


if __name__ == "__main__":
    st = credentials.status()
    rc = 0
    for svc, fn in (("dlr", dlr), ("cdse", cdse)):
        if not st.get(svc):
            print(f"{svc}: not configured, skipped"); continue
        print(f"{svc}:")
        try:
            if not fn():
                rc = 1
        except Exception as e:                       # noqa: BLE001
            print(f"  ERROR {type(e).__name__}: {e}"); rc = 1
    raise SystemExit(rc)
