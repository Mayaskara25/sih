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

STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A/items?limit=1"
CAS = "https://sso.eoc.dlr.de/eoc/auth/login"


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


if __name__ == "__main__":
    st = credentials.status()
    rc = 0
    for svc, fn in (("dlr", dlr),):
        if not st.get(svc):
            print(f"{svc}: not configured, skipped"); continue
        print(f"{svc}:")
        try:
            if not fn():
                rc = 1
        except Exception as e:                       # noqa: BLE001
            print(f"  ERROR {type(e).__name__}: {e}"); rc = 1
    raise SystemExit(rc)
