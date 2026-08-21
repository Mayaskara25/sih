#!/usr/bin/env python3
"""Target endmember spectra for D7 implantation (PLAN.md §1.6, §3B.1).

WHY THIS FETCHER CANNOT FETCH, AND WHY THAT IS A FINDING RATHER THAN A BUG
=========================================================================
`SPECTRA_POOLS["lib"] = ("usgs_splib07", "ecostress_aster")`. Both halves were
probed against their live authoritative endpoints on 2026-08-22, and **neither
serves bytes to an unattended client.** See PLAN.md D21 for the full record.

  usgs_splib07     DOI 10.5066/F7RR1WDJ resolves to ScienceBase item
                   5807a2a2e4b0841e59e3a18d. `usgs_splib07.zip` is 5 479 324 354 B
                   with `"pathOnDisk": "__s3__"`, `"published": false`, and an
                   `s3DownloadRequestPageUri`. The two URLs the catalogue
                   advertises for it (`.../manager/item/.../file/<cuid>` and
                   `.../manager/download/<cuid>`) both return an **HTML page**.
                   The second returns it as **HTTP 206 with Content-Type
                   text/html** in response to a Range request -- a range read
                   that "succeeds" while delivering a web page. Any naive
                   downloader writes 5 GB of HTML and reports success.

  ecostress_aster  https://speclib.jpl.nasa.gov/download is a "Request Download"
                   form. There is no unauthenticated direct-download URL for the
                   bulk library.

So this is the EnMAP situation again (§8.0a, O11): search and metadata are open,
retrieval is human-gated. Consistent with CLAUDE.md rule 4 -- when an input is
unavailable, stop and say which one, rather than inventing a value or silently
substituting an unauthenticated endpoint.

DO NOT "FIX" THIS BY POINTING IT AT A GITHUB MIRROR.
Several third-party repos redistribute ECOSTRESS/ASTER subsets. They are not
authoritative, they carry no checksum we can verify against the publisher, and
§1.6 requires each fetcher to write a provenance record (URL, date, size,
license, citation). A mirror satisfies the code path and fails the standard --
the same trade this project has already refused twice (D11's HAD100 archive,
D13's ABU/HYDICE claims). The verification standard in CLAUDE.md is explicit:
assume documentation is wrong until checked against the files.

WHAT THIS SCRIPT DOES
=====================
  --check    resolve the DOI, report what the catalogue says, and confirm
             whether retrieval is still gated. Run this to re-test D21; if a
             direct download appears, D21 is stale and should be reopened.
  --ingest   parse an archive a human has already placed on disk and emit
             target spectra on the 184-band canonical grid.

`--ingest` deliberately does NOT ship a parser for the splib07 ASCII layout.
Nobody on this project has opened one of those files. Writing a parser against
a documented-but-unverified format is exactly the failure mode D11 recorded
(HAD100's project page was wrong about its own archive in five separate ways)
and D13 recorded again (ABU and HYDICE, three more). The parser gets written
when the archive exists and can be read, not before.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "speclib"

SPLIB07_DOI = "10.5066/F7RR1WDJ"
SPLIB07_SB_ITEM = "5807a2a2e4b0841e59e3a18d"
SPLIB07_SB_JSON = f"https://www.sciencebase.gov/catalog/item/{SPLIB07_SB_ITEM}?format=json"
ECOSTRESS_DOWNLOAD_PAGE = "https://speclib.jpl.nasa.gov/download"

CITATIONS = {
    "usgs_splib07": (
        "Kokaly, R.F., Clark, R.N., Swayze, G.A., Livo, K.E., Hoefen, T.M., "
        "Pearson, N.C., et al., 2017, USGS Spectral Library Version 7: "
        "U.S. Geological Survey Data Series 1035, doi:10.3133/ds1035. "
        "Data: doi:10.5066/F7RR1WDJ."
    ),
    "ecostress_aster": (
        "Meerdink, S.K., Hook, S.J., Roberts, D.A., Abbott, E.A., 2019, "
        "The ECOSTRESS spectral library version 1.0: Remote Sensing of "
        "Environment, v. 230, 111196. Supersedes the ASTER Spectral Library."
    ),
}

HUMAN_INSTRUCTIONS = f"""\
A human with a browser must retrieve these once. Neither needs a paid account.

  1. USGS splib07 (5.48 GB)
     Open  https://www.sciencebase.gov/catalog/item/{SPLIB07_SB_ITEM}
     Use the download-request link for `usgs_splib07.zip` ("__s3__" staged
     files are delivered through a request page, not a direct link).
     Save the archive to  {DEST}/usgs_splib07.zip

  2. ECOSTRESS / ASTER
     Open  {ECOSTRESS_DOWNLOAD_PAGE}
     Request a category (Minerals, Rock, Soil, Man-made are the relevant ones
     for the `object` profile; Vegetation and Non-Photosynthetic Vegetation for
     `landcover`), or the complete 6 139-file set.
     Save the archive to  {DEST}/ecostress/

Then re-run:  .venv/bin/python scripts/fetch_speclib.py --ingest

Do NOT paste any credential into this process. These are open datasets; if a
site asks you to log in, you are on the wrong page.
"""


def _get_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"User-Agent": "sih-fetch-speclib/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check() -> int:
    """Re-test D21. Exits 0 if still gated (expected), 2 if a direct download appeared."""
    print(f"USGS splib07  DOI {SPLIB07_DOI}")
    try:
        item = _get_json(SPLIB07_SB_JSON)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  ScienceBase unreachable: {exc}")
        return 1

    print(f"  item     : {item.get('title')}")
    gated = True
    for f in item.get("files", []):
        if not f.get("name", "").endswith(".zip"):
            continue
        path_on_disk = f.get("pathOnDisk")
        print(f"  archive  : {f.get('name')}  {f.get('size'):,} B")
        print(f"  pathOnDisk: {path_on_disk!r}   published: {f.get('published')!r}")
        print(f"  checksum : {f.get('checksum')!r}")
        if path_on_disk != "__s3__" or f.get("s3DownloadRequestPageUri") is None:
            print("  *** pathOnDisk is no longer '__s3__' -- retrieval may now be direct.")
            print("  *** D21 may be stale. Re-probe the download URI before trusting it,")
            print("  *** and assert_not_html() the first bytes (D21: the advertised URI")
            print("  *** answers a Range request with HTTP 206 text/html).")
            gated = False

    print(f"\nECOSTRESS/ASTER  {ECOSTRESS_DOWNLOAD_PAGE}")
    print("  request-form only; no unauthenticated direct-download URL (D21).")

    if gated:
        print("\nRESULT: both sources still human-gated. D21 stands.")
        print(HUMAN_INSTRUCTIONS)
        return 0
    return 2


def ingest() -> int:
    have_splib = (DEST / "usgs_splib07.zip").exists()
    have_eco = (DEST / "ecostress").exists()
    if not (have_splib or have_eco):
        print(f"Nothing to ingest. Expected an archive under {DEST}\n", file=sys.stderr)
        print(HUMAN_INSTRUCTIONS, file=sys.stderr)
        return 1

    raise NotImplementedError(
        "An archive is present, so the parser can now be written -- but it must be "
        "written against the actual files, not against the format documentation.\n\n"
        "Required before target spectra are used for implantation (§3B.1):\n"
        "  1. Read a real record and pin its layout in a test fixture, the way\n"
        "     tests/test_benchmarks.py pins ABU's per-scene band counts.\n"
        "  2. Confirm the wavelength axis is ASCENDING before any np.interp call.\n"
        "     AVIRIS-Classic's was not (D11.4), np.interp neither requires nor\n"
        "     checks it, and the failure is silent. Reuse\n"
        "     preprocessing.harmonize.sort_spectral_axis -- do not hand-roll it.\n"
        "  3. Confirm the spectra are REFLECTANCE, not radiance. §3B.1 implants by\n"
        "     linear mixing m = a*t + (1-a)*s against HAD100 backgrounds, which D19\n"
        "     measured at radiance scale (components ranging roughly -6500..+8).\n"
        "     Mixing a [0,1] reflectance endmember into a radiance-scale background\n"
        "     produces an implant that is invisible at every abundance, and the\n"
        "     abundance sweep (§3B.8) would flatline for a reason that looks like a\n"
        "     model failure rather than a units error.\n"
        "  4. Run coverage_ok() before harmonize(). splib07 spectra span 0.2-3.0 um\n"
        "     and the canonical grid is 400-2500 nm, so coverage should be complete\n"
        "     -- but D16 found EnMAP L2A failing exactly this check after its\n"
        "     documentation implied otherwise. Check, do not assume.\n"
        "  5. Emit [K, RETAINED_BANDS] float32 plus a per-spectrum provenance tag,\n"
        "     which is what segmentation.synth.load_target_spectra returns, and\n"
        "     write the §1.6 provenance record into docs/datasets.md.\n\n"
        f"Citations to carry into that record:\n"
        f"  usgs_splib07   : {CITATIONS['usgs_splib07']}\n"
        f"  ecostress_aster: {CITATIONS['ecostress_aster']}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="re-test whether retrieval is still human-gated (D21)")
    g.add_argument("--ingest", action="store_true",
                   help="parse an archive a human has already placed on disk")
    args = ap.parse_args(argv)
    DEST.mkdir(parents=True, exist_ok=True)
    return check() if args.check else ingest()


if __name__ == "__main__":
    raise SystemExit(main())
