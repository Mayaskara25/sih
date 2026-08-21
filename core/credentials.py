"""Credential loading. Values are read; they are never logged, printed or returned
in an exception message.

WHERE SECRETS LIVE
    ~/.config/sih/credentials.env      (chmod 600, OUTSIDE the repository)

Deliberately not a `.env` inside the project. `.gitignore` stops `git add`; it does
not stop an agent globbing the working tree, a `tar` of the project folder, a
`git add -f`, or a stray editor backup. Keeping the file outside the repo removes
the whole class. `.env.example` in the repo documents the variable NAMES only.

Precedence: real environment variables win over the file, so CI can inject secrets
without a file existing at all.
"""
from __future__ import annotations
import os
from pathlib import Path

CRED_FILE = Path(os.environ.get("SIH_CREDENTIALS_FILE",
                                Path.home() / ".config" / "sih" / "credentials.env"))

_SERVICES = {
    "cdse": {
        # S3 keys only. OData *search* needs no credentials at all (verified: HTTP 200
        # unauthenticated); only the download leg is protected, and S3 covers it without
        # ever storing the account password. See PLAN.md 4.1b.
        "primary": ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY"),
        "how": ("Generate S3 keys at https://eodata-s3keysmanager.dataspace.copernicus.eu/ "
                "-> Add Credentials. The secret is shown ONCE. Endpoint: "
                "https://eodata.dataspace.copernicus.eu"),
    },
    "dlr": {
        # EnMAP L2A. Verified 2026-08-21 against the live service:
        #   STAC search  geoservice.dlr.de/eoc/ogc/stac/v1  -> 200, NO auth
        #   asset GET    download.geoservice.dlr.de/ENMAP/  -> EOC UMS SSO (CAS)
        # CAS is a username/password ticket flow -- there is no API token, so the
        # old EOWEB_TOKEN variable was removed rather than left unfillable.
        "primary": ("DLR_USERNAME", "DLR_PASSWORD"),
        "how": ("Register at "
                "https://sso.eoc.dlr.de/geoservice/selfservice/public/newuser?locale=en "
                "then confirm by email. EnMAP L2A download entitlement may need a separate "
                "role assignment -- see PLAN.md O11."),
    },
}


def _load_file() -> dict[str, str]:
    if not CRED_FILE.exists():
        return {}
    mode = CRED_FILE.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(
            f"{CRED_FILE} is mode {mode:o}; it is readable by others. "
            f"Run: chmod 600 {CRED_FILE}")
    out: dict[str, str] = {}
    for line in CRED_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        if v:
            out[k.strip()] = v
    return out


def get(name: str) -> str | None:
    """One variable. Environment wins over the file."""
    return os.environ.get(name) or _load_file().get(name)


def require(service: str) -> dict[str, str]:
    """All credentials for a service, or a clear error naming what is missing.

    The error names the VARIABLE, never the value, and tells the caller how to
    obtain it -- so a fetcher fails at startup with an actionable message instead
    of at the HTTP layer with '401 Unauthorized'.
    """
    spec = _SERVICES[service]
    env = _load_file() | {k: v for k, v in os.environ.items() if v}
    groups = [spec["primary"]] + ([spec["fallback"]] if spec.get("fallback") else [])
    for names in groups:
        if all(env.get(n) for n in names):
            return {n: env[n] for n in names}
    raise RuntimeError(
        f"No usable credentials for {service!r}.\n"
        f"  Set {' or '.join(' + '.join(g) for g in groups)}\n"
        f"  in {CRED_FILE} (chmod 600) or in the environment.\n"
        f"  {spec['how']}\n"
        f"  See .env.example for the template.")


def status() -> dict[str, bool]:
    """Which services are configured. Booleans only -- safe to print or log."""
    out = {}
    for svc, spec in _SERVICES.items():
        env = _load_file() | {k: v for k, v in os.environ.items() if v}
        out[svc] = any(all(env.get(n) for n in names)
                       for names in (spec["primary"], spec.get("fallback", ())) if names)
    return out
