"""Copernicus Data Space (CDSE) S3 access helpers.

CDSE's `eodata` bucket needs S3 keys (`core.credentials.require("cdse")`), never the
account password (PLAN.md 1.1b). Two access patterns are needed and both are proven
live against the real service (2026-08-23, see `scripts/verify_access.py::cdse`):

1. **Windowed raster reads** of the big JP2 band files, via GDAL's `/vsis3/` virtual
   filesystem through rasterio -- this is what keeps a multi-date fetch under the
   ~5 GB budget (PLAN.md Phase 5 L3): only the AOI window of only the needed bands
   is ever pulled over the wire, never a whole ~1 GB tile.

   `rasterio.Env(AWS_ACCESS_KEY_ID=..., ...)` REFUSES these keys directly in the
   installed rasterio (1.5.1): "AWS credentials are handled exclusively by boto3."
   boto3 is not a project dependency (heavy, and GDAL's own `/vsis3/` handler does
   not need it). The verified workaround is to set the same `AWS_*` variables as
   real PROCESS environment variables -- GDAL's VSICURL S3 driver reads them
   directly and does its own (non-boto3) request signing. `s3_env()` below does
   this and restores whatever was there before on exit.

2. **Whole-file reads of small metadata** (MTD_MSIL2A.xml ~55 KB, MTD_TL.xml
   ~550 KB) -- these are NOT raster and rasterio/GDAL has no simple "give me the
   bytes" call for an arbitrary VSI file from Python (no `osgeo` binding is
   installed; only rasterio's own GDAL). `sigv4_get()` is a small, dependency-free
   (stdlib `hmac`/`hashlib` + the project's existing `requests`) AWS Signature V4
   GET, verified byte-identical (54838 B / 547998 B, matching the OData `Nodes`
   listing's `ContentLength`) against the same product `s3_env()` opens.

Both were verified against a real product on 2026-08-23:
  S2B_MSIL2A_20260727T052649_N0512_R105_T43RGM_20260727T091136.SAFE
  (chosen only because it was the newest scene found; it is NOT the Level 3 site).

Region string: CDSE's S3-compatible signature check accepts region "default" (used
by both paths here) -- verified live; AWS's own "us-east-1" was not tried because
"default" already works and is what CDSE's own S3 examples use.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import hmac
import os

import requests

from core import credentials

S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
BUCKET = "eodata"
REGION = "default"
SERVICE = "s3"

# The AWS_* variables GDAL's VSICURL S3 driver reads from the process
# environment. Saved/restored by s3_env() so this module never leaves stray
# credentials sitting in os.environ after the `with` block exits.
_AWS_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT",
    "AWS_VIRTUAL_HOSTING", "AWS_HTTPS", "AWS_REGION",
)


@contextlib.contextmanager
def s3_env():
    """Set the process AWS_* env vars GDAL's /vsis3/ driver needs, from
    `credentials.require("cdse")`. Restores prior values (or absence) on exit.

    Usage:
        with cdse_s3.s3_env():
            with rasterio.open(f"/vsis3/{cdse_s3.BUCKET}/{key}") as ds:
                ...
    """
    creds = credentials.require("cdse")
    new = {
        "AWS_ACCESS_KEY_ID": creds["CDSE_S3_ACCESS_KEY"],
        "AWS_SECRET_ACCESS_KEY": creds["CDSE_S3_SECRET_KEY"],
        "AWS_S3_ENDPOINT": S3_ENDPOINT,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "YES",
        "AWS_REGION": REGION,
    }
    saved = {k: os.environ.get(k) for k in _AWS_ENV_KEYS}
    os.environ.update(new)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def vsis3_path(key: str, *, bucket: str = BUCKET) -> str:
    """`/vsis3/eodata/<key>` for use inside an `s3_env()` block."""
    return f"/vsis3/{bucket}/{key.lstrip('/')}"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sigv4_get(key: str, *, bucket: str = BUCKET, endpoint: str = S3_ENDPOINT,
              region: str = REGION, timeout: int = 60) -> requests.Response:
    """AWS Signature V4 GET of one whole S3 object. For small metadata files
    only -- large band files must go through `s3_env()` + rasterio windowed
    reads instead, or the whole-object bytes land in process memory and defeat
    the AOI-window byte budget (PLAN.md Phase 5 L3).

    Credentials come from `credentials.require("cdse")` on every call and are
    never logged/printed/returned -- only used to compute the HMAC signature.
    """
    creds = credentials.require("cdse")
    access = creds["CDSE_S3_ACCESS_KEY"]
    secret = creds["CDSE_S3_SECRET_KEY"]

    t = datetime.datetime.now(datetime.timezone.utc)
    amzdate = t.strftime("%Y%m%dT%H%M%SZ")
    datestamp = t.strftime("%Y%m%d")

    canonical_uri = "/" + bucket + "/" + "/".join(
        requests.utils.quote(p, safe="") for p in key.lstrip("/").split("/"))
    payload_hash = hashlib.sha256(b"").hexdigest()
    canonical_headers = (f"host:{endpoint}\nx-amz-content-sha256:{payload_hash}\n"
                         f"x-amz-date:{amzdate}\n")
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(["GET", canonical_uri, "", canonical_headers,
                                    signed_headers, payload_hash])

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{datestamp}/{region}/{SERVICE}/aws4_request"
    string_to_sign = "\n".join([
        algorithm, amzdate, credential_scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])

    k_signing = _sign(_sign(_sign(_sign(("AWS4" + secret).encode(), datestamp),
                                   region), SERVICE), "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

    authorization = (f"{algorithm} Credential={access}/{credential_scope}, "
                     f"SignedHeaders={signed_headers}, Signature={signature}")
    headers = {
        "x-amz-date": amzdate,
        "x-amz-content-sha256": payload_hash,
        "Authorization": authorization,
    }
    return requests.get(f"https://{endpoint}{canonical_uri}", headers=headers, timeout=timeout)
