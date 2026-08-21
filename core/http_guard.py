"""Reject HTML-masquerading-as-data before it reaches the parsers.

Three fetches in this project have returned an HTML page with HTTP 200 where a
binary payload was expected:

  * ehu.eus Indian Pines mirror        -> 1428 B error page
  * GitHub raw HYDICE mirror           -> 305 KB error page
  * DLR EnMAP asset without a session  -> 50 KB EOC UMS login page (verified
                                          2026-08-21; status was 200, NOT 401)

Status codes did not distinguish any of them. Content does. Anything that
downloads a scene must call `assert_not_html` / `assert_magic`, never trust
`response.ok`.
"""
from __future__ import annotations

from pathlib import Path

# Leading bytes that mean "this is markup, not a raster".
_HTML_SNIFF = (b"<!doctype", b"<html", b"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<!doctype")

MAGIC = {
    "tiff": (b"II*\x00", b"MM\x00*"),          # little/big endian TIFF, incl. COG
    "zip":  (b"PK\x03\x04",),
    "hdf5": (b"\x89HDF\r\n\x1a\n",),           # .mat v7.3, EnMAP HDF products
    "mat":  (b"MATLAB",),                      # .mat v5
    "jpeg": (b"\xff\xd8\xff",),               # EnMAP STAC thumbnails are JPEG
}


def assert_not_html(payload: bytes | Path, *, url: str = "") -> None:
    """Raise if `payload` starts like an HTML document.

    Case-insensitive: DLR returns `<!DOCTYPE html>`, other mirrors `<html`.
    """
    head = _head(payload)
    low = head[:512].lstrip().lower()
    if any(low.startswith(m) for m in _HTML_SNIFF) or low.startswith(b"<!doctype html"):
        raise ValueError(
            f"Server returned an HTML page, not data{f' for {url}' if url else ''}.\n"
            "  This is usually a login wall or an expired mirror, and it arrives with\n"
            "  HTTP 200 -- the status code will not tell you. Check credentials\n"
            "  (scripts/check_credentials.py) or the URL. See PLAN.md 8.0.")


def assert_magic(payload: bytes | Path, kind: str, *, url: str = "") -> None:
    """Raise unless `payload` begins with one of the magic numbers for `kind`."""
    assert_not_html(payload, url=url)          # better message for the common case
    head = _head(payload)
    expected = MAGIC[kind]
    if not any(head.startswith(m) for m in expected):
        raise ValueError(
            f"Payload is not {kind}{f' ({url})' if url else ''}: "
            f"leading bytes {head[:8]!r} match none of {expected!r}.")


def _head(payload: bytes | Path, n: int = 512) -> bytes:
    if isinstance(payload, (str, Path)):
        with open(payload, "rb") as fh:
            return fh.read(n)
    return bytes(payload[:n])
