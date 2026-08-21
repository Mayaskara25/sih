"""The guard must catch the real payloads that fooled this project."""
import pytest

from core.http_guard import assert_magic, assert_not_html


def test_rejects_dlr_login_page():
    # Exact opening bytes of the EOC UMS login page served with HTTP 200
    # in place of an EnMAP asset (verified 2026-08-21).
    page = b'<!DOCTYPE html><html lang="en">\n\n<head>\n    <meta charset="UTF-8" />'
    with pytest.raises(ValueError, match="HTML page, not data"):
        assert_not_html(page, url="download.geoservice.dlr.de/ENMAP/...")


@pytest.mark.parametrize("page", [
    b"<html><body>404</body></html>",
    b"\n  <!doctype HTML>",            # leading whitespace + mixed case
    b"<!DOCTYPE HTML PUBLIC>",
])
def test_rejects_html_variants(page):
    with pytest.raises(ValueError):
        assert_not_html(page)


def test_accepts_real_tiff():
    assert_magic(b"II*\x00" + b"\x00" * 32, "tiff")
    assert_magic(b"MM\x00*" + b"\x00" * 32, "tiff")


def test_html_beats_magic_for_message():
    """An HTML body must give the credential hint, not a bare magic mismatch."""
    with pytest.raises(ValueError, match="check credentials|Check credentials"):
        assert_magic(b"<!DOCTYPE html><html>", "tiff")


def test_rejects_truncated_non_html():
    with pytest.raises(ValueError, match="not tiff"):
        assert_magic(b"\x00\x00\x00\x00garbage", "tiff")


@pytest.mark.parametrize("magic,label", [
    (b"II*\x00", "classic LE"), (b"MM\x00*", "classic BE"),
    (b"II+\x00", "BigTIFF LE"), (b"MM\x00+", "BigTIFF BE"),
])
def test_accepts_bigtiff_and_classic(magic, label):
    """EnMAP L2A ships both. Verified 2026-08-21: 7 of 8 cubes were BigTIFF.

    Accepting only classic TIFF rejected every valid ~450 MB EnMAP product.
    """
    assert_magic(magic + b"\x00" * 32, "tiff")
