"""core/cdse_s3.py -- S3 access helpers for CDSE (Phase 5 Level 3).

No network access required to pass: `s3_env` is tested purely on process
environment save/restore, and `sigv4_get`'s signing is tested by capturing
the `requests.get` call rather than performing it.
"""
from __future__ import annotations

import os

import pytest

from core import cdse_s3, credentials


def _fake_require(monkeypatch, access="AKIAFAKE", secret="fakesecretfakesecretfakesecret"):
    def _fake(service):
        assert service == "cdse"
        return {"CDSE_S3_ACCESS_KEY": access, "CDSE_S3_SECRET_KEY": secret}
    monkeypatch.setattr(credentials, "require", _fake)


# --- s3_env -------------------------------------------------------------

def test_s3_env_sets_expected_aws_vars(monkeypatch):
    _fake_require(monkeypatch)
    for k in cdse_s3._AWS_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)

    with cdse_s3.s3_env():
        assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAFAKE"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "fakesecretfakesecretfakesecret"
        assert os.environ["AWS_S3_ENDPOINT"] == cdse_s3.S3_ENDPOINT
        assert os.environ["AWS_VIRTUAL_HOSTING"] == "FALSE"
        assert os.environ["AWS_HTTPS"] == "YES"
        assert os.environ["AWS_REGION"] == cdse_s3.REGION


def test_s3_env_restores_absence_afterwards(monkeypatch):
    _fake_require(monkeypatch)
    for k in cdse_s3._AWS_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)

    with cdse_s3.s3_env():
        assert "AWS_ACCESS_KEY_ID" in os.environ
    for k in cdse_s3._AWS_ENV_KEYS:
        assert k not in os.environ


def test_s3_env_restores_prior_value_afterwards(monkeypatch):
    _fake_require(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "some-prior-value")

    with cdse_s3.s3_env():
        assert os.environ["AWS_REGION"] == cdse_s3.REGION
    assert os.environ["AWS_REGION"] == "some-prior-value"


def test_s3_env_restores_even_on_exception(monkeypatch):
    _fake_require(monkeypatch)
    for k in cdse_s3._AWS_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)

    with pytest.raises(RuntimeError):
        with cdse_s3.s3_env():
            raise RuntimeError("boom")
    for k in cdse_s3._AWS_ENV_KEYS:
        assert k not in os.environ


def test_s3_env_propagates_missing_credential(monkeypatch):
    def _raise(service):
        raise RuntimeError("No usable credentials for 'cdse'.\n  Set CDSE_S3_ACCESS_KEY + CDSE_S3_SECRET_KEY")
    monkeypatch.setattr(credentials, "require", _raise)

    with pytest.raises(RuntimeError, match="CDSE_S3_ACCESS_KEY"):
        with cdse_s3.s3_env():
            pass


def test_vsis3_path():
    assert cdse_s3.vsis3_path("Sentinel-2/foo.jp2") == "/vsis3/eodata/Sentinel-2/foo.jp2"
    assert cdse_s3.vsis3_path("/Sentinel-2/foo.jp2") == "/vsis3/eodata/Sentinel-2/foo.jp2"


# --- sigv4_get ------------------------------------------------------------

def test_sigv4_get_never_puts_secret_in_url_or_unsigned_headers(monkeypatch):
    """The secret key must only ever be used to compute the HMAC signature --
    never appear verbatim in the URL or in a header value other than inside
    the opaque `Signature=` hex digest of the Authorization header."""
    _fake_require(monkeypatch, access="AKIAEXPOSEME", secret="topsecretvalueshouldnotleak12345")

    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        class _Resp:
            status_code = 200
            content = b"<?xml ok?>"
        return _Resp()

    monkeypatch.setattr(cdse_s3.requests, "get", fake_get)

    resp = cdse_s3.sigv4_get("Sentinel-2/MSI/L2A/foo/MTD_MSIL2A.xml")
    assert resp.status_code == 200

    assert "topsecretvalueshouldnotleak12345" not in captured["url"]
    for v in captured["headers"].values():
        assert "topsecretvalueshouldnotleak12345" not in v
    # the access key ID, unlike the secret, is expected in Authorization
    assert "AKIAEXPOSEME" in captured["headers"]["Authorization"]
    assert captured["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=")
    assert "SignedHeaders=host;x-amz-content-sha256;x-amz-date" in captured["headers"]["Authorization"]


def test_sigv4_get_url_is_path_style_https(monkeypatch):
    _fake_require(monkeypatch)
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        class _Resp:
            status_code = 200
            content = b""
        return _Resp()

    monkeypatch.setattr(cdse_s3.requests, "get", fake_get)
    cdse_s3.sigv4_get("Sentinel-2/MSI/L2A/2026/01/01/foo.SAFE/manifest.safe")
    assert captured["url"] == (
        "https://eodata.dataspace.copernicus.eu/eodata/"
        "Sentinel-2/MSI/L2A/2026/01/01/foo.SAFE/manifest.safe")


def test_sigv4_get_missing_credential_raises_before_any_request(monkeypatch):
    def _raise(service):
        raise RuntimeError("Set CDSE_S3_ACCESS_KEY + CDSE_S3_SECRET_KEY")
    monkeypatch.setattr(credentials, "require", _raise)

    called = []
    monkeypatch.setattr(cdse_s3.requests, "get", lambda *a, **k: called.append(1))

    with pytest.raises(RuntimeError, match="CDSE_S3_ACCESS_KEY"):
        cdse_s3.sigv4_get("some/key.xml")
    assert not called, "must not attempt a request with no credentials"
