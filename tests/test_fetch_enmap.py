"""scripts/fetch_enmap.py -- EnMAP L2A fetcher/reconciler.

No network access required to pass: `requests.Session.get`/`.head` are
monkeypatched to a URL dispatch table, `_cas_login` and `credentials.require`
are monkeypatched directly, and `stac_search` is monkeypatched to return
canned scenes. Required to FAIL (i.e. be caught by a test here) if it
regresses:
  - a missing DLR_USERNAME/DLR_PASSWORD is swallowed instead of reported
  - a truncated download (bytes written != Content-Length) is accepted
  - a wrong-magic-bytes download (HTML wall, O11-shaped) is accepted
  - a second manifest run overwrites the first instead of merging by id
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import credentials
from scripts import fetch_enmap as fe


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# A structurally real-shaped BigTIFF/TIFF magic header, not a full file.
TIFF_CONTENT = b"II*\x00" + b"\x00" * 500
XML_CONTENT = b"<level_X>fake EnMAP metadata</level_X>"
HTML_WALL = b"<!DOCTYPE html><html><body>please log in</body></html>"


class _FakeResp:
    def __init__(self, content: bytes = b"", status_code: int = 200, headers: dict | None = None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1 << 20):
        data = self.content
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    def json(self):
        return json.loads(self.content)


def _patch_session(monkeypatch, get_by_url: dict, head_by_url: dict | None = None):
    def fake_get(self, url, **kw):
        if url not in get_by_url:
            raise AssertionError(f"unexpected GET {url}")
        return get_by_url[url]

    def fake_head(self, url, **kw):
        if not head_by_url or url not in head_by_url:
            raise AssertionError(f"unexpected HEAD {url}")
        return head_by_url[url]

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(requests.Session, "head", fake_head)


def _fake_scene(scene_id: str, image_url: str, metadata_url: str, **extra) -> dict:
    d = dict(id=scene_id, date="2026-01-01T00:00:00", cloud=1.0, snow=0.0, epsg=32643,
              declared_dtype="int16", sun_elevation=55.0, datatake="DT0000000001",
              assets={"image": image_url, "metadata": metadata_url})
    d.update(extra)
    return d


# --- _asset_filename ---------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://x/a/B-SPECTRAL_IMAGE_COG.TIF", "B-SPECTRAL_IMAGE_COG.TIF"),
    ("https://x/a/B-METADATA.XML", "B-METADATA.XML"),
    ("https://x/a/B%20C-METADATA.XML", "B C-METADATA.XML"),
])
def test_asset_filename_is_url_basename_unnormalised(url, expected):
    assert fe._asset_filename(url) == expected


# --- both METADATA spellings honored on the READ side ------------------------

def test_local_path_for_metadata_prefers_existing_double_suffix(tmp_path):
    scene_id = "ENMAP01-____L2A-DT0000000099_FAKE"
    spectral = tmp_path / f"{scene_id}{fe.SPECTRAL_SUFFIX}"
    spectral.write_bytes(TIFF_CONTENT)
    double = tmp_path / f"{scene_id}-METADATA.XML.XML"
    double.write_bytes(XML_CONTENT)

    url = f"https://x/{scene_id}-METADATA.XML"  # URL always gives the single form
    resolved = fe._local_path_for(tmp_path, scene_id, "metadata", url)
    assert resolved == double
    assert fe._asset_is_present(tmp_path, scene_id, "metadata", url)


def test_local_path_for_metadata_single_suffix_when_thats_all_there_is(tmp_path):
    scene_id = "ENMAP01-____L2A-DT0000000098_FAKE"
    single = tmp_path / f"{scene_id}-METADATA.XML"
    single.write_bytes(XML_CONTENT)
    url = f"https://x/{scene_id}-METADATA.XML"
    assert fe._local_path_for(tmp_path, scene_id, "metadata", url) == single


def test_asset_is_present_false_when_neither_spelling_exists(tmp_path):
    scene_id = "ENMAP01-____L2A-DT0000000097_FAKE"
    url = f"https://x/{scene_id}-METADATA.XML"
    assert not fe._asset_is_present(tmp_path, scene_id, "metadata", url)


# --- _download_asset: truncation and entitlement must be REJECTED, not accepted --

def test_download_asset_accepts_matching_tiff(tmp_path):
    session = requests.Session()
    url = "https://x/scene-SPECTRAL_IMAGE_COG.TIF"

    def fake_get(self, u, **kw):
        assert u == url
        return _FakeResp(TIFF_CONTENT, headers={"content-length": str(len(TIFF_CONTENT))})
    session.get = fake_get.__get__(session)

    dest = tmp_path / "scene-SPECTRAL_IMAGE_COG.TIF"
    res = fe._download_asset(session, url, dest, "tiff")
    assert res["status"] == "downloaded"
    assert dest.exists()
    assert dest.read_bytes() == TIFF_CONTENT
    assert res["sha256"] == _sha256(TIFF_CONTENT)
    assert not (tmp_path / (dest.name + ".part")).exists()


def test_download_asset_rejects_truncated_download(tmp_path):
    """Magic bytes alone would pass here -- content starts with a real TIFF
    header. Only the Content-Length comparison catches the truncation."""
    session = requests.Session()
    url = "https://x/scene-SPECTRAL_IMAGE_COG.TIF"
    declared_length = len(TIFF_CONTENT) + 10_000_000  # server claims far more than sent

    def fake_get(self, u, **kw):
        return _FakeResp(TIFF_CONTENT, headers={"content-length": str(declared_length)})
    session.get = fake_get.__get__(session)

    dest = tmp_path / "scene-SPECTRAL_IMAGE_COG.TIF"
    with pytest.raises(fe.TruncatedDownloadError):
        fe._download_asset(session, url, dest, "tiff")
    assert not dest.exists(), "a truncated download must NOT be accepted as the real file"
    assert (dest.parent / (dest.name + ".part")).exists(), "partial bytes stay as .part for diagnosis"


def test_download_asset_rejects_html_wall_as_entitlement_error(tmp_path):
    """HTTP 200 + HTML body is DLR's O11 signature: authenticated, but not
    entitled to this asset. Content-Length matches the HTML body exactly, so
    only the magic-byte check catches it."""
    session = requests.Session()
    url = "https://x/scene-SPECTRAL_IMAGE_COG.TIF"

    def fake_get(self, u, **kw):
        return _FakeResp(HTML_WALL, headers={"content-length": str(len(HTML_WALL))})
    session.get = fake_get.__get__(session)

    dest = tmp_path / "scene-SPECTRAL_IMAGE_COG.TIF"
    with pytest.raises(fe.AssetEntitlementError):
        fe._download_asset(session, url, dest, "tiff")
    assert not dest.exists()


def test_download_asset_metadata_uses_html_check_not_magic(tmp_path):
    """The XML sidecar has no TIFF magic bytes and must not be checked for
    one -- only assert_not_html applies."""
    session = requests.Session()
    url = "https://x/scene-METADATA.XML"

    def fake_get(self, u, **kw):
        return _FakeResp(XML_CONTENT, headers={"content-length": str(len(XML_CONTENT))})
    session.get = fake_get.__get__(session)

    dest = tmp_path / "scene-METADATA.XML"
    res = fe._download_asset(session, url, dest, "xml")
    assert res["status"] == "downloaded"
    assert dest.read_bytes() == XML_CONTENT


# --- merge_manifest: must MERGE by id, not overwrite -------------------------

def test_merge_manifest_second_run_keeps_first_products_record(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    rec_a = dict(id="A", date="2026-01-01", assets={"image": dict(bytes=10, sha256="x", status="downloaded", url="u")})
    rec_b = dict(id="B", date="2026-01-02", assets={"image": dict(bytes=20, sha256="y", status="downloaded", url="u")})

    m1 = fe.merge_manifest(manifest_path, [rec_a])
    assert m1["n_products"] == 1

    m2 = fe.merge_manifest(manifest_path, [rec_b])
    assert m2["n_products"] == 2, "second run must MERGE, not overwrite, the first product"
    ids = {p["id"] for p in m2["products"]}
    assert ids == {"A", "B"}

    on_disk = json.loads(manifest_path.read_text())
    assert {p["id"] for p in on_disk["products"]} == {"A", "B"}


def test_merge_manifest_soft_merges_assets_within_one_product(tmp_path):
    """A later run that only touches core assets (image/metadata) must not
    erase an earlier run's quality-mask asset records for the same product,
    and vice versa -- this is what makes --reconcile safe to run after a
    --with-quality-masks fetch, and after a plain fetch."""
    manifest_path = tmp_path / "manifest.json"
    rec1 = dict(id="A", date="2026-01-01",
                assets={"image": dict(bytes=10, sha256="x", status="downloaded", url="u"),
                        "quality_cloud": dict(bytes=1, sha256="q", status="downloaded", url="uq")})
    fe.merge_manifest(manifest_path, [rec1])

    rec2 = dict(id="A", date="2026-01-01",
                assets={"image": dict(bytes=10, sha256="x2", status="cached", url="u")})
    m2 = fe.merge_manifest(manifest_path, [rec2])
    assert m2["n_products"] == 1
    assets = m2["products"][0]["assets"]
    assert "quality_cloud" in assets, "prior quality_cloud record must survive"
    assert assets["image"]["sha256"] == "x2", "the touched asset should reflect the newer run"


def test_merge_manifest_survives_unreadable_prior_file(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not valid json")
    rec = dict(id="A", date="2026-01-01", assets={})
    m = fe.merge_manifest(manifest_path, [rec])
    assert m["n_products"] == 1
    assert "WARNING" in capsys.readouterr().err


# --- credential failure (case a) ----------------------------------------------

def test_main_fails_loudly_on_missing_credential(monkeypatch, capsys, tmp_path):
    def _raise(service):
        raise RuntimeError(
            "No usable credentials for 'dlr'.\n  Set DLR_USERNAME + DLR_PASSWORD")
    monkeypatch.setattr(credentials, "require", _raise)
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--out-dir", str(tmp_path), "--manifest", str(tmp_path / "m.json")])

    rc = fe.main()
    assert rc == 2
    err = capsys.readouterr().err
    assert "DLR_USERNAME" in err
    assert "DLR_PASSWORD" in err


# --- CAS login rejected (case b) ----------------------------------------------

def test_main_reports_rejected_credentials_distinctly(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(credentials, "require", lambda service: {"DLR_USERNAME": "u", "DLR_PASSWORD": "p"})
    scene = _fake_scene("ENMAP01-____L2A-DT0000000001_FAKE",
                        "https://x/S-SPECTRAL_IMAGE_COG.TIF", "https://x/S-METADATA.XML")
    monkeypatch.setattr(fe, "stac_search", lambda *a, **k: [scene])
    monkeypatch.setattr(fe, "_cas_login", lambda session, cas, creds: (401, []))
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--out-dir", str(tmp_path), "--manifest", str(tmp_path / "m.json"),
        "--yes", "--min-free-gb", "0"])

    rc = fe.main()
    assert rc == 3
    err = capsys.readouterr().err
    assert "rejected" in err.lower() or "not activated" in err.lower()


# --- entitlement failure (case c) does not abort the whole run ---------------

def test_main_asset_entitlement_failure_recorded_not_fatal(monkeypatch, capsys, tmp_path):
    image_url = "https://x/S-SPECTRAL_IMAGE_COG.TIF"
    meta_url = "https://x/S-METADATA.XML"
    scene = _fake_scene("ENMAP01-____L2A-DT0000000002_FAKE", image_url, meta_url)

    monkeypatch.setattr(credentials, "require", lambda service: {"DLR_USERNAME": "u", "DLR_PASSWORD": "p"})
    monkeypatch.setattr(fe, "stac_search", lambda *a, **k: [scene])
    monkeypatch.setattr(fe, "_cas_login", lambda session, cas, creds: (200, ["TGC-1"]))

    get_by_url = {
        image_url: _FakeResp(HTML_WALL, headers={"content-length": str(len(HTML_WALL))}),
        meta_url: _FakeResp(XML_CONTENT, headers={"content-length": str(len(XML_CONTENT))}),
    }
    head_by_url = {
        image_url: _FakeResp(headers={"content-length": str(len(HTML_WALL))}),
        meta_url: _FakeResp(headers={"content-length": str(len(XML_CONTENT))}),
    }
    _patch_session(monkeypatch, get_by_url, head_by_url)
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--out-dir", str(tmp_path), "--manifest", str(tmp_path / "m.json"),
        "--yes", "--min-free-gb", "0"])

    rc = fe.main()
    assert rc == 1  # a real failure happened, but the run completed rather than crashing
    manifest = json.loads((tmp_path / "m.json").read_text())
    assets = manifest["products"][0]["assets"]
    assert assets["image"]["status"] == "failed_entitlement"
    assert assets["metadata"]["status"] == "downloaded"  # the OTHER asset still succeeded
    out = capsys.readouterr().out
    assert "entitlement" in out.lower()


# --- end-to-end: download, skip-on-rerun, merge on a new scene, --limit ------

def test_main_end_to_end_download_then_cache_then_merge_and_limit(monkeypatch, capsys, tmp_path):
    out_dir = tmp_path / "enmap"
    manifest_path = tmp_path / "manifest.json"

    def scene(n):
        return _fake_scene(f"ENMAP01-____L2A-DT000000000{n}_FAKE",
                            f"https://x/S{n}-SPECTRAL_IMAGE_COG.TIF",
                            f"https://x/S{n}-METADATA.XML")

    scene1 = scene(1)
    monkeypatch.setattr(credentials, "require", lambda service: {"DLR_USERNAME": "u", "DLR_PASSWORD": "p"})
    monkeypatch.setattr(fe, "_cas_login", lambda session, cas, creds: (200, ["TGC-1"]))

    get_by_url = {
        scene1["assets"]["image"]: _FakeResp(TIFF_CONTENT, headers={"content-length": str(len(TIFF_CONTENT))}),
        scene1["assets"]["metadata"]: _FakeResp(XML_CONTENT, headers={"content-length": str(len(XML_CONTENT))}),
    }
    head_by_url = {
        scene1["assets"]["image"]: _FakeResp(headers={"content-length": str(len(TIFF_CONTENT))}),
        scene1["assets"]["metadata"]: _FakeResp(headers={"content-length": str(len(XML_CONTENT))}),
    }
    _patch_session(monkeypatch, get_by_url, head_by_url)
    monkeypatch.setattr(fe, "stac_search", lambda *a, **k: [scene1])
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--out-dir", str(out_dir), "--manifest", str(manifest_path),
        "--yes", "--min-free-gb", "0"])

    rc = fe.main()
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_products"] == 1
    assert manifest["products"][0]["assets"]["image"]["status"] == "downloaded"

    # Re-run unchanged: must be skipped (cached), no new GETs allowed.
    def fail_get(self, url, **kw):
        raise AssertionError(f"should not re-GET {url} on an unchanged re-run")
    monkeypatch.setattr(requests.Session, "get", fail_get)
    rc2 = fe.main()
    assert rc2 == 0
    manifest2 = json.loads(manifest_path.read_text())
    assert manifest2["n_products"] == 1
    assert manifest2["products"][0]["assets"]["image"]["status"] == "cached"

    # Now two NEW scenes appear upstream; --limit caps how many are fetched
    # this run, and the manifest must still hold scene1 from before (merge,
    # not overwrite) plus exactly one new one.
    scene2, scene3 = scene(2), scene(3)
    for s in (scene2, scene3):
        get_by_url[s["assets"]["image"]] = _FakeResp(TIFF_CONTENT, headers={"content-length": str(len(TIFF_CONTENT))})
        get_by_url[s["assets"]["metadata"]] = _FakeResp(XML_CONTENT, headers={"content-length": str(len(XML_CONTENT))})
        head_by_url[s["assets"]["image"]] = _FakeResp(headers={"content-length": str(len(TIFF_CONTENT))})
        head_by_url[s["assets"]["metadata"]] = _FakeResp(headers={"content-length": str(len(XML_CONTENT))})
    _patch_session(monkeypatch, get_by_url, head_by_url)
    monkeypatch.setattr(fe, "stac_search", lambda *a, **k: [scene1, scene2, scene3])
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--out-dir", str(out_dir), "--manifest", str(manifest_path),
        "--yes", "--min-free-gb", "0", "--limit", "1"])

    rc3 = fe.main()
    assert rc3 == 0
    manifest3 = json.loads(manifest_path.read_text())
    assert manifest3["n_products"] == 2, "scene1 (cached) + exactly one new scene under --limit 1"
    ids = {p["id"] for p in manifest3["products"]}
    assert scene1["id"] in ids
    assert len(ids & {scene2["id"], scene3["id"]}) == 1


def test_main_refuses_without_confirmation_in_non_interactive_session(monkeypatch, tmp_path):
    scene = _fake_scene("ENMAP01-____L2A-DT0000000004_FAKE",
                        "https://x/S4-SPECTRAL_IMAGE_COG.TIF", "https://x/S4-METADATA.XML")
    monkeypatch.setattr(credentials, "require", lambda service: {"DLR_USERNAME": "u", "DLR_PASSWORD": "p"})
    monkeypatch.setattr(fe, "stac_search", lambda *a, **k: [scene])
    monkeypatch.setattr(fe, "_cas_login", lambda session, cas, creds: (200, ["TGC-1"]))
    head_by_url = {
        scene["assets"]["image"]: _FakeResp(headers={"content-length": "500"}),
        scene["assets"]["metadata"]: _FakeResp(headers={"content-length": "50"}),
    }

    def fail_get(self, url, **kw):
        raise AssertionError("must not download before a confirmation is obtained")
    monkeypatch.setattr(requests.Session, "get", fail_get)
    monkeypatch.setattr(requests.Session, "head", lambda self, url, **kw: head_by_url[url])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--out-dir", str(tmp_path / "enmap"), "--manifest", str(tmp_path / "m.json"),
        "--min-free-gb", "0"])  # no --yes

    rc = fe.main()
    assert rc == 1
    assert not (tmp_path / "m.json").exists()


# --- --reconcile ---------------------------------------------------------------

def test_reconcile_indexes_existing_files_with_catalog_match(monkeypatch, capsys, tmp_path):
    out_dir = tmp_path / "enmap"
    out_dir.mkdir()
    scene_id = "ENMAP01-____L2A-DT0000000005_FAKE"
    spectral = out_dir / f"{scene_id}{fe.SPECTRAL_SUFFIX}"
    spectral.write_bytes(TIFF_CONTENT)
    meta = out_dir / f"{scene_id}-METADATA.XML.XML"
    meta.write_bytes(XML_CONTENT)

    catalog_doc = {
        "properties": {"datetime": "2026-01-01T00:00:00Z", "eo:cloud_cover": 2.0,
                       "eo:snow_cover": 0.0, "proj:epsg": 32643},
        "assets": {"image": {"href": f"https://x/{scene_id}-SPECTRAL_IMAGE_COG.TIF"},
                   "metadata": {"href": f"https://x/{scene_id}-METADATA.XML"}},
    }

    def fake_requests_get(url, timeout=None):
        assert scene_id in url
        return _FakeResp(json.dumps(catalog_doc).encode(), status_code=200)
    monkeypatch.setattr(fe.requests, "get", fake_requests_get)
    manifest_path = tmp_path / "m.json"
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--reconcile", "--out-dir", str(out_dir), "--manifest", str(manifest_path)])

    rc = fe.main()
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_products"] == 1
    prod = manifest["products"][0]
    assert prod["catalog_lookup_ok"] is True
    assert prod["cloud"] == 2.0
    assert prod["assets"]["metadata"]["filename"] == meta.name  # double-suffix form preserved
    assert prod["assets"]["image"]["bytes"] == len(TIFF_CONTENT)
    assert prod["assets"]["image"]["sha256"] == _sha256(TIFF_CONTENT)


def test_reconcile_records_local_facts_when_catalog_lookup_404s(monkeypatch, tmp_path):
    out_dir = tmp_path / "enmap"
    out_dir.mkdir()
    scene_id = "ENMAP01-____L2A-DT0000000006_FAKE"
    spectral = out_dir / f"{scene_id}{fe.SPECTRAL_SUFFIX}"
    spectral.write_bytes(TIFF_CONTENT)
    # No metadata sidecar at all for this one -- reconcile must still record the image.

    monkeypatch.setattr(fe.requests, "get", lambda url, timeout=None: _FakeResp(b"not found", status_code=404))
    manifest_path = tmp_path / "m.json"
    monkeypatch.setattr(sys, "argv", [
        "fetch_enmap.py", "--reconcile", "--out-dir", str(out_dir), "--manifest", str(manifest_path)])

    rc = fe.main()
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    prod = manifest["products"][0]
    assert prod["catalog_lookup_ok"] is False
    assert prod["cloud"] is None
    assert prod["assets"]["image"]["bytes"] == len(TIFF_CONTENT)
    assert "metadata" not in prod["assets"]


def test_reconcile_with_no_products_on_disk_fails_cleanly(tmp_path):
    out_dir = tmp_path / "empty_enmap"
    out_dir.mkdir()
    import scripts.fetch_enmap as fe_mod
    rc = fe_mod._run_reconcile(out_dir, tmp_path / "m.json")
    assert rc == 1
