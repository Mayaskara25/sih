"""scripts/fetch_sentinel2.py -- Sentinel-2 L2A fetcher (Phase 5 Level 3, O5).

No network access required to pass: `cdse_s3.sigv4_get` and the OData
`requests.get` calls are monkeypatched everywhere. Two failure modes are
required to FAIL cleanly (not silently substitute a guessed value):
  - a missing credential variable (`CDSE_S3_ACCESS_KEY`/`CDSE_S3_SECRET_KEY`)
  - a product whose MTD_MSIL2A.xml lacks BOA_ADD_OFFSET or
    BOA_QUANTIFICATION_VALUE (the exact PLAN.md L2A-offset trap)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import credentials
from scripts import fetch_sentinel2 as fs


class _FakeResp:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _mtd_msil2a_xml(*, with_offset=True, with_quant=True, n_bands=13) -> bytes:
    """A minimal but structurally real MTD_MSIL2A.xml -- same element names/
    nesting as the fields this project's parser reads (verified against a
    real product, 2026-08-23), far fewer other fields."""
    band_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
                  "B9", "B10", "B11", "B12"][:n_bands]
    resolutions = {"B1": 60, "B2": 10, "B3": 10, "B4": 10, "B5": 20, "B6": 20,
                   "B7": 20, "B8": 10, "B8A": 20, "B9": 60, "B10": 60,
                   "B11": 20, "B12": 20}
    centers = {"B1": 442.3, "B2": 492.3, "B3": 559.0, "B4": 665.0, "B5": 703.8,
               "B6": 739.1, "B7": 779.7, "B8": 833.0, "B8A": 864.0, "B9": 943.2,
               "B10": 1376.9, "B11": 1610.4, "B12": 2185.7}
    spectral = "".join(
        f'<Spectral_Information bandId="{i}" physicalBand="{b}">'
        f"<RESOLUTION>{resolutions[b]}</RESOLUTION>"
        f'<Wavelength><MIN unit="nm">1</MIN><MAX unit="nm">1</MAX>'
        f'<CENTRAL unit="nm">{centers[b]}</CENTRAL></Wavelength>'
        "</Spectral_Information>"
        for i, b in enumerate(band_names)
    )
    quant = ('<QUANTIFICATION_VALUES_LIST><BOA_QUANTIFICATION_VALUE unit="none">'
             "10000</BOA_QUANTIFICATION_VALUE></QUANTIFICATION_VALUES_LIST>") if with_quant else ""
    offset = ("<BOA_ADD_OFFSET_VALUES_LIST>" +
              "".join(f'<BOA_ADD_OFFSET band_id="{i}">-1000</BOA_ADD_OFFSET>'
                      for i in range(n_bands)) +
              "</BOA_ADD_OFFSET_VALUES_LIST>") if with_offset else ""
    xml = (
        '<?xml version="1.0"?><n1:Level-2A_User_Product xmlns:n1="x">'
        "<n1:General_Info><Product_Info>"
        "<PRODUCT_START_TIME>2026-01-01T00:00:00.000Z</PRODUCT_START_TIME>"
        "<PROCESSING_BASELINE>05.12</PROCESSING_BASELINE>"
        "</Product_Info>"
        "<Datatake><DATATAKE_SENSING_START>2026-01-01T00:00:00.000Z</DATATAKE_SENSING_START></Datatake>"
        "</n1:General_Info>"
        f"<n1:General_Info>{quant}{offset}"
        f"<Product_Image_Characteristics><Spectral_Information_List>{spectral}"
        "</Spectral_Information_List></Product_Image_Characteristics>"
        "</n1:General_Info>"
        "</n1:Level-2A_User_Product>"
    )
    return xml.encode()


def _mtd_tl_xml(*, with_sensing_time=True) -> bytes:
    sensing = ("<SENSING_TIME>2026-01-01T00:14:00.000000Z</SENSING_TIME>"
              if with_sensing_time else "")
    return (
        "<?xml version=\"1.0\"?><n1:Tile xmlns:n1=\"x\"><n1:General_Info>"
        f"<TILE_ID>fake</TILE_ID>{sensing}"
        "</n1:General_Info><n1:Geometric_Info>"
        "<Tile_Geocoding><HORIZONTAL_CS_CODE>EPSG:32643</HORIZONTAL_CS_CODE></Tile_Geocoding>"
        "</n1:Geometric_Info></n1:Tile>"
    ).encode()


def _patch_sigv4(monkeypatch, content_by_suffix: dict[str, bytes]):
    def fake_sigv4_get(key, **kw):
        for suffix, content in content_by_suffix.items():
            if key.endswith(suffix):
                return _FakeResp(content)
        raise AssertionError(f"unexpected key {key!r}")
    monkeypatch.setattr(fs.cdse_s3, "sigv4_get", fake_sigv4_get)


# --- _normalize_band_code --------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("B1", "B01"), ("B2", "B02"), ("B9", "B09"),
    ("B8A", "B8A"), ("B11", "B11"), ("B12", "B12"),
])
def test_normalize_band_code(raw, expected):
    assert fs._normalize_band_code(raw) == expected


# --- fetch_mtd_msil2a -------------------------------------------------------

def test_fetch_mtd_msil2a_parses_real_shaped_xml(monkeypatch):
    _patch_sigv4(monkeypatch, {"/MTD_MSIL2A.xml": _mtd_msil2a_xml()})
    mtd = fs.fetch_mtd_msil2a("some/key/prefix")
    assert mtd["quantification_value"] == 10000.0
    assert mtd["boa_add_offset_distinct"] == [-1000.0]
    assert mtd["boa_add_offset_uniform"] is True
    assert mtd["processing_baseline"] == "05.12"
    assert mtd["spectral"]["B02"]["central_nm"] == 492.3
    assert mtd["spectral"]["B8A"]["resolution_m"] == 20.0
    assert set(fs.BANDS_20M) <= set(mtd["spectral"])


def test_fetch_mtd_msil2a_raises_without_boa_add_offset(monkeypatch):
    """The exact PLAN.md trap: a product whose metadata lacks the offset
    must fail loudly, never silently assume -1000 or skip rescaling."""
    _patch_sigv4(monkeypatch, {"/MTD_MSIL2A.xml": _mtd_msil2a_xml(with_offset=False)})
    with pytest.raises(ValueError, match="BOA_ADD_OFFSET"):
        fs.fetch_mtd_msil2a("some/key/prefix")


def test_fetch_mtd_msil2a_raises_without_quantification_value(monkeypatch):
    _patch_sigv4(monkeypatch, {"/MTD_MSIL2A.xml": _mtd_msil2a_xml(with_quant=False)})
    with pytest.raises(ValueError, match="BOA_QUANTIFICATION_VALUE"):
        fs.fetch_mtd_msil2a("some/key/prefix")


def test_fetch_mtd_msil2a_nonuniform_offset_reported_not_hidden(monkeypatch):
    xml = _mtd_msil2a_xml()
    # Poison one band's offset to a different value -- verifies the
    # gain_uniform-style shape (distinct list + uniform flag), mirroring
    # verify_enmap.py's GainOfBand handling, rather than reading band 0 alone.
    poisoned = xml.replace(b'band_id="5">-1000', b'band_id="5">-2000', 1)
    _patch_sigv4(monkeypatch, {"/MTD_MSIL2A.xml": poisoned})
    mtd = fs.fetch_mtd_msil2a("some/key/prefix")
    assert mtd["boa_add_offset_uniform"] is False
    assert sorted(mtd["boa_add_offset_distinct"]) == [-2000.0, -1000.0]


# --- fetch_mtd_tl -----------------------------------------------------------

def test_fetch_mtd_tl_parses_sensing_time(monkeypatch):
    _patch_sigv4(monkeypatch, {"/MTD_TL.xml": _mtd_tl_xml()})
    tl = fs.fetch_mtd_tl("some/key/prefix", "GRANULE_X")
    assert tl["sensing_time"] == "2026-01-01T00:14:00.000000Z"
    assert tl["crs"] == "EPSG:32643"


def test_fetch_mtd_tl_raises_without_sensing_time(monkeypatch):
    """D33: meta.acquired must never silently become None for Sentinel-2."""
    _patch_sigv4(monkeypatch, {"/MTD_TL.xml": _mtd_tl_xml(with_sensing_time=False)})
    with pytest.raises(ValueError, match="SENSING_TIME"):
        fs.fetch_mtd_tl("some/key/prefix", "GRANULE_X")


# --- discover_band_files ----------------------------------------------------

def test_discover_band_files_matches_20m_filenames(monkeypatch):
    nodes = [{"Name": f"T43RGM_20260101T000000_{b}_20m.jp2"}
             for b in ("B02", "B03", "B04", "B8A", "B11", "B12", "SCL")]
    nodes.append({"Name": "T43RGM_20260101T000000_TCI_20m.jp2"})  # decoy, not wanted
    monkeypatch.setattr(fs, "_nodes", lambda *a, **k: nodes)
    files = fs.discover_band_files("pid", "name", "gran")
    assert set(files) == set(fs.BANDS_20M) | {fs.SCL_NAME}
    assert files["B02"] == "T43RGM_20260101T000000_B02_20m.jp2"


def test_discover_band_files_raises_on_missing_band(monkeypatch):
    nodes = [{"Name": f"T43RGM_20260101T000000_{b}_20m.jp2"}
             for b in ("B02", "B03")]  # missing the rest
    monkeypatch.setattr(fs, "_nodes", lambda *a, **k: nodes)
    with pytest.raises(RuntimeError, match="missing band"):
        fs.discover_band_files("pid", "name", "gran")


# --- credential failure ------------------------------------------------------

def test_main_fails_loudly_on_missing_credential(monkeypatch, capsys):
    def _raise(service):
        raise RuntimeError(
            "No usable credentials for 'cdse'.\n"
            "  Set CDSE_S3_ACCESS_KEY + CDSE_S3_SECRET_KEY")
    monkeypatch.setattr(credentials, "require", _raise)
    monkeypatch.setattr(sys, "argv", [
        "fetch_sentinel2.py", "--lon", "77.61", "--lat", "28.17",
        "--start", "2026-01-01", "--end", "2026-01-02"])

    rc = fs.main()
    assert rc == 2
    err = capsys.readouterr().err
    assert "CDSE_S3_ACCESS_KEY" in err
    assert "CDSE_S3_SECRET_KEY" in err
