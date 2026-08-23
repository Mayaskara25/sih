"""Dispatch-on-extension scene loader (PLAN.md §2.1).

.mat  -> scipy.io.loadmat; NO georeferencing and NO wavelengths exist in any
         .mat we use (D13.1/D13.2/D13.3), so a synthetic affine is attached
         per D2, meta.georef == "synthetic" and meta.wavelengths is None.
.tif  -> rasterio; real CRS/transform read from the file, meta.georef == "real".
         An EnMAP L2A product (source="enmap", filename ending
         "-SPECTRAL_IMAGE_COG.TIF") is dispatched to the same path PLUS a
         METADATA.XML sidecar parse for wavelengths -- see below. A Sentinel-2
         AOI stack (source="sentinel2", filename ending "_stack.tif", written
         by scripts/fetch_sentinel2.py) is dispatched to the same path PLUS a
         GDAL-tag parse for wavelengths/acquired -- see
         `_load_sentinel2_tags`, no sidecar file involved. Any other .tif
         (including every existing fixture/test) is unaffected: both lookups
         only fire on their exact (source, filename suffix) combination.
.hdr  -> spectral.envi for the cube + wavelengths; GDAL's ENVI driver (via
         rasterio, opened on the sibling data file) for CRS/transform/nodata,
         so meta.georef == "real" (D11.5 -- HAD100 ENVI scenes are genuinely
         georeferenced, including rotation, which GDAL handles and a
         hand-rolled 'map info' parse would not).

This module does NOT sort or otherwise fix a non-ascending wavelength axis --
AVIRIS-Classic wavelength arrays are non-monotonic in the raw header (D11.4)
and sorting is `preprocessing/harmonize.py`'s job (D9), not the loader's. A
raw wavelength array is handed to SceneMeta exactly as parsed; validate_scene
correctly rejects a non-ascending one until it has been harmonized.

EnMAP wavelengths are NOT in the TIFF (verified: no per-band description tags
in the COG -- PLAN.md D32) -- they live in the sibling
METADATA.XML/.XML.XML sidecar's <bandCharacterisation>/<bandID> elements.
`_load_enmap_wavelengths` parses that sidecar and is INTENTIONALLY STRICT: a
missing sidecar, unparseable XML, a band count mismatch against the cube, or
a non-finite/non-strictly-ascending wavelength axis all raise rather than
falling back to `wavelengths=None` -- a real EnMAP product silently entering
the pipeline with no wavelength array would pass every existing contract
check and then fail `harmonize()` (D9) confusingly, far from the cause.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from pathlib import Path

import affine
import numpy as np
import rasterio
import rasterio.crs
from scipy.io import loadmat

from core.contracts import SceneMeta

SYNTHETIC_AFFINE_ORIGIN = (500_000.0, 4_480_000.0)   # UTM 16N, NW Indiana
SYNTHETIC_CRS = "EPSG:32616"
SYNTHETIC_GSD_M = 20.0

# Variable name inside each source's .mat file. ABU and HYDICE both ship
# ('data', 'map') -- verified against every ABU scene and HYDICE-urban.mat
# (D13.2/D13.3). Indian Pines ships one array under its own name (D13.1).
_MAT_CUBE_KEY = {
    "indian_pines": "indian_pines_corrected",
    "abu": "data",
    "hydice_urban_anomaly": "data",
}

_ALLOWED_SOURCE_DTYPES = (
    np.dtype(np.int16), np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64),
)

def cast_to_float32(raw: np.ndarray, *, source_dtype: np.dtype) -> np.ndarray:
    """Explicit, dtype-aware widening to float32. NEVER `raw.astype(np.float32)`
    on an array whose dtype was assumed.

    The real hazard is reading SIGNED data as UNSIGNED. Verified in ABU: 8 of
    13 scenes are int16 and genuinely contain small negatives (min -50..-1, up
    to 45626 negative pixels in abu-urban-2) -- normal residuals after
    dark-current/atmospheric correction. Reinterpreted as uint16, -1 becomes
    65535: a value 3x the scene maximum, fed straight into a detector whose
    entire job is to flag extreme values.
    """
    if raw.dtype not in _ALLOWED_SOURCE_DTYPES:
        raise ValueError(f"unhandled source dtype {raw.dtype}")
    assert raw.dtype == source_dtype, "dtype changed between header and read"
    return np.ascontiguousarray(raw.astype(np.float32))


def _synthetic_affine() -> tuple[rasterio.crs.CRS, affine.Affine]:
    ox, oy = SYNTHETIC_AFFINE_ORIGIN
    transform = affine.Affine(SYNTHETIC_GSD_M, 0, ox, 0, -SYNTHETIC_GSD_M, oy)
    return rasterio.crs.CRS.from_string(SYNTHETIC_CRS), transform


def _load_mat(path: Path, *, source: str) -> tuple[np.ndarray, SceneMeta]:
    key = _MAT_CUBE_KEY.get(source)
    if key is None:
        raise ValueError(f"no known cube variable name for source={source!r}")
    mat = loadmat(path)
    if key not in mat:
        raise ValueError(f"{path}: expected variable {key!r}, found {sorted(mat)}")
    raw = mat[key]
    cube = cast_to_float32(raw, source_dtype=raw.dtype)
    crs, transform = _synthetic_affine()
    b = cube.shape[-1]
    meta = SceneMeta(
        scene_id=path.stem,
        crs=crs,
        transform=transform,
        wavelengths=None,
        bad_bands=np.zeros(b, dtype=bool),
        gsd_m=SYNTHETIC_GSD_M,
        source=source,
        georef="synthetic",
    )
    return cube, meta


_ENMAP_SPECTRAL_SUFFIX = "-SPECTRAL_IMAGE_COG.TIF"


def _find_enmap_metadata_sidecar(spectral_path: Path) -> Path:
    """Locate the METADATA.XML sidecar for an EnMAP `*-SPECTRAL_IMAGE_COG.TIF`.

    Filenames on disk are inconsistent -- some products end `-METADATA.XML`,
    others `-METADATA.XML.XML` (verified across the 8 products in
    data/raw/enmap/: both forms occur). Both are tried; neither existing
    raises FileNotFoundError rather than silently returning wavelengths=None.
    """
    if not spectral_path.name.endswith(_ENMAP_SPECTRAL_SUFFIX):
        raise ValueError(f"{spectral_path}: not an EnMAP SPECTRAL_IMAGE_COG filename")
    stem = spectral_path.name[: -len(_ENMAP_SPECTRAL_SUFFIX)]
    for suffix in ("-METADATA.XML.XML", "-METADATA.XML"):
        candidate = spectral_path.with_name(stem + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{spectral_path}: no METADATA.XML(.XML) sidecar found beside this product -- "
        "EnMAP wavelengths cannot be recovered without it")


def _load_enmap_wavelengths(metadata_path: Path) -> tuple[np.ndarray, str | None]:
    """Parse <bandCharacterisation>/<bandID> from an EnMAP L2A METADATA.XML.

    Returns (wavelengths [B] float32 nm, acquired ISO-8601 str or None).
    Ordered by the XML's own `<bandID number="...">` attribute, not file
    order, then validated: empty, non-finite, or non-strictly-ascending all
    raise (D11.4 -- a poisoned axis must fail here, not silently reach
    validate_scene/harmonize downstream).

    Does NOT read GainOfBand/OffsetOfBand (the reflectance scale factor) --
    SceneMeta has no field for it and this loader does not rescale the cube
    (see docs/enmap_verified.json / PLAN.md D32 for the value found and why
    it is recorded but not applied).
    """
    try:
        root = ET.parse(metadata_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{metadata_path}: unparseable XML ({exc})") from exc

    # Scoped to <bandCharacterisation> specifically -- the schema reuses
    # <bandID number="..."> elsewhere (e.g. per-band defective-pixel/artifact
    # statistics) with DIFFERENT children and no wavelengthCenterOfBand.
    # root.iter("bandID") over the whole document would pick those up too;
    # scoping avoids relying on the "no wavelengthCenterOfBand -> skip" guard
    # below to do that filtering silently.
    band_char = root.find(".//bandCharacterisation")
    entries: list[tuple[int, float]] = []
    for band in (band_char.iter("bandID") if band_char is not None else ()):
        num = band.get("number")
        wl_text = band.findtext("wavelengthCenterOfBand")
        if num is None or wl_text is None:
            continue
        entries.append((int(num), float(wl_text)))
    if not entries:
        raise ValueError(
            f"{metadata_path}: no <bandCharacterisation>/<bandID>/"
            "<wavelengthCenterOfBand> entries found -- unparseable or wrong schema")
    entries.sort(key=lambda e: e[0])
    wavelengths = np.array([wl for _, wl in entries], dtype=np.float32)

    if not np.all(np.isfinite(wavelengths)):
        raise ValueError(f"{metadata_path}: wavelength array contains non-finite values")
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError(
            f"{metadata_path}: wavelength array is not strictly ascending in bandID order")

    acquired = root.findtext(".//startTime")
    return wavelengths, acquired


_S2_STACK_SUFFIX = "_stack.tif"


def _load_sentinel2_tags(tags: dict, band_count: int) -> tuple[np.ndarray, str]:
    """Parse the WAVELENGTHS_NM / SENSING_TIME GDAL tags that
    `scripts/fetch_sentinel2.py` bakes into every `*_stack.tif` it writes.

    Unlike EnMAP, Sentinel-2 has no sidecar file to parse here: the fetcher
    already read MTD_MSIL2A.xml (per-product Spectral_Information -- centre
    wavelength varies by S2A/S2B/S2C platform, verified 2026-08-23: S2A's
    B12 centre is 2202.4 nm vs S2B's 2185.7 nm on the same tile) and
    MTD_TL.xml (per-tile SENSING_TIME, the correct acquisition-time source --
    NOT MTD_MSIL2A.xml's PRODUCT_START_TIME, which is the whole DATATAKE's
    start and was measured to differ from SENSING_TIME by ~12-14 minutes) at
    download time, and recorded both directly as tags on the GeoTIFF -- the
    same mechanism `save_score_raster` already uses for its own tags.

    Strict by construction, mirroring `_load_enmap_wavelengths` (D32): a
    missing tag, a band-count mismatch, or a non-finite/non-ascending
    wavelength axis all raise rather than silently returning None. D33 was
    exactly the bug a silently-empty `acquired` field caused; this must not
    reintroduce it for the one branch (Phase 5 Level 3) where `acquired`
    is load-bearing for every downstream comparison.
    """
    wl_text = tags.get("WAVELENGTHS_NM")
    if not wl_text:
        raise ValueError(
            "no WAVELENGTHS_NM tag -- not a *_stack.tif written by "
            "scripts/fetch_sentinel2.py, or the tag was stripped")
    wavelengths = np.array([float(x) for x in wl_text.split(",")], dtype=np.float32)
    if wavelengths.shape != (band_count,):
        raise ValueError(
            f"WAVELENGTHS_NM tag has {wavelengths.shape[0]} values, cube has "
            f"{band_count} bands -- refusing to attach a mismatched wavelength array")
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("WAVELENGTHS_NM tag contains non-finite values")
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("WAVELENGTHS_NM tag is not strictly ascending")

    acquired = tags.get("SENSING_TIME")
    if not acquired:
        raise ValueError(
            "no SENSING_TIME tag -- meta.acquired would be None, reintroducing "
            "D33 for the one branch (Level 3) where it is load-bearing")
    return wavelengths, acquired


def _load_tif(path: Path, *, source: str) -> tuple[np.ndarray, SceneMeta]:
    with rasterio.open(path) as ds:
        raw = ds.read()                              # [B, H, W]
        cube = cast_to_float32(raw, source_dtype=raw.dtype)
        cube = np.ascontiguousarray(np.moveaxis(cube, 0, -1))   # -> [H, W, B]
        if ds.nodata is not None:
            cube = np.where(cube == np.float32(ds.nodata), np.nan, cube)
        crs, transform = ds.crs, ds.transform
        tags = ds.tags()          # cheap; only used by the sentinel2 branch below
    b = cube.shape[-1]

    # A band that is nodata for EVERY pixel in the scene carries zero
    # information -- and worse, it poisons every OTHER band's per-pixel
    # statistics downstream: preprocessing.normalize.standardize's per-band
    # nanmean/nanvar returns NaN for that one band at every pixel, and
    # anomaly.rx.global_rx's `~np.any(np.isnan(flat), axis=-1)` validity mask
    # then excludes EVERY pixel in the scene, not just that band, because
    # every pixel has a NaN in this one band. Verified concretely: all 8
    # EnMAP L2A products on disk have bands 131-135 (~1342.8-1390.5 nm, the
    # edge of the 1350-1450 nm water-absorption window) 100% nodata,
    # constant across every scene (PLAN.md D32) -- this is not hardcoded to
    # those indices, since a different product/acquisition could differ;
    # it is re-derived from the actual pixels on every load.
    fully_nodata_bands = np.zeros(b, dtype=bool)
    if np.isnan(cube).any():
        fully_nodata_bands = np.all(np.isnan(cube.reshape(-1, b)), axis=0)

    wavelengths = None
    acquired = None
    if source == "enmap" and path.name.endswith(_ENMAP_SPECTRAL_SUFFIX):
        metadata_path = _find_enmap_metadata_sidecar(path)
        wavelengths, acquired = _load_enmap_wavelengths(metadata_path)
        if wavelengths.shape != (b,):
            raise ValueError(
                f"{path}: METADATA.XML sidecar has {wavelengths.shape[0]} bands, "
                f"cube has {b} -- refusing to attach a mismatched wavelength array")
    elif source == "sentinel2" and path.name.endswith(_S2_STACK_SUFFIX):
        wavelengths, acquired = _load_sentinel2_tags(tags, b)

    meta = SceneMeta(
        scene_id=path.stem,
        crs=crs,
        transform=transform,
        wavelengths=wavelengths,
        bad_bands=fully_nodata_bands,
        gsd_m=abs(transform.a),
        source=source,
        georef="real",
        acquired=acquired,
    )
    return cube, meta


def _load_envi(path: Path, *, source: str) -> tuple[np.ndarray, SceneMeta]:
    """CRS/transform/nodata come from GDAL's own ENVI driver (opened on the
    sibling data file spectral resolves via `img.filename`), not a hand-rolled
    parse of 'map info'. That field carries an optional rotation, and D11.5's
    real HAD100 scenes are, in practice, always rotated (verified: 33 deg on
    an AVIRIS-NG scene, 90 deg on an AVIRIS-Classic one) -- an axis-aligned
    affine built by hand from tie-point + pixel-size alone is measurably wrong
    for these files (confirmed by diffing against GDAL's parsed transform).
    GDAL's ENVI driver already implements the rotation correctly, so this
    defers to it instead of re-deriving unverified rotation math.
    """
    import spectral.io.envi as envi

    img = envi.open(str(path))
    raw = np.asarray(img.load())
    cube = cast_to_float32(raw, source_dtype=raw.dtype)

    with rasterio.open(img.filename) as ds:
        crs, transform, nodata = ds.crs, ds.transform, ds.nodata

    if nodata is not None:
        cube = np.where(cube == np.float32(nodata), np.nan, cube)

    centers = img.bands.centers
    wavelengths = np.array(centers, dtype=np.float32) if centers else None

    b = cube.shape[-1]
    meta = SceneMeta(
        scene_id=path.stem,
        crs=crs,
        transform=transform,
        wavelengths=wavelengths,
        bad_bands=np.zeros(b, dtype=bool),
        # sqrt(a^2+b^2) is the true column pixel width under rotation; this
        # assumes square pixels (px == py), true of every HAD100 header seen
        # so far but not asserted -- a non-square-pixel sensor would need
        # gsd_m to carry (x, y) separately.
        gsd_m=float(np.hypot(transform.a, transform.b)),
        source=source,
        georef="real",
    )
    return cube, meta


def load_scene(path: str | Path, *, source: str) -> tuple[np.ndarray, SceneMeta]:
    """Dispatch on extension. Returns C1-compliant (cube [H,W,B] float32, meta).

    DTYPE IS READ, NEVER ASSUMED -- meta records nothing about source dtype
    directly (that lives in the raw array before cast_to_float32 runs); the
    cast itself refuses to guess.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".mat":
        return _load_mat(path, source=source)
    if ext in (".tif", ".tiff"):
        return _load_tif(path, source=source)
    if ext == ".hdr":
        return _load_envi(path, source=source)
    raise ValueError(f"unhandled extension {ext!r} for {path}")


def load_sentinel2_scl(stack_path: str | Path) -> np.ndarray:
    """Load the SCL sibling of a Sentinel-2 `*_stack.tif` written by
    `scripts/fetch_sentinel2.py` (`*_scl.tif`, same AOI window and grid --
    verified byte-for-byte same transform/CRS as the stack by
    `scripts/verify_sentinel2.py`'s `scl_grid_matches_stack` check).

    Returns uint8 [H,W], native SCL class values (0-11, ESA PSD-15) -- for
    `preprocessing.cloud_mask.cloud_shadow_mask`'s `scl=` argument. That
    function REQUIRES an SCL array whenever `meta.source == "sentinel2"` and
    raises rather than falling back to spectral thresholds tuned for
    hyperspectral sensors (PLAN.md §3C.7) -- SCL is not baked into the cube
    `load_scene` returns, so callers must load it separately with this
    function rather than slicing a 7th band off the cube.
    """
    stack_path = Path(stack_path)
    scl_path = stack_path.with_name(stack_path.name.replace("_stack.tif", "_scl.tif"))
    if not scl_path.exists():
        raise FileNotFoundError(f"{stack_path}: no matching SCL file {scl_path.name}")
    with rasterio.open(scl_path) as ds:
        scl = ds.read(1)
    return scl.astype(np.uint8)


def save_score_raster(score: np.ndarray, meta: SceneMeta, out_path: str | Path,
                       *, method: str, normalize: bool = True) -> tuple[Path, Path]:
    """Writes the C2 raw/norm pair with all required tags. Returns both paths.

    `out_path` is the naming base (no extension), e.g. `.../{scene_id}_anom`;
    this writes `{out_path}_raw.tif` and, if normalize, `{out_path}_norm.tif`.
    """
    out_path = Path(out_path)
    raw_path = out_path.with_name(out_path.name + "_raw.tif")
    norm_path = out_path.with_name(out_path.name + "_norm.tif")

    score = score.astype(np.float32)
    profile = dict(
        driver="GTiff", height=score.shape[0], width=score.shape[1],
        count=1, dtype="float32", crs=meta.crs, transform=meta.transform,
        nodata=np.nan,
    )

    with rasterio.open(raw_path, "w", **profile) as ds:
        ds.write(score, 1)
        ds.update_tags(SCORE_METHOD=method, SCENE_ID=meta.scene_id, GEOREF=meta.georef)

    if normalize:
        from anomaly.scoring import percentile_normalize

        norm, v_lo, v_hi = percentile_normalize(score)
        with rasterio.open(norm_path, "w", **profile) as ds:
            ds.write(norm.astype(np.float32), 1)
            ds.update_tags(
                NORM_METHOD="percentile_clip", NORM_P_LO="1.0", NORM_P_HI="99.9",
                NORM_V_LO=repr(float(v_lo)), NORM_V_HI=repr(float(v_hi)),
                SCORE_METHOD=method, SCENE_ID=meta.scene_id, GEOREF=meta.georef,
            )

    return raw_path, norm_path
