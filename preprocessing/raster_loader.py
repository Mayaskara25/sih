"""Dispatch-on-extension scene loader (PLAN.md §2.1).

.mat  -> scipy.io.loadmat; NO georeferencing and NO wavelengths exist in any
         .mat we use (D13.1/D13.2/D13.3), so a synthetic affine is attached
         per D2, meta.georef == "synthetic" and meta.wavelengths is None.
.tif  -> rasterio; real CRS/transform read from the file, meta.georef == "real".
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
"""
from __future__ import annotations

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


def _load_tif(path: Path, *, source: str) -> tuple[np.ndarray, SceneMeta]:
    with rasterio.open(path) as ds:
        raw = ds.read()                              # [B, H, W]
        cube = cast_to_float32(raw, source_dtype=raw.dtype)
        cube = np.ascontiguousarray(np.moveaxis(cube, 0, -1))   # -> [H, W, B]
        if ds.nodata is not None:
            cube = np.where(cube == np.float32(ds.nodata), np.nan, cube)
        crs, transform = ds.crs, ds.transform
    b = cube.shape[-1]
    meta = SceneMeta(
        scene_id=path.stem,
        crs=crs,
        transform=transform,
        wavelengths=None,
        bad_bands=np.zeros(b, dtype=bool),
        gsd_m=abs(transform.a),
        source=source,
        georef="real",
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
