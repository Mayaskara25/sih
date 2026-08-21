"""§12 test_harmonize.py -- D9 band arithmetic pinned, D11.3 join, D11.4
sorting, D11.6 self-defending coverage gate.
"""
from pathlib import Path

import numpy as np
import pytest

from core.contracts import SceneMeta
from preprocessing import harmonize
from preprocessing.raster_loader import load_scene

ROOT = Path(__file__).resolve().parents[1]
HAD100_NG_HDR = (ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data"
                 / "aviris_ng_normal" / "ang20191004t185054_13.hdr")
HAD100_CLASSIC_HDR = (ROOT / "data" / "benchmark" / "had100" / "HAD100" / "data"
                       / "aviris_normal" / "f170507t01p00r10_1.hdr")
INDIAN_PINES = ROOT / "data" / "benchmark" / "indian_pines" / "Indian_pines_corrected.mat"

_have_had100 = HAD100_NG_HDR.exists() and HAD100_CLASSIC_HDR.exists()
_have_indian_pines = INDIAN_PINES.exists()

# Transcribed verbatim from scripts/verify_had100.py / D11.3 -- main.py's own
# band_select, five disjoint index slices per sensor.
BAND_SELECT_NG = np.r_[15:109, 118:145, 158:187, 227:274, 328:407]


# --- D9 band arithmetic, pinned ---------------------------------------------

def test_canonical_wl_arithmetic():
    assert len(harmonize.CANONICAL_WL) == 211
    assert harmonize.water_mask().sum() == 27
    assert harmonize.RETAINED_BANDS == 184
    assert int((~harmonize.water_mask()).sum()) == harmonize.RETAINED_BANDS


def test_water_mask_endpoints_inclusive():
    wl = np.array([1349.0, 1350.0, 1450.0, 1451.0, 1799.0, 1800.0, 1950.0, 1951.0],
                  dtype=np.float32)
    mask = harmonize.water_mask(wl)
    np.testing.assert_array_equal(mask, [False, True, True, False, False, True, True, False])


# --- D11.4 sorting -----------------------------------------------------------

@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_sort_spectral_axis_real_classic_header_is_strictly_increasing():
    cube, meta = load_scene(HAD100_CLASSIC_HDR, source="had100")
    assert np.any(np.diff(meta.wavelengths) <= 0), "fixture should be the known non-monotonic case"
    cube_sorted, wl_sorted = harmonize.sort_spectral_axis(cube, meta.wavelengths)
    assert np.all(np.diff(wl_sorted) > 0)
    assert cube_sorted.shape[:2] == cube.shape[:2]


def test_sort_spectral_axis_raises_on_nan_wavelength():
    cube = np.zeros((2, 2, 3), dtype=np.float32)
    wl = np.array([500.0, np.nan, 600.0], dtype=np.float32)
    with pytest.raises(AssertionError):
        harmonize.sort_spectral_axis(cube, wl)


def test_sort_spectral_axis_collapses_exact_duplicates_by_mean():
    # band 0 and band 2 both claim wavelength 500; band 1 is 400.
    cube = np.zeros((1, 1, 3), dtype=np.float32)
    cube[0, 0] = [10.0, 20.0, 30.0]
    wl = np.array([500.0, 400.0, 500.0], dtype=np.float32)
    cube_sorted, wl_sorted = harmonize.sort_spectral_axis(cube, wl)
    assert wl_sorted.tolist() == [400.0, 500.0]
    assert cube_sorted[0, 0, 0] == pytest.approx(20.0)      # the 400nm band, untouched
    assert cube_sorted[0, 0, 1] == pytest.approx(20.0)      # mean(10, 30) at 500nm


# --- interpolation correctness, cross-checked against np.interp ------------

def test_linear_interp_matrix_matches_np_interp():
    rng = np.random.default_rng(0)
    wl_src = np.sort(rng.uniform(400, 900, size=20))
    target = np.linspace(wl_src[0] + 1, wl_src[-1] - 1, 15)
    spectrum = rng.normal(size=20)

    W = harmonize._linear_interp_matrix(target, wl_src)
    got = W @ spectrum
    expected = np.interp(target, wl_src, spectrum)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_interpolate_bands_matches_np_interp():
    """The actual production interpolation path (gather-based, NaN-safe),
    cross-checked against np.interp on NaN-free input where both must agree.
    """
    rng = np.random.default_rng(1)
    wl_src = np.sort(rng.uniform(400, 900, size=20))
    # include values below/above the source range to exercise edge-clamping too
    target = np.concatenate([[wl_src[0] - 5, wl_src[-1] + 5],
                              np.linspace(wl_src[0] + 1, wl_src[-1] - 1, 15)])
    spectrum = rng.normal(size=20)
    cube = spectrum.reshape(1, 1, -1)

    got = harmonize._interpolate_bands(cube, wl_src, target)[0, 0]
    expected = np.interp(target, wl_src, spectrum)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


# --- D11.6: self-defending coverage gate ------------------------------------

@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_coverage_ok_true_for_both_raw_sensors():
    _, meta_ng = load_scene(HAD100_NG_HDR, source="had100")
    _, meta_c = load_scene(HAD100_CLASSIC_HDR, source="had100")
    retained_wl = harmonize.CANONICAL_WL[~harmonize.water_mask()]

    assert harmonize.coverage_ok(meta_ng.wavelengths, retained_wl) is True
    assert harmonize.coverage_ok(meta_c.wavelengths, retained_wl) is True


@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_coverage_ok_false_and_harmonize_raises_on_band_select_style_gap():
    """The self-defending property: feeding harmonize a band_select-style
    gapped axis must RAISE, never emit a plausible-looking, partly-fabricated
    cube (D11.6). Reconstructed from the real NG header: verified 43/184
    canonical bands uncovered, matching the plan's measured figure exactly.
    """
    cube, meta = load_scene(HAD100_NG_HDR, source="had100")
    wl_sorted_idx = np.argsort(meta.wavelengths)   # NG is already monotonic (D11.4)
    gapped_wl = meta.wavelengths[wl_sorted_idx][BAND_SELECT_NG]
    gapped_cube = cube[..., wl_sorted_idx][..., BAND_SELECT_NG]

    retained_wl = harmonize.CANONICAL_WL[~harmonize.water_mask()]
    assert harmonize.coverage_ok(gapped_wl, retained_wl) is False

    from dataclasses import replace
    gapped_meta = replace(meta, wavelengths=gapped_wl,
                           bad_bands=np.zeros(len(gapped_wl), dtype=bool))
    with pytest.raises(ValueError, match="no source wavelength"):
        harmonize.harmonize(gapped_cube, gapped_meta)


def test_coverage_ok_false_for_a_single_interior_hole_isolated_from_truncation():
    """The band_select fixture above has four holes AND truncated endpoints at
    once -- it can't distinguish "caught the hole" from "caught the
    truncation". This isolates the hole: a source axis dense (10nm step)
    everywhere except one clean 200nm gap placed in a retained (non-water)
    region, with full canonical-range coverage otherwise.
    """
    wl_dense = np.arange(400.0, 2500.0, 10.0)
    hole = (wl_dense > 1000) & (wl_dense < 1200)   # 200nm interior gap, no water window here
    assert hole.sum() > 0
    wl_src = wl_dense[~hole]

    retained_wl = harmonize.CANONICAL_WL[~harmonize.water_mask()]
    assert harmonize.coverage_ok(wl_src, retained_wl) is False

    cube = np.ones((1, 1, len(wl_src)), dtype=np.float32)
    meta = _synthetic_meta(wl_src.astype(np.float32))
    with pytest.raises(ValueError, match="no source wavelength"):
        harmonize.harmonize(cube, meta)


def test_coverage_ok_true_for_a_coarse_but_gapless_axis():
    """The inverse case, so the tolerance isn't accidentally too strict:
    a uniformly coarse (40nm step) axis with NO holes must still pass --
    coarse-but-gapless is genuinely interpolable, unlike a hole."""
    wl_src = np.arange(400.0, 2500.0, 40.0)
    retained_wl = harmonize.CANONICAL_WL[~harmonize.water_mask()]
    assert harmonize.coverage_ok(wl_src, retained_wl) is True


# --- D11.3: the join ---------------------------------------------------------

@pytest.mark.skipif(not _have_had100, reason="HAD100 not fetched")
def test_harmonize_ng_and_classic_both_184_and_stack():
    cube_ng, meta_ng = load_scene(HAD100_NG_HDR, source="had100")
    cube_c, meta_c = load_scene(HAD100_CLASSIC_HDR, source="had100")

    out_ng, new_meta_ng = harmonize.harmonize(cube_ng, meta_ng)
    out_c, new_meta_c = harmonize.harmonize(cube_c, meta_c)

    assert out_ng.shape[-1] == harmonize.RETAINED_BANDS
    assert out_c.shape[-1] == harmonize.RETAINED_BANDS
    np.testing.assert_array_equal(new_meta_ng.wavelengths, new_meta_c.wavelengths)

    # D11.3's join: 425-band NG and 224-band Classic patches now stack.
    stacked = np.concatenate(
        [out_ng.reshape(-1, harmonize.RETAINED_BANDS),
         out_c.reshape(-1, harmonize.RETAINED_BANDS)], axis=0)
    assert stacked.shape == (out_ng.shape[0] * out_ng.shape[1]
                              + out_c.shape[0] * out_c.shape[1], harmonize.RETAINED_BANDS)

    # meta.bad_bands is all-False for real HAD100 raw ENVI -- coverage_ok
    # already guarantees this for anything that reaches a returned output.
    assert not new_meta_ng.bad_bands.any()
    assert not new_meta_c.bad_bands.any()
    assert new_meta_ng.crs == meta_ng.crs
    assert new_meta_ng.transform == meta_ng.transform


@pytest.mark.skipif(not _have_indian_pines, reason="Indian Pines not fetched")
def test_harmonize_raises_clearly_when_wavelengths_are_none():
    cube, meta = load_scene(INDIAN_PINES, source="indian_pines")
    assert meta.wavelengths is None    # D13.1 / O8
    with pytest.raises(ValueError, match="wavelengths"):
        harmonize.harmonize(cube, meta)


# --- NaN propagation: per-band, not assumed whole-pixel --------------------

def _synthetic_meta(wl: np.ndarray) -> SceneMeta:
    import affine
    import rasterio.crs

    return SceneMeta(
        scene_id="synthetic", crs=rasterio.crs.CRS.from_epsg(32615),
        transform=affine.Affine(5.0, 0, 400_000.0, 0, -5.0, 3_000_000.0),
        wavelengths=wl, bad_bands=np.zeros(len(wl), dtype=bool),
        gsd_m=5.0, source="had100", georef="real",
    )


def test_harmonize_nan_propagates_only_to_target_bands_touching_the_nan_source_band():
    """Real HAD100 data currently has zero nodata pixels of either shape
    (verified: all 616 raw scenes, no pixel matches its declared sentinel at
    float32 precision) -- so this is a synthetic check that harmonize()
    handles a PER-BAND nodata pixel correctly rather than assuming nodata is
    always whole-pixel, which nothing in the files currently proves either way.
    """
    # Offset +3nm from the canonical grid so every retained target genuinely
    # blends two source bands -- an exactly-aligned grid resolves every
    # target via a single-tap exact match, which would make this test pass
    # vacuously (no band ever shares a bracket with its neighbour).
    wl = np.arange(403.0, 2500.0, 10.0, dtype=np.float32)
    cube = np.ones((1, 1, len(wl)), dtype=np.float32)
    poison_idx = 59   # source wl = 993nm -- well inside canonical range, no water window
    cube[0, 0, poison_idx] = np.nan   # exactly one bad band; neighbours stay valid

    meta = _synthetic_meta(wl)
    out, _new_meta = harmonize.harmonize(cube, meta)

    retained_wl = harmonize.CANONICAL_WL[~harmonize.water_mask()]
    W = harmonize._linear_interp_matrix(retained_wl, np.sort(wl.astype(np.float64)))
    touches_poison = W[:, poison_idx] != 0

    nan_out = np.isnan(out[0, 0])
    assert nan_out.any(), "expected at least one NaN target band"
    assert not nan_out.all(), "expected NOT every target band to go NaN (would mean whole-pixel assumption)"
    np.testing.assert_array_equal(nan_out, touches_poison)


# --- reduce_bands (§3B.3): PCA/kPCA, fit_on-driven refit-vs-transform-only --

def _rng_cube(seed, h=6, w=6, b=harmonize.RETAINED_BANDS):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(h, w, b)).astype(np.float32)


def test_reduce_bands_output_shape():
    cube = _rng_cube(0)
    out, transformer = harmonize.reduce_bands(cube, n_components=5)
    assert out.shape == (6, 6, 5)
    assert out.dtype == np.float32
    assert hasattr(transformer, "transform")


def test_reduce_bands_fit_on_external_pixel_pool():
    fit_pool = np.random.default_rng(1).normal(
        size=(500, harmonize.RETAINED_BANDS)).astype(np.float32)
    cube = _rng_cube(2)
    out, transformer = harmonize.reduce_bands(cube, n_components=5, fit_on=fit_pool)
    assert out.shape == (6, 6, 5)
    np.testing.assert_allclose(transformer.mean_, fit_pool.mean(axis=0), rtol=1e-4)


def test_reduce_bands_with_prefit_transformer_does_not_refit():
    """fit_on=<already-fitted transformer> must transform-only. Verified by
    fitting once on pool A, then calling reduce_bands with a DIFFERENT pool
    B's cube passed as fit_on's ndarray-branch companion -- if it silently
    refit, the transformer's mean_ would shift to reflect the new cube."""
    pool_a = np.random.default_rng(3).normal(
        loc=0.0, size=(500, harmonize.RETAINED_BANDS)).astype(np.float32)
    _out_a, transformer = harmonize.reduce_bands(_rng_cube(4), n_components=5, fit_on=pool_a)
    mean_after_first_fit = transformer.mean_.copy()

    cube_b = np.random.default_rng(5).normal(
        loc=1000.0, size=(6, 6, harmonize.RETAINED_BANDS)).astype(np.float32)
    _out_b, transformer_returned = harmonize.reduce_bands(
        cube_b, n_components=5, fit_on=transformer)

    assert transformer_returned is transformer
    np.testing.assert_array_equal(transformer.mean_, mean_after_first_fit)


def test_reduce_bands_nan_pixels_stay_nan_and_transformer_never_sees_them():
    fit_pool = np.random.default_rng(6).normal(
        size=(500, harmonize.RETAINED_BANDS)).astype(np.float32)
    cube = _rng_cube(7)
    cube[2, 3, :] = np.nan
    out, _transformer = harmonize.reduce_bands(cube, n_components=5, fit_on=fit_pool)
    assert np.all(np.isnan(out[2, 3]))
    assert not np.any(np.isnan(np.delete(out.reshape(-1, 5), 2 * 6 + 3, axis=0)))


def test_reduce_bands_reconstruction_error_under_2_percent_on_held_out_pixels():
    """§3A.1's original accept criterion (moved to §3B.3, D15): a fit/transform
    round-trip on held-out pixels reconstructs with < 2% mean relative error
    at n_components=30. Structured (low-rank + small noise) data, since truly
    random 184-band noise has no 30-dim structure to recover."""
    rng = np.random.default_rng(8)
    n, b, rank = 4000, harmonize.RETAINED_BANDS, 20
    basis = rng.normal(size=(rank, b))
    loadings = rng.normal(size=(n, rank))
    data = (loadings @ basis).astype(np.float32) + rng.normal(scale=0.01, size=(n, b)).astype(np.float32)

    fit_data, held_out = data[:3000], data[3000:]
    cube = held_out.reshape(1, -1, b)
    reduced, transformer = harmonize.reduce_bands(cube, n_components=30, fit_on=fit_data)

    reconstructed = transformer.inverse_transform(reduced.reshape(-1, 30))
    rel_error = np.abs(reconstructed - held_out) / (np.abs(held_out) + 1e-8)
    assert rel_error.mean() < 0.02


def test_reduce_bands_rejects_unknown_method():
    with pytest.raises(ValueError):
        harmonize.reduce_bands(_rng_cube(9), n_components=5, method="bogus")
