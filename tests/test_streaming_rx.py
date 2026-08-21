"""PLAN.md §3A.5 -- streaming RX correctness, memory-bound and interface tests.

Read `anomaly/streaming_rx.py`'s module docstring in full before touching this
file; it records two deliberate deviations from a literal reading of the
spec, and the tests below are written to match them rather than fight them:

1. STREAMING IS FORMAT-DEPENDENT. Only `.tif`/`.hdr` are genuinely streamed
   (windowed reads off disk); `.mat` (Indian Pines, ABU -- this project's
   only local real benchmark scenes) has no partial-read API in
   `scipy.io.loadmat`, so it pays one unavoidable full read no matter what
   this module does. The peak-RSS accept criterion is therefore tested on a
   synthetic `.tif` fixture, NOT on Indian Pines or ABU -- asserting an RSS
   bound on `.mat` would be asserting something the module correctly says
   is impossible.
2. SPEC DISCREPANCY: `StreamingCovariance.cov` is ddof=1 (unbiased) as
   specced, but `streaming_rx()` itself scores using the ddof=0 (biased)
   covariance recovered from the same accumulator's raw `_m2`, to match
   `global_rx`'s `centered.T @ centered / N` convention exactly. Both facts
   are tested explicitly below, separately.

EXECUTION-VERIFIED EXTENSION OF THE MODULE'S "SECOND, LARGER SPEC
DISCREPANCY" NOTE -- worth recording here since it is not in the module
docstring. That note measures `streaming_rx` vs `global_rx` on Indian Pines
at ~8.6e-4 max relative difference (float32 imprecision in `global_rx`
itself, not a `streaming_rx` bug -- verified there against an independent
float64 reference). The same experiment run here on ABU (native radiance,
D22's regime) measures a MUCH larger gap: max relative difference ~4.3e-2
on `abu-urban-3` (up to ~2.9e-2 / ~1.7e-2 on `abu-airport-1` /
`abu-beach-2`), i.e. ~50x looser than the Indian Pines figure and ~4300x
looser than the literal `rtol=1e-5` spec. Root-caused the same way: an
independent float64 recomputation of `global_rx`'s own formula (upcasting
the cube to float64 before any arithmetic, otherwise identical) matches
`streaming_rx` on `abu-urban-3` to `rtol=1e-5` easily (measured max relative
difference ~5.9e-8) -- so the ~4.3e-2 gap against the real (float32)
`global_rx` is entirely `global_rx`'s float32 imprecision, amplified by
ABU's larger band count / condition number / native radiance scale relative
to Indian Pines, not a `streaming_rx` defect. `test_streaming_rx_matches_
global_rx_on_abu` below uses a tolerance wide enough to cover the measured
gap, documented at the assertion.

No source file other than this one was modified. `anomaly/streaming_rx.py`
was read only.
"""
from __future__ import annotations

import gc
import inspect
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import rasterio

from anomaly.rx import global_rx
from anomaly.streaming_rx import StreamingCovariance, streaming_rx
from preprocessing.raster_loader import load_scene

ROOT = Path(__file__).resolve().parents[1]
INDIAN_PINES = ROOT / "data" / "benchmark" / "indian_pines" / "Indian_pines_corrected.mat"
ABU_SCENE = ROOT / "data" / "benchmark" / "abu" / "abu-urban-3.mat"
_have_indian_pines = INDIAN_PINES.exists()
_have_abu = ABU_SCENE.exists()


def _write_tif(path: Path, cube: np.ndarray, *, nodata: float | None = None) -> None:
    """`cube` is `[H, W, B]` float32, matching this project's C1 cube
    contract. Same construction pattern as `tests/test_loader.py`'s
    `test_tif_dispatch_reads_real_crs_and_transform`.
    """
    h, w, b = cube.shape
    transform = rasterio.Affine(10.0, 0, 100.0, 0, -10.0, 200.0)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=b, dtype="float32",
        crs="EPSG:32616", transform=transform, nodata=nodata,
    ) as ds:
        ds.write(np.moveaxis(cube, -1, 0))


# =============================================================================
# StreamingCovariance -- the accumulator itself
# =============================================================================

def test_incremental_updates_equal_one_bulk_update():
    """The core correctness property: several small `update()` calls must
    accumulate to exactly (within float64 rounding) the same mean/cov as one
    `update()` over the whole array, and both must match an independent
    `np.cov`/`.mean()` reference.
    """
    rng = np.random.default_rng(0)
    n, b = 5000, 12
    data = rng.normal(loc=3.0, scale=2.0, size=(n, b))

    bulk = StreamingCovariance(n_bands=b)
    bulk.update(data)

    incremental = StreamingCovariance(n_bands=b)
    chunk = 137  # deliberately not a divisor of n, exercises a ragged last chunk
    for i in range(0, n, chunk):
        incremental.update(data[i:i + chunk])

    ref_mean = data.mean(axis=0)
    ref_cov = np.cov(data.T, ddof=1)

    np.testing.assert_allclose(bulk.mean, incremental.mean, rtol=1e-10)
    np.testing.assert_allclose(bulk.cov, incremental.cov, rtol=1e-10)
    np.testing.assert_allclose(bulk.mean, ref_mean, rtol=1e-10)
    np.testing.assert_allclose(bulk.cov, ref_cov, rtol=1e-8)
    assert incremental.n == n


def test_update_accepts_3d_strip_and_2d_flat_equivalently():
    """`update()` accepts `[rows, W, B]` (a raster strip) or `[N, B]`
    (already flattened), per the §3A.5 spec comment. A strip and its
    row-major flattening must accumulate identically.
    """
    rng = np.random.default_rng(1)
    rows, w, b = 20, 7, 6
    strip = rng.normal(size=(rows, w, b))
    flat = strip.reshape(-1, b)

    acc_3d = StreamingCovariance(n_bands=b)
    acc_3d.update(strip)

    acc_2d = StreamingCovariance(n_bands=b)
    acc_2d.update(flat)

    assert acc_3d.n == acc_2d.n == rows * w
    np.testing.assert_allclose(acc_3d.mean, acc_2d.mean, rtol=1e-12)
    np.testing.assert_allclose(acc_3d.cov, acc_2d.cov, rtol=1e-12)


def test_update_rejects_wrong_band_count_and_bad_ndim():
    acc = StreamingCovariance(n_bands=4)
    with pytest.raises(ValueError):
        acc.update(np.zeros((10, 3)))          # wrong band count
    with pytest.raises(ValueError):
        acc.update(np.zeros((10,)))             # ndim=1, neither [rows,W,B] nor [N,B]
    with pytest.raises(ValueError):
        acc.update(np.zeros((2, 3, 4, 4)))      # ndim=4


def test_cov_is_ddof1_against_np_cov():
    """Spec: `StreamingCovariance.cov` is `[B, B]`, `ddof=1`. Tested
    directly against `np.cov(..., ddof=1)`, not against `ddof=0` -- see the
    module docstring's "SPEC DISCREPANCY" note for why `streaming_rx`
    itself does NOT go through this property for RX scoring.
    """
    rng = np.random.default_rng(2)
    n, b = 401, 9  # odd, small N so the ddof=1 vs ddof=0 factor (N/(N-1)) is not negligible
    data = rng.normal(size=(n, b))

    acc = StreamingCovariance(n_bands=b)
    acc.update(data[:200])
    acc.update(data[200:])

    ref_ddof1 = np.cov(data.T, ddof=1)
    ref_ddof0 = np.cov(data.T, ddof=0)

    np.testing.assert_allclose(acc.cov, ref_ddof1, rtol=1e-10)
    # Sanity: ddof=1 and ddof=0 genuinely differ at this N, so the assertion
    # above is discriminating, not vacuous.
    assert not np.allclose(acc.cov, ref_ddof0, rtol=1e-6)


def test_cov_requires_at_least_two_samples():
    acc = StreamingCovariance(n_bands=3)
    with pytest.raises(ValueError):
        _ = acc.cov
    acc.update(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        _ = acc.cov


def test_mean_requires_at_least_one_sample():
    acc = StreamingCovariance(n_bands=3)
    with pytest.raises(ValueError):
        _ = acc.mean


def test_nan_pixels_excluded_from_accumulation():
    """A pixel with ANY NaN band must be excluded entirely from the running
    mean/co-moment, not propagated -- see the module's D15 cross-reference
    in `StreamingCovariance.update`'s docstring.
    """
    rng = np.random.default_rng(3)
    rows, w, b = 15, 6, 5
    strip = rng.normal(loc=1.0, scale=1.0, size=(rows, w, b))
    flat = strip.reshape(-1, b)
    nan_idx = [0, 7, 22, 44, 89]
    flat_with_nan = flat.copy()
    for i in nan_idx:
        flat_with_nan[i, 2] = np.nan   # a single NaN band poisons the whole pixel

    acc = StreamingCovariance(n_bands=b)
    acc.update(flat_with_nan.reshape(rows, w, b))

    keep = np.ones(flat.shape[0], dtype=bool)
    keep[nan_idx] = False
    ref_mean = flat[keep].mean(axis=0)
    ref_cov = np.cov(flat[keep].T, ddof=1)

    assert acc.n == keep.sum()
    np.testing.assert_allclose(acc.mean, ref_mean, rtol=1e-10)
    np.testing.assert_allclose(acc.cov, ref_cov, rtol=1e-8)


def test_all_nan_strip_is_a_no_op():
    acc = StreamingCovariance(n_bands=4)
    acc.update(np.full((3, 3, 4), np.nan))
    assert acc.n == 0
    acc.update(np.ones((2, 4)))
    assert acc.n == 2  # the all-NaN strip contributed nothing


def test_float64_accumulation_is_required_for_large_mean_offset_precision():
    """Accept criterion: "accumulation in float64 -- float32 co-moment
    accumulation loses precision over 20 000+ pixels and quietly biases the
    covariance." This is a test of the ACCUMULATOR's dtype (the running
    mean/co-moment across many merged strips), not merely storage -- so it
    must be discriminating in a way `dtype == np.float64` is not.

    Construction: values cluster tightly (std=1e-3) around a large offset
    (1e5). float32 has ~7.2 decimal digits of precision, so representing
    ~1e5 leaves only ~1e-2 absolute resolution -- an order of magnitude
    coarser than the std we're trying to measure. Accumulating in float32
    (`StreamingCovariance(n_bands, dtype=np.float32)`, exercising the class's
    own `dtype` parameter to stand in for "someone changed the default")
    must produce a covariance that is grossly wrong, while the DEFAULT
    construction (no `dtype` kwarg -- what `streaming_rx` actually uses)
    must stay tight against an independent float64 reference. If the
    module's default were ever changed to float32, the first assertion
    below (using the default constructor) would fail.
    """
    rng = np.random.default_rng(4)
    offset, std = 1.0e5, 1.0e-3
    n_bands, n_total, chunk = 5, 24_000, 400
    data = rng.normal(loc=offset, scale=std, size=(n_total, n_bands))

    ref_mean = data.mean(axis=0)
    ref_cov = np.cov(data.T, ddof=1)

    acc_default = StreamingCovariance(n_bands=n_bands)   # no dtype kwarg -- what streaming_rx uses
    acc_f32 = StreamingCovariance(n_bands=n_bands, dtype=np.float32)
    for i in range(0, n_total, chunk):
        c = data[i:i + chunk]
        acc_default.update(c)
        acc_f32.update(c)

    # The actual accept-criterion assertion: default (float64) accumulation
    # stays tight against the true statistics despite the large offset.
    np.testing.assert_allclose(acc_default.mean, ref_mean, rtol=1e-6)
    np.testing.assert_allclose(acc_default.cov, ref_cov, rtol=1e-4)

    # Proof the test above is discriminating, not just loosely-toleranced:
    # the float32 accumulator, run over the SAME data, is grossly wrong --
    # measured relative error saturates at 1.0 (diagonal entries collapse
    # towards 0, i.e. the true ~1e-6 variance is destroyed by cancellation
    # against the 1e5 offset carried in float32).
    cov_rel_err_default = np.max(np.abs(acc_default.cov - ref_cov) / np.abs(ref_cov))
    cov_rel_err_f32 = np.max(np.abs(acc_f32.cov - ref_cov) / np.abs(ref_cov))
    assert cov_rel_err_default < 1e-3
    assert cov_rel_err_f32 > 0.5, (
        f"expected float32 accumulation to be grossly wrong on this fixture, "
        f"got rel err {cov_rel_err_f32}; fixture may no longer be discriminating"
    )
    assert cov_rel_err_f32 > 1000 * cov_rel_err_default


# =============================================================================
# streaming_rx -- interface
# =============================================================================

def test_streaming_rx_takes_a_path_not_a_cube():
    """Deliberate exception to the `CONTRIBUTING.md` detector interface
    (`def <name>(cube: np.ndarray, *, ...) -> np.ndarray`): `streaming_rx`
    takes `scene_path`, not a cube, because accepting a materialized cube
    would defeat the module's purpose before a line of it runs. This test
    exists so the Phase 4 registry / Phase 5 harness authors hit an
    intentional, documented assertion here rather than discovering the
    mismatch as an unexplained `TypeError` deep in `Path()`.
    """
    sig = inspect.signature(streaming_rx)
    params = list(sig.parameters)
    assert params[0] == "scene_path"
    assert params[0] != "cube"

    rng = np.random.default_rng(5)
    fake_cube = rng.normal(size=(4, 4, 3)).astype(np.float32)
    with pytest.raises(TypeError):
        streaming_rx(fake_cube)  # not a path-like -- Path() itself raises


# =============================================================================
# streaming_rx -- equivalence with global_rx
# =============================================================================

@pytest.mark.skipif(not _have_indian_pines, reason="Indian Pines not fetched")
def test_streaming_rx_matches_global_rx_on_indian_pines():
    """Accept criterion: matches `global_rx` to `rtol=1e-5` on Indian Pines
    -- LITERALLY UNACHIEVABLE, per the module's own "SECOND, LARGER SPEC
    DISCREPANCY" note: `global_rx` computes in the cube's native float32
    (never upcasts), so `global_rx`'s own mu/sigma already differ from true
    float64 statistics by ~4e-5 / ~5.4e-4 relative. Measured directly here:
    max relative difference ~8.5e-4 (module docstring reports ~8.6e-4,
    consistent). `streaming_rx`'s OWN accumulator, run in one shot over the
    whole cube, matches an independent float64 reference to `0.0` relative
    difference -- so this gap is entirely `global_rx`'s imprecision, not
    `streaming_rx`'s. Tolerance below (2e-3) is set with headroom above the
    measured ~8.5e-4, not equal to it.
    """
    cube, _meta = load_scene(INDIAN_PINES, source="indian_pines")
    g = global_rx(cube)
    s = streaming_rx(INDIAN_PINES, strip_rows=16)

    assert g.shape == s.shape
    valid = ~np.isnan(g)
    assert valid.sum() > 20_000  # "over 20 000+ pixels", per the float64-accumulation rationale
    np.testing.assert_allclose(g[valid], s[valid], rtol=2e-3, atol=0)


@pytest.mark.skipif(not _have_abu, reason="ABU benchmark not fetched")
def test_streaming_rx_matches_global_rx_on_abu():
    """Same equivalence check on `abu-urban-3` -- native radiance scale,
    one of D22's three previously-`LinAlgError`-crashing scenes (now fixed
    by the scale-relative ridge both modules share). This is the regime
    where `global_rx`'s float32 imprecision is amplified far more than on
    Indian Pines: measured max relative difference here is ~4.3e-2 (see
    this file's module docstring for the full write-up and the independent
    float64-reference check that attributes 100% of this gap to
    `global_rx`, not `streaming_rx`). Tolerance below is set with headroom
    above that measured value.
    """
    cube, _meta = load_scene(ABU_SCENE, source="abu")
    g = global_rx(cube)
    s = streaming_rx(ABU_SCENE, strip_rows=16)

    assert g.shape == s.shape
    valid = ~np.isnan(g)
    np.testing.assert_allclose(g[valid], s[valid], rtol=0.1, atol=0)


def test_streaming_rx_regularization_matches_global_rx_scale_relative_ridge():
    """D22: both `global_rx` and `streaming_rx` must use the SAME
    scale-relative ridge (`reg * (trace(sigma)/b) * I`), or the rtol
    equivalence above is comparing two different estimators. This is the
    regime D22 says diverges most visibly under an absolute ridge: a
    synthetic cube at large radiance-like scale (covariance diagonals
    ~1e5), on which an absolute `reg=1e-6` would be arithmetically inert
    (as D22 found for `global_rx` on 3/13 real ABU scenes) while the
    scale-relative ridge conditions it properly. Both detectors must
    therefore still agree tightly here.
    """
    rng = np.random.default_rng(6)
    h, w, b = 24, 24, 8
    cube = (rng.normal(size=(h, w, b)) * 300 + 5000).astype(np.float32)  # radiance-like scale

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "radiance_scale.tif"
        _write_tif(path, cube)

        g = global_rx(cube, reg=1e-6)
        s = streaming_rx(path, strip_rows=4, reg=1e-6)

        valid = ~np.isnan(g)
        np.testing.assert_allclose(g[valid], s[valid], rtol=5e-3, atol=0)


# =============================================================================
# streaming_rx -- NaN handling, out_path, and peak RSS
# =============================================================================

def test_streaming_rx_nan_in_nan_out_positionally(tmp_path):
    rng = np.random.default_rng(7)
    h, w, b = 12, 10, 6
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[5, 3, :] = np.nan

    path = tmp_path / "with_nan.tif"
    _write_tif(path, cube)

    s = streaming_rx(path, strip_rows=3)
    assert np.isnan(s[5, 3])
    flat = s.ravel()
    poisoned = 5 * w + 3
    assert not np.any(np.isnan(np.delete(flat, poisoned)))


def test_streaming_rx_out_path_matches_returned_array(tmp_path):
    rng = np.random.default_rng(8)
    h, w, b = 20, 9, 7
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[2, 2, :] = np.nan

    in_path = tmp_path / "scene.tif"
    _write_tif(in_path, cube)
    out_path = tmp_path / "scored.tif"

    returned = streaming_rx(in_path, strip_rows=5, out_path=out_path)

    assert out_path.exists()
    with rasterio.open(out_path) as ds:
        written = ds.read(1)
        assert ds.crs is not None
        # written raster round-trips NaN via its own nodata tag
        assert np.isnan(ds.nodata)

    valid = ~np.isnan(returned)
    np.testing.assert_array_equal(written[valid], returned[valid])
    assert np.isnan(written[2, 2]) and np.isnan(returned[2, 2])


def test_streaming_rx_peak_rss_under_eighth_of_full_cube_path(tmp_path):
    """Accept criterion: peak RSS via `tracemalloc` is < 1/8 of the
    full-cube path at `strip_rows=16`. Per the module's "STREAMING IS
    FORMAT-DEPENDENT" note, this bound only makes sense on a genuinely
    streamed format -- `.tif` here, NOT Indian Pines/ABU's `.mat` (which
    pays one unavoidable full read regardless of this module's code).

    Full-cube path = `load_scene` (materializes the whole cube) + `global_rx`
    (which itself holds several O(N*B) float64 copies -- centered, dv,
    solved -- simultaneously alive, since none are freed before the
    function returns). Streaming path = `streaming_rx` alone on the same
    file. Measured on this fixture (2000x50x100, ~40MB float32 cube): full
    ~200MB peak, streaming ~2.4MB peak, ratio ~84x -- comfortable margin
    over the required 8x, reported here rather than assumed.
    """
    rng = np.random.default_rng(9)
    h, w, b = 2000, 50, 100
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    path = tmp_path / "big_scene.tif"
    _write_tif(path, cube)
    del cube
    gc.collect()

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        c2, _meta = load_scene(path, source="enmap")
        g = global_rx(c2)
        _cur, full_peak = tracemalloc.get_traced_memory()
        del c2, g
        gc.collect()

        tracemalloc.reset_peak()
        s = streaming_rx(path, strip_rows=16)
        _cur, stream_peak = tracemalloc.get_traced_memory()
        del s
    finally:
        tracemalloc.stop()

    assert stream_peak < full_peak / 8, (
        f"peak RSS ratio {full_peak / stream_peak:.1f}x, full={full_peak}, "
        f"stream={stream_peak} -- required streaming peak < full/8"
    )
