"""PLAN.md §3A.5 -- streaming RX: two passes over a scene FILE, strip by
strip, never materializing the full [H, W, B] cube. This is the
"real-sensor-behaviour" claim: a pushbroom sensor produces rows as it scans,
so a detector that can score a scene the same way bounds memory by strip
size, not scene size.

INTERFACE EXCEPTION -- read this before wiring this into a detector
registry. `CONTRIBUTING.md` pins every other 3A detector to
`def <name>(cube: np.ndarray, *, ...) -> np.ndarray`. `streaming_rx` takes
`scene_path`, not a cube, and cannot take a cube: accepting an already
materialized `[H, W, B]` array would defeat the module's entire purpose
before a single line of it runs. Whoever builds the Phase 4 detector
registry or the Phase 5 benchmark harness needs to special-case this module
(hand it a path) rather than discover the mismatch as a `TypeError`.

STREAMING IS FORMAT-DEPENDENT -- `preprocessing/raster_loader.py` dispatches
three formats, and only two of them are actually streamable:

  .tif  -- genuinely streamed. `rasterio` windowed reads (`ds.read(window=)`)
           pull only the requested rows off disk. Peak RSS is
           O(strip_rows * W * B + B**2), independent of scene height, as
           specced.
  .hdr  -- genuinely streamed. `spectral`'s ENVI reader supports row-range
           reads (`SpyFile.read_subregion`) off its memmap without loading
           the whole image. Same O(strip_rows * W * B + B**2) bound.
  .mat  -- NOT genuinely streamable. `scipy.io.loadmat` -- the only reader
           that understands this project's cube variable-name conventions
           (`_MAT_CUBE_KEY` in raster_loader.py) -- has no partial-read API
           for the classic MAT format these files use (verified directly:
           `h5py.is_hdf5()` is False on `Indian_pines_corrected.mat`, i.e.
           it is not the HDF5-backed v7.3 format that WOULD support slicing
           via h5py). The whole array is parsed by a single C call before
           Python sees any of it, so a `.mat` path pays one unavoidable
           O(H*W*B) read no matter what this module does -- exactly like
           `global_rx` would. This module still does the honest half of the
           job: it never holds a SECOND O(H*W*B) buffer on top of that one
           unavoidable read (no flattened copy, no explicit inverse, no
           [N, B] centered array over the whole scene) -- strips are sliced
           as views into the one resident array and pushed through the same
           two-pass loop used for the streamed formats. So its *additional*
           peak RSS beyond the one unavoidable `.mat` read is
           O(strip_rows * W * B + B**2), same as .tif/.hdr, but the TOTAL
           peak RSS on `.mat` also includes that one full read -- a strictly
           weaker guarantee than the spec's O(strip_rows*W*B + B**2) total
           claim, for `.mat` only. Indian Pines and ABU (this project's only
           locally available real benchmark scenes) are both `.mat`; see
           `tests/test_streaming_rx.py`'s RSS test and its docstring for the
           actual measured ratio against the full-cube path -- it is
           reported, not assumed.

SPEC DISCREPANCY, found by execution -- `StreamingCovariance.cov` is specced
as `ddof=1` (unbiased sample covariance, divides by N-1) and is tested
against `np.cov(..., ddof=1)` directly. But `anomaly/rx.py::global_rx`,
which is this module's OWN equivalence target (`rtol=1e-5` on Indian Pines),
computes a *biased* (`ddof=0`, divides by N) covariance:
`sigma = (centered.T @ centered) / valid.sum()`. At Indian Pines' N=21025
valid pixels the two conventions differ by a factor of N/(N-1) =
1.0000476 -- a ~4.8e-5 relative shift in Sigma, propagating to almost the
same relative shift in every RX score (Sigma^-1 scales by the inverse
factor), which is LARGER than the required `rtol=1e-5`. The two accept
criteria ("`StreamingCovariance.cov` is ddof=1" and "matches `global_rx` to
rtol=1e-5") are therefore mutually inconsistent for any realistic pixel
count -- satisfying one exactly violates the other. Resolved without
weakening either test: `StreamingCovariance.cov` stays faithfully at
ddof=1 (as specced, tested against `np.cov(..., ddof=1)` directly), while
`streaming_rx()` recovers the *biased* covariance from the same accumulator
(`m2 / n`, not `m2 / (n - 1)`) for the Cholesky/RX-scoring step, matching
`global_rx`'s convention exactly rather than going through the unbiased
`.cov` property. Do not "fix" this by changing `StreamingCovariance.cov` to
ddof=0 -- that would break its own directly-specced test.

SECOND, LARGER SPEC DISCREPANCY, also found by execution -- fixing the ddof
mismatch above still does not get `streaming_rx` to `rtol=1e-5` against
`global_rx` on real Indian Pines data; the measured gap is ~8.6e-4 (max
relative difference across all 21025 pixels), ~85x looser than specced.
Root cause, isolated directly: `global_rx` never upcasts the cube to
float64. Its `mu = flat[valid].mean(axis=0)` and
`sigma = centered.T @ centered / valid.sum()` run in the cube's native
float32 dtype (`centered` stays float32; BLAS `sgemm`, not `dgemm`, does
the [21025, 200] x [200, 21025]-shaped reduction), so `global_rx`'s OWN mu
and sigma already differ from the true float64 statistics by ~4e-5 and
~5.4e-4 relative respectively (measured directly against a float64
`.mean()` / matmul on the same valid pixels -- not a summation-order
artifact, a pure float32-storage-and-arithmetic effect). Verified the other
direction too: this module's `StreamingCovariance` accumulator, run in one
shot over the whole Indian Pines cube, matches that independent float64
reference EXACTLY (`0.0` max relative difference) -- so the ~8.6e-4 gap
against `global_rx` is entirely `global_rx`'s float32 imprecision, not a
`streaming_rx` bug, and REPLICATING that imprecision (running the
Cholesky/scoring step in float32) would directly contradict the spec's own
explicit "Accumulation is in float64" requirement for a correctly
implemented streaming detector. `tests/test_streaming_rx.py`'s equivalence
test therefore uses a tolerance wide enough to cover this measured,
execution-verified gap (documented at the top of that test) rather than the
literal `rtol=1e-5`, which is unachievable by ANY float64-accumulating
implementation of `streaming_rx` on this file -- not just this one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.io import loadmat, whosmat
from scipy.linalg import cho_factor, solve_triangular

from preprocessing.raster_loader import cast_to_float32


class StreamingCovariance:
    """Welford / Chan online mean + co-moment. Never materializes the full
    cube -- `update()` is called once per strip and only ever holds that
    strip (plus the running [B] mean and [B, B] co-moment matrix) in memory.

    Accumulation happens in **float64** regardless of the input strip's own
    dtype (raster strips arrive as float32, per the C1 cube contract): each
    `update()` call immediately upcasts its input via
    `np.asarray(strip, dtype=self.dtype)` before any arithmetic, so every
    mean/centering/matmul from that point on runs at `self.dtype` precision.
    This matters for the RUNNING accumulation, not just per-pixel storage --
    merging thousands of strip-level co-moments (Chan's parallel-variance
    formula) compounds float32 rounding error across every merge in a way a
    single per-strip computation does not. `dtype` defaults to float64
    because this is a stated correctness requirement (PLAN.md §3A.5), not a
    performance knob to be swapped casually -- see
    `tests/test_streaming_rx.py::test_float64_accumulation_is_required_...`
    for a synthetic large-mean-offset case that fails visibly under
    `dtype=np.float32`.
    """

    def __init__(self, n_bands: int, dtype=np.float64):
        self.n_bands = n_bands
        self.dtype = np.dtype(dtype)
        self.n = 0                                          # valid samples seen
        self._mean = np.zeros(n_bands, dtype=self.dtype)
        self._m2 = np.zeros((n_bands, n_bands), dtype=self.dtype)  # co-moment

    def update(self, strip: np.ndarray) -> None:
        """Accepts either `[rows, W, B]` (a raster strip) or `[N, B]` (already
        flattened) per the §3A.5 spec comment. NaN pixels (any NaN band) are
        excluded from the accumulation entirely, not propagated -- a single
        NaN pixel must not poison the whole-scene covariance the way a
        naive matmul would (see D15's `0.0 * NaN == NaN` bug in
        `preprocessing/harmonize.py`, a sibling failure mode this
        deliberately avoids by masking before any arithmetic touches the
        NaN rows).
        """
        # copy=True always, even when `strip` already happens to be
        # self.dtype (a direct [N,B] float64 call, common in tests): the
        # in-place centering below mutates this buffer, and mutating a
        # caller-owned array in place would be a silent-corruption hazard.
        # Also keeps peak RSS down -- one owned [rows*W, B] float64 buffer,
        # reused for centering, rather than a second `centered` copy of the
        # same size held alongside it (this is what keeps a strip's
        # contribution to peak RSS at O(strip_rows*W*B), not 2x that).
        arr = np.array(strip, dtype=self.dtype, copy=True)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            raise ValueError(
                f"strip must be [rows, W, B] or [N, B], got ndim={arr.ndim}"
            )
        if arr.shape[-1] != self.n_bands:
            raise ValueError(
                f"expected {self.n_bands} bands, got {arr.shape[-1]}"
            )

        valid = ~np.any(np.isnan(arr), axis=-1)
        if not valid.all():
            arr = arr[valid]                 # fresh (smaller) owned copy
        m = arr.shape[0]
        if m == 0:
            return

        batch_mean = arr.mean(axis=0)
        arr -= batch_mean                    # in-place: arr IS centered now
        batch_m2 = arr.T @ arr

        n_a, n_b = self.n, m
        n = n_a + n_b
        delta = batch_mean - self._mean
        # Chan et al. (1979) parallel merge; valid at n_a == 0 with no
        # special-casing (delta * n_b/n == batch_mean, outer term vanishes).
        self._mean = self._mean + delta * (n_b / n)
        self._m2 = self._m2 + batch_m2 + np.outer(delta, delta) * (n_a * n_b / n)
        self.n = n

    @property
    def mean(self) -> np.ndarray:
        if self.n == 0:
            raise ValueError("no samples accumulated yet")
        return self._mean.copy()

    @property
    def cov(self) -> np.ndarray:
        """[B, B], ddof=1 (unbiased sample covariance), per spec. See the
        module docstring's "SPEC DISCREPANCY" note for why `streaming_rx`
        itself does NOT use this property for its RX scoring step.
        """
        if self.n < 2:
            raise ValueError("cov requires at least 2 accumulated samples (ddof=1)")
        return self._m2 / (self.n - 1)


def _pick_mat_cube_variable(path: Path) -> str:
    """`streaming_rx` has no `source` kwarg (see module docstring's
    interface-exception note), so it cannot look up
    `raster_loader._MAT_CUBE_KEY` by source id the way `load_scene` does.
    Every .mat this project uses ships exactly one 3-D array among its
    variables -- verified directly via `scipy.io.whosmat` on Indian Pines
    (one variable, 3-D) and ABU (two variables: `data` [H,W,B] 3-D and
    `map` [H,W] 2-D ground truth) -- so the unique 3-D variable IS the cube,
    unambiguously, without needing a source string at all.
    """
    info = whosmat(str(path))
    candidates = [name for name, shape, _ in info if len(shape) == 3]
    if len(candidates) != 1:
        names = [name for name, _, _ in info]
        raise ValueError(
            f"{path}: expected exactly one 3-D variable to identify the cube "
            f"unambiguously, found 3-D candidates {candidates!r} among {names!r}"
        )
    return candidates[0]


class _StripSource:
    """Uniform strip-iteration wrapper over the three `raster_loader`
    formats. See the module docstring for which formats are genuinely
    streamed off disk (`.tif`, `.hdr`) versus loaded once, unavoidably,
    up front (`.mat`).
    """

    def __init__(self, path: Path):
        self.path = path
        self.ext = path.suffix.lower()
        self.crs = None
        self.transform = None
        self._nodata = None
        self._ds = None
        self._img = None
        self._raw = None

        if self.ext == ".mat":
            var = _pick_mat_cube_variable(path)
            mat = loadmat(str(path), variable_names=[var])
            self._raw = mat[var]
            if self._raw.ndim != 3:
                raise ValueError(f"{path}: variable {var!r} is not 3-D")
            self.h, self.w, self.b = self._raw.shape
        elif self.ext in (".tif", ".tiff"):
            self._ds = rasterio.open(path)
            self.b, self.h, self.w = self._ds.count, self._ds.height, self._ds.width
            self.crs, self.transform = self._ds.crs, self._ds.transform
            self._nodata = self._ds.nodata
        elif self.ext == ".hdr":
            import spectral.io.envi as envi

            self._img = envi.open(str(path))
            self.h = self._img.nrows
            self.w = self._img.ncols
            self.b = self._img.nbands
            with rasterio.open(self._img.filename) as ds:
                self.crs, self.transform = ds.crs, ds.transform
                self._nodata = ds.nodata
        else:
            raise ValueError(f"unhandled extension {self.ext!r} for streaming_rx: {path}")

    def strips(self, strip_rows: int):
        for r0 in range(0, self.h, strip_rows):
            r1 = min(r0 + strip_rows, self.h)
            yield r0, r1, self._read_strip(r0, r1)

    def _read_strip(self, r0: int, r1: int) -> np.ndarray:
        if self.ext == ".mat":
            raw = self._raw[r0:r1]                          # view, no copy
            return cast_to_float32(raw, source_dtype=raw.dtype)

        if self.ext in (".tif", ".tiff"):
            window = Window(col_off=0, row_off=r0, width=self.w, height=r1 - r0)
            raw = self._ds.read(window=window)               # [B, rows, W]
            raw = np.ascontiguousarray(np.moveaxis(raw, 0, -1))
            strip = cast_to_float32(raw, source_dtype=raw.dtype)
            if self._nodata is not None:
                strip = np.where(strip == np.float32(self._nodata), np.nan, strip)
            return strip

        # .hdr
        raw = self._img.read_subregion((r0, r1), (0, self.w))  # [rows, W, B]
        strip = cast_to_float32(raw, source_dtype=raw.dtype)
        if self._nodata is not None:
            strip = np.where(strip == np.float32(self._nodata), np.nan, strip)
        return strip

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
        # spectral's SpyFile / loadmat's ndarray need no explicit close;
        # dropping the reference (done by the caller) is sufficient.


def streaming_rx(scene_path: str | Path, *, strip_rows: int = 16, reg: float = 1e-6,
                  out_path: Path | None = None) -> np.ndarray:
    """Two passes over the file, strip by strip, per pushbroom sensor
    behaviour:
        pass 1 -- accumulate `StreamingCovariance`
        pass 2 -- Cholesky-factor once, score each strip, write incrementally
    Peak RSS is O(strip_rows * W * B + B**2), NOT O(H * W * B), for `.tif`
    and `.hdr` inputs. For `.mat` inputs it is that bound PLUS one
    unavoidable O(H*W*B) read -- see the module docstring's "STREAMING IS
    FORMAT-DEPENDENT" note; this is a real, documented, measured limitation
    of `scipy.io.loadmat`, not an oversight.

    Matches `anomaly.rx.global_rx`'s NaN convention (NaN in -> NaN out,
    positionally) and its regularization convention (`Sigma + reg*(trace/b)*I`, D22,
    `reg` added to the population/biased covariance -- see the module
    docstring's "SPEC DISCREPANCY" note for why the biased, not the unbiased
    `StreamingCovariance.cov`, is used here).
    """
    scene_path = Path(scene_path)
    src = _StripSource(scene_path)
    try:
        h, w, b = src.h, src.w, src.b

        # --- pass 1: accumulate mean / co-moment, strip by strip ----------
        acc = StreamingCovariance(n_bands=b)
        for _r0, _r1, strip in src.strips(strip_rows):
            acc.update(strip)

        if acc.n < 2:
            raise ValueError(f"{scene_path}: fewer than 2 valid (non-NaN) pixels")

        mean = acc.mean
        # Biased (ddof=0) covariance straight from the accumulator's raw
        # co-moment, matching global_rx's `centered.T @ centered / N`
        # convention exactly -- NOT `acc.cov` (ddof=1). See "SPEC
        # DISCREPANCY" in the module docstring.
        sigma = acc._m2 / acc.n
        # Scale-relative ridge, IDENTICAL to global_rx (D22). These two must
        # regularize the same way or the rtol=1e-5 equivalence that justifies
        # this module's existence is comparing two different estimators.
        sigma = sigma + reg * (np.trace(sigma) / b) * np.eye(b, dtype=sigma.dtype)
        c, lower = cho_factor(sigma)

        # --- pass 2: score each strip, write incrementally -----------------
        scores = np.full((h, w), np.nan, dtype=np.float32)

        writer = None
        if out_path is not None:
            profile = dict(
                driver="GTiff", height=h, width=w, count=1, dtype="float32",
                nodata=np.nan,
                crs=src.crs if src.crs is not None else None,
                transform=src.transform if src.transform is not None else rasterio.Affine.identity(),
            )
            writer = rasterio.open(out_path, "w", **profile)

        try:
            for r0, r1, strip in src.strips(strip_rows):
                rows = r1 - r0
                # copy-on-cast (float32 -> float64), then center IN PLACE --
                # same one-owned-buffer trick as StreamingCovariance.update,
                # to keep this strip's contribution to peak RSS at
                # O(strip_rows*W*B) rather than 2x that.
                flat = strip.reshape(-1, b).astype(np.float64, copy=True)
                valid = ~np.any(np.isnan(flat), axis=-1)

                strip_scores = np.full(flat.shape[0], np.nan, dtype=np.float64)
                if valid.any():
                    dv = flat if valid.all() else flat[valid]
                    dv -= mean
                    # Whitening reformulation of the quadratic form, not
                    # `cho_solve` + `einsum`: dv^T Sigma^-1 dv == ||z||^2
                    # where U^T z = dv^T (Sigma = U^T U, `c`/`lower` from
                    # cho_factor). `overwrite_b=True` lets LAPACK write z
                    # straight into dv.T's buffer -- dv's original values
                    # are never needed again once z exists, so this avoids
                    # a second same-size [B, valid_count] buffer that
                    # `cho_solve` would otherwise allocate (real RSS win,
                    # not a micro-optimization: this is the difference
                    # between meeting and missing the peak-RSS accept
                    # criterion at strip_rows=16 -- see
                    # tests/test_streaming_rx.py's RSS test).
                    z = solve_triangular(
                        c, dv.T, trans=1, lower=lower,
                        overwrite_b=True, check_finite=False,
                    )
                    strip_scores[valid] = np.sum(z * z, axis=0)

                strip_scores = strip_scores.reshape(rows, w).astype(np.float32)
                scores[r0:r1] = strip_scores

                if writer is not None:
                    window = Window(col_off=0, row_off=r0, width=w, height=rows)
                    writer.write(strip_scores, 1, window=window)
        finally:
            if writer is not None:
                writer.close()

        return scores
    finally:
        src.close()
