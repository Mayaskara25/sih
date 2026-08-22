"""Tests for edge/streaming.py (3D.1)"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import rasterio

from edge.streaming import MemoryBudgetExceeded, StripPipeline
from anomaly.streaming_rx import _StripSource


def _write_tif(path: Path, cube: np.ndarray) -> None:
    h, w, b = cube.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=b, dtype="float32",
        crs="EPSG:32616", transform=rasterio.Affine(10.0, 0, 100.0, 0, -10.0, 200.0),
    ) as ds:
        ds.write(np.moveaxis(cube, -1, 0))


def test_register_signature_and_run_identity(tmp_path):
    """No stages registered -> run() is a streaming identity: output equals input."""
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(33, 20, 5)).astype(np.float32)
    path = tmp_path / "scene.tif"
    _write_tif(path, cube)

    pipe = StripPipeline()
    out = pipe.run(path, strip_rows=16)
    np.testing.assert_allclose(out, cube, rtol=1e-6)


def test_single_stage_receives_halo_and_preserves_core_rows(tmp_path):
    """A stage with lookahead_rows=k must see k extra rows each side but must
    return exactly the requested core rows; assembled output covers the scene."""
    rng = np.random.default_rng(1)
    h, w, b = 40, 12, 3
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    path = tmp_path / "scene.tif"
    _write_tif(path, cube)

    seen = []

    def stage(strip):
        seen.append(strip.shape[0])
        return strip * 2.0

    pipe = StripPipeline()
    pipe.register("double", stage, lookahead_rows=2)
    out = pipe.run(path, strip_rows=16)

    # The FIRST window saw its 16 core rows plus the 2-row halo below; the
    # halo above clips at the scene's top edge (16+2=18). Ragged trailing
    # windows legitimately see fewer.
    assert seen[0] == 18
    np.testing.assert_allclose(out, cube * 2.0, rtol=1e-6)


def test_stage_returning_wrong_row_count_raises(tmp_path):
    rng = np.random.default_rng(2)
    cube = rng.normal(size=(20, 8, 3)).astype(np.float32)
    path = tmp_path / "scene.tif"
    _write_tif(path, cube)

    def bad(strip):
        return strip[:-1]           # drops a row -- forbidden contract break

    pipe = StripPipeline()
    pipe.register("bad", bad)
    with pytest.raises(ValueError, match="preserve row count"):
        pipe.run(path, strip_rows=8)


def test_negative_lookahead_rejected():
    pipe = StripPipeline()
    with pytest.raises(ValueError, match="lookahead_rows"):
        pipe.register("x", lambda s: s, lookahead_rows=-1)


def test_memory_budget_exceeded_raised_before_cap(tmp_path):
    """The pipeline's OWN ceiling must fire via psutil sampling -- not rely on
    the kernel's silent SIGKILL (exit 137), which never raises MemoryError."""
    rng = np.random.default_rng(3)
    big = rng.normal(size=(200, 200, 200)).astype(np.float32)   # ~32 MB per copy
    path = tmp_path / "big.tif"
    _write_tif(path, big)

    def hog(strip):
        # Touch every page: np.zeros is lazy (calloc + overcommit), so RSS
        # only rises where pages are written. This spike is ~2.5 GB.
        hog_arr = np.empty((4000, 4000, 40), dtype=np.float32)
        hog_arr.fill(1.0)
        return strip

    pipe = StripPipeline(rss_ceiling_mb=1024)                   # 1 GB ceiling
    pipe.register("hog", hog)
    with pytest.raises(MemoryBudgetExceeded):
        pipe.run(path, strip_rows=50)


def test_two_stage_chaining_propagates_lookahead(tmp_path):
    """Stage 2's lookahead requires stage-1 output beyond the core window;
    chaining must produce results identical to running stage 1 then stage 2
    on the whole scene at once."""
    rng = np.random.default_rng(4)
    h, w, b = 24, 10, 2
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    path = tmp_path / "scene.tif"
    _write_tif(path, cube)

    pipe = StripPipeline()
    pipe.register("s1", lambda s: s + 1.0, lookahead_rows=1)
    pipe.register("s2", lambda s: s * 3.0, lookahead_rows=1)
    out = pipe.run(path, strip_rows=7)

    np.testing.assert_allclose(out, (cube + 1.0) * 3.0, rtol=1e-6)


def test_run_validates_strip_rows():
    pipe = StripPipeline()
    with pytest.raises(ValueError, match="strip_rows"):
        pipe.run("/nonexistent", strip_rows=0)


def test_pipeline_uses_shared_strip_source_not_a_reimplementation():
    """Reuse rule (plan.md 'Reuse -- do not reimplement'): StripPipeline reads
    through anomaly/streaming_rx._StripSource."""
    src = inspect.getsource(_StripSource)
    import edge.streaming as m
    assert m._StripSource is _StripSource


if __name__ == "__main__":
    pytest.main([__file__, "-v"])