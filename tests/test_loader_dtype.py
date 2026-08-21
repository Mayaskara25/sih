"""D13.2 — dtype safety in the loader.

ABU ships int16 (8 scenes), uint16 (4) and float64 (1). The plan previously
assumed one dtype for the dataset. These tests pin both directions of the
misread, and they exist because one of the two is evidenced in real data.

EVIDENCED (the dangerous one): 8 ABU scenes are int16 and genuinely contain
small negative values -- min -50 to -1, and up to 45 626 negative pixels in
abu-urban-2. These are ordinary residuals after dark-current / atmospheric
correction. Reinterpreted as uint16, -1 becomes 65535: three times the scene
maximum, at tens of thousands of pixels, fed into a detector whose whole job is
to flag extreme values. RX would fire on decoding artefacts and the resulting
anomaly map would look entirely plausible.

NOT EVIDENCED, guarded anyway: a genuine DN above 32767 stored as uint16 and
read as int16 would wrap negative. No current benchmark scene triggers this --
the maximum observed anywhere in ABU is 19 492 -- but nothing guarantees the
next dataset is as well behaved, so the guard stays.
"""
import numpy as np
import pytest

raster_loader = pytest.importorskip(
    "preprocessing.raster_loader",
    reason="Phase 1 module not yet implemented; test is committed ahead of it",
)


def test_int16_negatives_survive_the_cast():
    """The evidenced hazard: signed data must not be read as unsigned."""
    raw = np.array([[-1, -50, 0, 6604]], dtype=np.int16)
    out = raster_loader.cast_to_float32(raw, source_dtype=np.dtype(np.int16))
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(-1.0)
    assert out[0, 1] == pytest.approx(-50.0)
    # the specific corruption we are guarding against
    assert out[0, 0] != pytest.approx(65535.0)
    assert not np.any(out > 32767), "negative int16 was reinterpreted as uint16"


def test_uint16_high_values_do_not_wrap_negative():
    """The reverse, unevidenced in current data but guarded."""
    raw = np.array([[40000, 65535, 0]], dtype=np.uint16)
    out = raster_loader.cast_to_float32(raw, source_dtype=np.dtype(np.uint16))
    assert out[0, 0] == pytest.approx(40000.0)
    assert out[0, 1] == pytest.approx(65535.0)
    assert not np.any(out < 0), "uint16 wrapped negative -- read as int16"


def test_declared_dtype_must_match_the_array():
    """A dtype recorded in meta that disagrees with the array is a bug, not a cast."""
    raw = np.array([[1, 2]], dtype=np.int16)
    with pytest.raises((AssertionError, ValueError)):
        raster_loader.cast_to_float32(raw, source_dtype=np.dtype(np.uint16))


def test_unhandled_dtype_raises_rather_than_guessing():
    raw = np.array([[1, 2]], dtype=np.int8)
    with pytest.raises(ValueError):
        raster_loader.cast_to_float32(raw, source_dtype=np.dtype(np.int8))


@pytest.mark.parametrize("dtype,value", [(np.int16, -1), (np.uint16, 40000)])
def test_roundtrip_is_value_preserving(dtype, value):
    raw = np.full((2, 2), value, dtype=dtype)
    out = raster_loader.cast_to_float32(raw, source_dtype=np.dtype(dtype))
    assert np.all(out == np.float32(value))
