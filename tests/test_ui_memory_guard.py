"""D35's OOM guard, pinned to the two measurements it was derived from.

An earlier COPY_FACTOR of 4.0 estimated a full EnMAP scene at 5,189 MB --
under MEM_LIMIT_MB -- and would have ALLOWED the exact scene the kernel
OOM-killed at 8,700 MB RSS during Phase 5 Level 2 (plan.md D32/D35). The
guard passed its own unit tests while failing the only job it has, so
these assert against the MEASURED outcomes, not against the constant.
"""
import pytest

from ui.app import MEM_LIMIT_MB, estimate_mb

# (h, w, bands, measured peak RSS in MB, was it killed) -- plan.md D32.
FULL_SCENE = (1178, 1229, 224, 8700, True)
WINDOW_600 = (600, 402, 224, 829, False)


@pytest.mark.parametrize("h,w,b,measured_mb,was_killed", [FULL_SCENE, WINDOW_600])
def test_guard_agrees_with_what_the_kernel_actually_did(h, w, b, measured_mb, was_killed):
    """Refuse what died; allow what completed."""
    refused = estimate_mb(h, w, b) > MEM_LIMIT_MB
    assert refused == was_killed, (
        f"{h}x{w}x{b} measured {measured_mb} MB and "
        f"{'was OOM-killed' if was_killed else 'completed'}, but the guard "
        f"{'refuses' if refused else 'allows'} it")


def test_full_scene_estimate_is_not_optimistic():
    """The estimate must not sit below the measured peak -- under-predicting
    is the direction that gets the process killed."""
    h, w, b, measured_mb, _ = FULL_SCENE
    assert estimate_mb(h, w, b) >= measured_mb * 0.95


def test_estimate_scales_with_every_dimension():
    base = estimate_mb(600, 402, 224)
    assert estimate_mb(1200, 402, 224) > base
    assert estimate_mb(600, 804, 224) > base
    assert estimate_mb(600, 402, 448) > base
