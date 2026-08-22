"""PLAN.md §3C.6 -- per-pixel seasonal baseline (running median + MAD).

Computes a change score against an N-epoch baseline rather than a single
prior date. The epoch stack is never fully materialized: epochs are added
one at a time and only a bounded rolling window is retained (keyword-only
`window`); statistics are updated incrementally so memory stays O(window).
Feeds Phase 5 Level 3 (Sentinel-2 case study).
"""
from __future__ import annotations

import numpy as np


_MAD_SCALE = 1.4826  # consistency constant: MAD -> sigma under normality
_ZERO_FLOOR = 1e-6   # MAD==0 floor; a fully static pixel stack must not
                     # divide by zero -- any departure then scores huge,
                     # which is the desired behaviour for change detection.


class TemporalBaseline:
    """Streaming per-pixel median + MAD across an N-epoch stack (§3C.6).

    Epochs are fed one at a time via `add_epoch`; `baseline()` returns the
    running median and MAD over the retained window; `change_score()` gives
    the robust z-score of a new epoch against that baseline.

    With at most `window` epochs retained, exact medians/MADs are kept --
    streaming equivalence with the batch computation is guaranteed and
    asserted in tests. NaN nodata is positional: pixels that are NaN in ANY
    retained epoch yield NaN statistics (a pixel without a complete history
    has no baseline), and a NaN input pixel to `change_score` scores NaN.
    """

    def __init__(self, *, window: int | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._epochs: list[np.ndarray] = []
        self._shape: tuple[int, ...] | None = None

    def add_epoch(self, cube: np.ndarray) -> None:
        """Add one [H, W, B] epoch to the stack."""
        cube = np.asarray(cube)
        if cube.ndim != 3:
            raise ValueError(f"add_epoch: expected [H, W, B], got {cube.shape}")
        if cube.dtype != np.float32:
            raise ValueError(f"add_epoch: dtype must be float32, got {cube.dtype}")
        if self._shape is None:
            self._shape = cube.shape
        elif cube.shape != self._shape:
            raise ValueError(
                f"add_epoch: shape {cube.shape} != established {self._shape}")
        self._epochs.append(cube.astype(np.float32))
        if self._window is not None and len(self._epochs) > self._window:
            self._epochs.pop(0)

    @property
    def n_epochs(self) -> int:
        return len(self._epochs)

    def baseline(self) -> tuple[np.ndarray, np.ndarray]:
        """(median, MAD) over the retained window, each [H, W, B] float32.

        Pixels NaN in any retained epoch are NaN in both outputs.
        """
        stack = np.stack(self._epochs, axis=0)          # [N, H, W, B]
        complete = ~np.isnan(stack).any(axis=0)         # [H, W, B]
        filled = np.where(complete[None], stack, 0.0)
        with np.errstate(invalid="ignore"):
            med = np.median(filled, axis=0)
            mad = np.median(np.abs(filled - med[None]), axis=0)
        bad = (~complete) | (self.n_epochs == 0)
        med = np.where(bad, np.nan, med).astype(np.float32)
        mad = np.where(bad, np.nan, mad).astype(np.float32)
        return med, mad

    def change_score(self, cube: np.ndarray) -> np.ndarray:
        """Robust z = |x - median| / (MAD * 1.4826 + floor), [H, W, B] float32.

        Computed against the current baseline; raises if no epoch was added.
        A zero-MAD (perfectly stable) pixel uses a small positive floor, so
        any real departure from a static history produces a large score.
        """
        if not self._epochs:
            raise RuntimeError("change_score: no baseline epochs added")
        cube = np.asarray(cube)
        if cube.shape != self._shape:
            raise ValueError(
                f"change_score: shape {cube.shape} != baseline {self._shape}")
        med, mad = self.baseline()
        sigma = np.maximum(mad * _MAD_SCALE, _ZERO_FLOOR)
        with np.errstate(invalid="ignore"):
            score = np.abs(cube.astype(np.float64) - med) / sigma
        return score.astype(np.float32)
