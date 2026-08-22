"""3D.1 -- StripPipeline: multi-stage strip streaming with halo management
and a hard RSS ceiling (plan.md §6.4 3D.1).

The scheduler feeds overlapping strips so a stage needing *k* rows of context
(lookahead) gets its halo without the caller managing it: requesting output
rows [r0, r1) from stage j recursively requests rows widened by that stage's
lookahead from stage j-1, down to disk reads off `_StripSource`
(`anomaly/streaming_rx.py` -- reused, not reimplemented).

MEMORY MODEL -- the cgroup trap does not apply inside one process: a cgroup
memory cap SIGKILLs silently (exit 137, no traceback); it never raises
MemoryError. So this pipeline enforces its OWN ceiling by sampling RSS with
psutil between stages and raising MemoryBudgetExceeded BEFORE the cap is
reached. The cgroup cap (edge/constrained_sim.py) sits ABOVE this ceiling as
the kernel backstop, never as the mechanism.
"""
from __future__ import annotations

import resource
from pathlib import Path

import numpy as np
import psutil

from anomaly.streaming_rx import _StripSource

DEFAULT_RSS_CEILING_MB = 6144   # 6 GB hard ceiling; machine OOM-kills rather than swaps


class MemoryBudgetExceeded(RuntimeError):
    """Raised BEFORE the process RSS reaches the configured ceiling.

    A pipeline that silently swaps invalidates every latency number taken
    from it; raising here keeps the failure loud and attributable.
    """


class StripPipeline:
    """Ordered chain of strip stages over a scene file.

    Each stage maps `[rows, W, C_in] -> [rows, W, C_out]` for whatever rows
    it is handed (plus a halo it may consume). A stage registered with
    ``lookahead_rows=k`` receives up to k EXTRA rows on each side of the
    requested window; its return value must cover exactly the requested
    core rows. The first stage consumes raw spectral strips `[rows, W, B]`.
    """

    def __init__(self, *, rss_ceiling_mb: float = DEFAULT_RSS_CEILING_MB):
        self.rss_ceiling_mb = float(rss_ceiling_mb)
        self.stages: list[tuple[str, object, int]] = []   # (name, fn, lookahead)
        # High-water mark at construction. The ceiling below is enforced
        # against what THIS RUN allocates and holds -- not against the
        # process's lifetime maximum, which can never decrease and would
        # otherwise pin every later pipeline on the largest thing the
        # process has ever done (a long-lived driver process, or one pytest
        # session that ran a training test first, would fail forever).
        self._peak_at_start_mb = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

    def register(self, name: str, fn, *, lookahead_rows: int = 0) -> "StripPipeline":
        if lookahead_rows < 0:
            raise ValueError(f"lookahead_rows must be >= 0, got {lookahead_rows}")
        self.stages.append((name, fn, lookahead_rows))
        return self

    # ------------------------------------------------------------------ #

    def _check_rss(self, where: str) -> None:
        """Two samples, because a transient spike frees before a point-in-time
        check can see it: current RSS (psutil) AND the kernel high-water mark
        (`ru_maxrss`), which survives frees -- taken as GROWTH since this
        pipeline was constructed, so pre-existing process history is not held
        against the run. A stage that allocates 2.5 GB and releases it before
        returning is STILL over a 1 GB ceiling -- the machine would have been
        at the OOM-killer's mercy during that window, and on this host there
        is no swap to degrade through."""
        rss_mb = psutil.Process().memory_info().rss / 1e6
        # Linux reports ru_maxrss in KiB.
        peak_total_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        peak_delta_mb = peak_total_mb - self._peak_at_start_mb
        worst = max(rss_mb, peak_delta_mb)
        if worst > self.rss_ceiling_mb:
            raise MemoryBudgetExceeded(
                f"RSS {rss_mb:.0f} MB (run-attributable high-water "
                f"{peak_delta_mb:.0f} MB) exceeded ceiling "
                f"{self.rss_ceiling_mb:.0f} MB at {where} -- refusing to "
                f"continue toward swap"
            )

    def _source_rows(self, src: _StripSource, r0: int, r1: int) -> np.ndarray:
        # Same private-read reuse streaming_rx's own two-pass loop relies on;
        # .mat inputs pay their one unavoidable full read here (see that
        # module's docstring), .tif/.hdr are genuinely windowed off disk.
        return src._read_strip(r0, r1)

    def _compute(self, src: _StripSource, idx: int, r0: int, r1: int) -> np.ndarray:
        """Rows [r0, r1) of stage `idx`'s output (halo-clipped to the scene)."""
        h = src.h
        r0c, r1c = max(0, r0), min(h, r1)

        if idx < 0:
            arr = self._source_rows(src, r0c, r1c)
        else:
            name, fn, lookahead = self.stages[idx]
            wide = self._compute(src, idx - 1, r0c - lookahead, r1c + lookahead)
            out = fn(wide)
            out = np.asarray(out)
            if out.shape[0] != wide.shape[0]:
                raise ValueError(
                    f"stage {name!r} returned {out.shape[0]} rows for "
                    f"{wide.shape[0]} input rows -- stages must preserve row count"
                )
            # Trim back to exactly the requested core rows (wide was clipped).
            lo = r0c - max(0, r0c - lookahead)
            arr = out[lo:lo + (r1c - r0c)]

        self._check_rss(f"stage {idx} rows [{r0},{r1})")
        return arr

    def run(self, scene_path: str | Path, *, strip_rows: int = 16,
            blas_threads: int | None = None):
        """Stream the whole scene through the registered stages.

        Returns the FINAL stage's full-scene output `[H, W, C_final]`. When no
        stages are registered, returns the raw strips re-assembled (identity),
        which also makes this usable as a pure streaming loader.
        """
        if strip_rows < 1:
            raise ValueError(f"strip_rows must be >= 1, got {strip_rows}")

        src = _StripSource(Path(scene_path))
        try:
            chunks = [
                self._compute(src, len(self.stages) - 1 if self.stages else -1,
                              r0, min(r0 + strip_rows, src.h))
                for r0 in range(0, src.h, strip_rows)
            ]
        finally:
            src.close()

        out = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0, 0))
        self._check_rss("run() assembly")
        return out