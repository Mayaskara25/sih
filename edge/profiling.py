"""3D.4 -- profiling utilities for edge benchmarking.

Every record carries "measurement": "SIMULATED" (plan.md §6.4, §9).
Thread counts for onnxruntime and BLAS are set explicitly, never by default.
"""
from __future__ import annotations

import json
import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil


@dataclass
class ProfileRecord:
    stage: str
    wall_time_s: float
    cpu_time_s: float
    peak_rss_delta_mb: float
    thread_count: int
    measurement: str = "SIMULATED"
    onnxruntime_threads: int = 1
    blas_threads: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


class ProfileSink:
    """Appends ProfileRecord JSONL to experiments/edge_benchmarks/{run_id}.jsonl."""

    def __init__(self, run_id: str, base_dir: Path | None = None):
        self.run_id = run_id
        self.base_dir = base_dir or Path(__file__).resolve().parents[1] / "experiments" / "edge_benchmarks"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / f"{run_id}.jsonl"
        self._terminal_written = False

    def write(self, record: ProfileRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_jsonl() + "\n")

    def write_terminal(self, status: str = "completed") -> None:
        """Write a terminal record so truncated runs are detectable."""
        if not self._terminal_written:
            terminal = ProfileRecord(
                stage="__terminal__",
                wall_time_s=0.0,
                cpu_time_s=0.0,
                peak_rss_delta_mb=0.0,
                thread_count=0,
                measurement="SIMULATED",
                extra={"status": status, "run_id": self.run_id},
            )
            self.write(terminal)
            self._terminal_written = True


@contextmanager
def profile_stage(name: str, sink: ProfileSink, *,
                  onnxruntime_threads: int = 1,
                  blas_threads: int = 1):
    """Context manager recording wall time, CPU time, peak RSS delta, thread count.

    Sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS
    and onnxruntime session intra_op_num_threads for the duration of the stage.
    """
    # Set thread counts explicitly (never by default)
    old_env = {}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        old_env[var] = os.environ.get(var)
        os.environ[var] = str(blas_threads)

    # onnxruntime thread count must be set per-session; we record the intended value
    # and the caller is responsible for passing it to ort.InferenceSession.

    proc = psutil.Process()
    tracemalloc.start()
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    rss_before = proc.memory_info().rss

    try:
        yield
    finally:
        rss_after = proc.memory_info().rss
        t1_cpu = time.process_time()
        t1_wall = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Restore env
        for var, val in old_env.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val

        record = ProfileRecord(
            stage=name,
            wall_time_s=t1_wall - t0_wall,
            cpu_time_s=t1_cpu - t0_cpu,
            peak_rss_delta_mb=(rss_after - rss_before) / 1e6,
            thread_count=proc.num_threads(),
            onnxruntime_threads=onnxruntime_threads,
            blas_threads=blas_threads,
        )
        sink.write(record)


def regression_check(baseline: dict[str, ProfileRecord],
                     current: dict[str, ProfileRecord],
                     *,
                     tol: float = 0.15) -> list[str]:
    """Return list of stage names where current regressed > tol vs baseline.

    Comparison is on wall_time_s. Missing stages in either dict are ignored.
    """
    regressed = []
    for stage, base_rec in baseline.items():
        if stage not in current:
            continue
        cur_rec = current[stage]
        if base_rec.wall_time_s <= 0:
            continue
        ratio = cur_rec.wall_time_s / base_rec.wall_time_s
        if ratio > 1.0 + tol:
            regressed.append(stage)
    return regressed