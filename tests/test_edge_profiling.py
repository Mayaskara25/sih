"""Tests for edge/profiling.py"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from edge.profiling import ProfileRecord, ProfileSink, profile_stage, regression_check


def test_profile_record_to_jsonl():
    rec = ProfileRecord(
        stage="test_stage",
        wall_time_s=1.5,
        cpu_time_s=1.2,
        peak_rss_delta_mb=10.0,
        thread_count=4,
        measurement="SIMULATED",
        onnxruntime_threads=1,
        blas_threads=4,
    )
    jsonl = rec.to_jsonl()
    assert '"stage":"test_stage"' in jsonl
    assert '"measurement":"SIMULATED"' in jsonl
    # Verify it's valid JSON
    import json
    parsed = json.loads(jsonl)
    assert parsed["stage"] == "test_stage"
    assert parsed["measurement"] == "SIMULATED"


def test_profile_sink_write(tmp_path: Path):
    sink = ProfileSink("test_run", base_dir=tmp_path)
    rec = ProfileRecord(
        stage="stage1",
        wall_time_s=1.0,
        cpu_time_s=0.8,
        peak_rss_delta_mb=5.0,
        thread_count=2,
    )
    sink.write(rec)

    content = (tmp_path / "test_run.jsonl").read_text()
    assert "stage1" in content
    assert "SIMULATED" in content


def test_profile_sink_terminal_record(tmp_path: Path):
    sink = ProfileSink("test_run2", base_dir=tmp_path)
    sink.write_terminal("completed")

    content = (tmp_path / "test_run2.jsonl").read_text()
    assert "__terminal__" in content
    assert "completed" in content


def test_profile_stage_context_manager(tmp_path: Path):
    sink = ProfileSink("test_run3", base_dir=tmp_path)

    with profile_stage("test_stage", sink, blas_threads=2):
        time.sleep(0.01)  # Small delay to have measurable time

    content = (tmp_path / "test_run3.jsonl").read_text()
    assert "test_stage" in content
    assert "SIMULATED" in content
    assert "blas_threads" in content


def test_regression_check_detects_regression():
    baseline = {
        "stage1": ProfileRecord("stage1", 1.0, 0.8, 10.0, 2),
        "stage2": ProfileRecord("stage2", 2.0, 1.5, 20.0, 4),
    }
    current = {
        "stage1": ProfileRecord("stage1", 1.0, 0.8, 10.0, 2),  # no change
        "stage2": ProfileRecord("stage2", 2.5, 2.0, 25.0, 4),  # 25% regression
    }
    regressed = regression_check(baseline, current, tol=0.15)
    assert "stage2" in regressed
    assert "stage1" not in regressed


def test_regression_check_ignores_missing_stages():
    baseline = {"stage1": ProfileRecord("stage1", 1.0, 0.8, 10.0, 2)}
    current = {"stage2": ProfileRecord("stage2", 2.0, 1.5, 20.0, 4)}
    regressed = regression_check(baseline, current)
    assert regressed == []


def test_regression_check_handles_zero_baseline():
    baseline = {"stage1": ProfileRecord("stage1", 0.0, 0.0, 10.0, 2)}
    current = {"stage1": ProfileRecord("stage1", 1.0, 0.8, 10.0, 2)}
    regressed = regression_check(baseline, current)
    assert regressed == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])