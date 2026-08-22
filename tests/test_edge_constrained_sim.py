"""Tests for edge/constrained_sim.py"""
from __future__ import annotations

from edge.constrained_sim import ConstrainedResult, run_constrained


def test_run_constrained_basic():
    """Test basic constrained run with a simple command."""
    result = run_constrained(["echo", "hello"], cores=2, mem_mb=512, cpu_quota_pct=200, timeout_s=10.0)

    assert isinstance(result, ConstrainedResult)
    assert result.cmd == ["echo", "hello"]
    assert result.cores_requested == 2
    assert result.mem_mb_requested == 512
    assert result.cpu_quota_pct_requested == 200
    assert result.measurement == "SIMULATED"
    assert result.host_cpu is not None
    assert "taskset_cores" in result.applied_constraints
    assert result.applied_constraints["taskset_cores"] == "0,1"
    assert result.status == "ok"
    assert "hello" in result.stdout


def test_run_constrained_string_cmd():
    """Test that string commands are parsed correctly."""
    result = run_constrained("echo hello", cores=1, mem_mb=256, cpu_quota_pct=100, timeout_s=10.0)
    assert result.cmd == ["echo", "hello"]


def test_run_constrained_exit_code_propagated():
    """Test that non-zero exit codes are captured."""
    result = run_constrained(["false"], cores=1, mem_mb=256, cpu_quota_pct=100, timeout_s=10.0)
    assert result.exit_code == 1
    assert result.status.startswith("FAILED:")


def test_run_constrained_timeout():
    """Test timeout handling."""
    # Sleep longer than timeout
    result = run_constrained(["sleep", "5"], cores=1, mem_mb=256, cpu_quota_pct=100, timeout_s=0.1)
    assert result.status == "TIMEOUT"
    assert result.exit_code == -1


def test_constrained_result_to_jsonl():
    """Test JSONL serialization."""
    result = ConstrainedResult(
        cmd=["test"],
        cores_requested=2,
        mem_mb_requested=1024,
        cpu_quota_pct_requested=200,
        wall_time_s=1.5,
        exit_code=0,
        status="ok",
        measurement="SIMULATED",
        host_cpu="test_cpu",
        applied_constraints={"test": "value"},
    )
    jsonl = result.to_jsonl()
    import json
    parsed = json.loads(jsonl)
    assert parsed["cmd"] == ["test"]
    assert parsed["measurement"] == "SIMULATED"
    assert parsed["status"] == "ok"
    assert parsed["applied_constraints"]["test"] == "value"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])