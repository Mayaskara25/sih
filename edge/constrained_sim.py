"""3D.5 -- constrained simulation using cgroup v2 + taskset.

No Raspberry Pi exists. Every number is SIMULATED and labelled so.
A throttled x86 core is NOT a Cortex-A76 -- nothing from this module may be
presented as a Pi number (plan.md §6.4, §9).
"""
from __future__ import annotations

import json
import platform
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConstrainedResult:
    cmd: list[str]
    cores_requested: int
    mem_mb_requested: int
    cpu_quota_pct_requested: int
    wall_time_s: float
    exit_code: int
    status: str  # "ok" | "OOM-killed" | "FAILED"
    measurement: str = "SIMULATED"
    host_cpu: str = field(default_factory=lambda: platform.processor() or "unknown")
    applied_constraints: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


def _read_cgroup_limits() -> dict[str, str]:
    """Read back the actual cgroup limits applied to this process."""
    try:
        cg_line = Path("/proc/self/cgroup").read_text().strip()
        # Format: 0::/user.slice/user-1000.slice/session-XX.scope
        parts = cg_line.split("::")
        if len(parts) < 2:
            return {}
        cg_path = parts[1].strip()
        base = Path("/sys/fs/cgroup") / cg_path

        result = {}
        for fname in ("memory.max", "cpu.max", "cpuset.cpus"):
            fpath = base / fname
            if fpath.exists():
                result[fname] = fpath.read_text().strip()
        return result
    except Exception:
        return {}


def run_constrained(
    cmd: list[str] | str,
    *,
    cores: int = 4,
    mem_mb: int = 8192,
    cpu_quota_pct: int = 100,
    timeout_s: float | None = 300.0,
) -> ConstrainedResult:
    """Run command under cgroup v2 constraints via systemd-run --user + taskset.

    Args:
        cmd: Command to run (list or string)
        cores: Number of cores to pin via taskset (0 to cores-1)
        mem_mb: Memory limit in MB via MemoryMax
        cpu_quota_pct: CPU quota in systemd sense (100% = 1 core, 400% = 4 cores)
        timeout_s: Wall-clock timeout in seconds

    Returns:
        ConstrainedResult with timings, exit code, and applied constraints.

    Note:
        - CPUQuota=100% means ONE core's worth of time, not "unthrottled"
        - For 4 cores at full speed, pass cpu_quota_pct=400%
        - Exit 137 (SIGKILL) is reported as "OOM-killed at {mem_mb}MB"
    """
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)

    # Build taskset core list: 0 to cores-1
    core_list = ",".join(str(i) for i in range(cores))

    # systemd-run --user --scope -q -p MemoryMax=... -p CPUQuota=... --slice=sih3d.slice
    # Note: CPUQuota is in systemd's percentage units (100% = 1 CPU)
    systemd_cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p", f"MemoryMax={mem_mb}M",
        "-p", f"CPUQuota={cpu_quota_pct}%",
        "--slice=sih3d.slice",
        "taskset", "-c", core_list,
    ] + cmd_list

    applied = {
        "taskset_cores": core_list,
        "memory_max_mb": mem_mb,
        "cpu_quota_pct": cpu_quota_pct,
        "slice": "sih3d.slice",
    }

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            systemd_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        wall_time = time.perf_counter() - t0
        exit_code = proc.returncode

        # Read back actual cgroup limits
        cgroup_limits = _read_cgroup_limits()
        applied["cgroup_limits"] = cgroup_limits

        if exit_code == 137:
            status = f"OOM-killed at {mem_mb}MB"
        elif exit_code == 0:
            status = "ok"
        else:
            status = f"FAILED:{exit_code}"

        return ConstrainedResult(
            cmd=cmd_list,
            cores_requested=cores,
            mem_mb_requested=mem_mb,
            cpu_quota_pct_requested=cpu_quota_pct,
            wall_time_s=wall_time,
            exit_code=exit_code,
            status=status,
            applied_constraints=applied,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    except subprocess.TimeoutExpired as e:
        wall_time = time.perf_counter() - t0
        applied["cgroup_limits"] = _read_cgroup_limits()
        return ConstrainedResult(
            cmd=cmd_list,
            cores_requested=cores,
            mem_mb_requested=mem_mb,
            cpu_quota_pct_requested=cpu_quota_pct,
            wall_time_s=wall_time,
            exit_code=-1,
            status="TIMEOUT",
            applied_constraints=applied,
            stdout=e.stdout.decode() if e.stdout else "",
            stderr=e.stderr.decode() if e.stderr else "",
        )
    except FileNotFoundError:
        wall_time = time.perf_counter() - t0
        return ConstrainedResult(
            cmd=cmd_list,
            cores_requested=cores,
            mem_mb_requested=mem_mb,
            cpu_quota_pct_requested=cpu_quota_pct,
            wall_time_s=wall_time,
            exit_code=-1,
            status="FAILED:systemd-run_not_found",
            applied_constraints=applied,
            stderr="systemd-run not found in PATH",
        )
    except Exception as e:
        wall_time = time.perf_counter() - t0
        applied["cgroup_limits"] = _read_cgroup_limits()
        return ConstrainedResult(
            cmd=cmd_list,
            cores_requested=cores,
            mem_mb_requested=mem_mb,
            cpu_quota_pct_requested=cpu_quota_pct,
            wall_time_s=wall_time,
            exit_code=-1,
            status=f"FAILED:{type(e).__name__}",
            applied_constraints=applied,
            stderr=str(e),
        )