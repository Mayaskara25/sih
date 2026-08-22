"""Tests for edge/benchmark.py (3D.6 harness).

The full ABU-Airport-1 acceptance run is `skipif`'d on the data file so a
fresh clone stays green (repo convention).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import edge.benchmark as bench
from edge.profiling import ProfileSink

ROOT = Path(__file__).resolve().parents[1]
ABU_AIRPORT_1 = ROOT / "data" / "benchmark" / "abu" / "abu-airport-1.mat"
CKPT = ROOT / "experiments" / "seg_arch" / "unet_pretext.pt"
TRANSFORMER = ROOT / "experiments" / "seg_arch" / "reduce_bands_transformer.pkl"


def test_run_quantization_benchmark_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "OUT_DIR", tmp_path)
    report = bench.run_quantization_benchmark()
    assert report["measurement"] == "SIMULATED"
    assert report["fp32_bytes"] > 0
    assert report["quantized_bytes"] > 0
    assert report["size_reduction_x"] > 1.0
    assert "fp32_vs_quantized_max_abs_diff" in report


@pytest.mark.skipif(not ABU_AIRPORT_1.exists(), reason="ABU benchmark not fetched")
@pytest.mark.skipif(not CKPT.exists() or not TRANSFORMER.exists(),
                    reason="UNet checkpoint or PCA transformer missing")
def test_edge_benchmark_abu_airport_1_end_to_end(tmp_path, monkeypatch):
    """The branch's actual claim, run on the real scene. Reports MET/NOT MET;
    the assertion here is that the harness RUNS and reports honestly, not
    that the criterion is met."""
    monkeypatch.setattr(bench, "OUT_DIR", tmp_path)

    sink = ProfileSink("test_abu", base_dir=tmp_path)
    reports = bench.run(ABU_AIRPORT_1, sink=sink)

    r = reports["roi_vs_full"]
    assert r["measurement"] == "SIMULATED"
    assert isinstance(r["accept"]["criterion_met"], bool)

    # Every JSONL record carries the SIMULATED tag -- spot-check the file.
    lines = [json.loads(l) for l in sink.path.read_text().splitlines() if l.strip()]
    assert lines
    for rec in lines:
        if "measurement" in rec:
            assert rec["measurement"] == "SIMULATED"

    print(f"\nABU-Airport-1 accept criterion met: {r['accept']['criterion_met']}")
    print(f"  recall={r['recall_achieved']:.4f}  "
          f"stage2 fraction={r['stage2']['fraction_processed_at_stage2']:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])