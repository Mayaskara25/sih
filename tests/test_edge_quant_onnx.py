"""Tests for edge/quantization.py + edge/onnx_inference.py (3D.2 / 3D.3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from edge.onnx_inference import ONNXRunner, round_trip_check
from edge.quantization import (
    accept_quantization,
    accuracy_delta,
    export_onnx,
    quantize_mixed,
)


class TinyUNet(nn.Module):
    """Conv-heavy stand-in with the same op families as LightUNet."""

    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.c3 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        return self.c3(torch.relu(self.c2(torch.relu(self.c1(x)))))


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return TinyUNet().eval()


def test_export_onnx_produces_loadable_model(tmp_path, tiny_model):
    sample = torch.randn(1, 4, 32, 32)
    path = export_onnx(tiny_model, sample, tmp_path / "m.onnx")
    assert path.exists()
    runner = ONNXRunner(path)
    assert runner.providers == ["CPUExecutionProvider"]


def test_round_trip_within_atol(tmp_path, tiny_model):
    sample = torch.randn(2, 4, 32, 32)
    path = export_onnx(tiny_model, sample, tmp_path / "m.onnx")
    report = round_trip_check(tiny_model, path, sample, atol=1e-4)
    assert report["max_abs_diff"] <= 1e-4


def test_round_trip_fails_when_atol_impossible(tmp_path, tiny_model):
    sample = torch.randn(1, 4, 8, 8)
    path = export_onnx(tiny_model, sample, tmp_path / "m.onnx")
    with pytest.raises(AssertionError, match="round-trip"):
        round_trip_check(tiny_model, path, sample, atol=0.0)


def test_runner_rejects_wrong_input_count(tmp_path, tiny_model):
    path = export_onnx(tiny_model, torch.randn(1, 4, 8, 8), tmp_path / "m.onnx")
    runner = ONNXRunner(path)
    with pytest.raises(ValueError, match="inputs"):
        runner.run(np.zeros((1, 4, 8, 8), np.float32),
                   np.zeros((1, 4, 8, 8), np.float32))


def test_explicit_thread_count_recorded(tmp_path, tiny_model):
    path = export_onnx(tiny_model, torch.randn(1, 4, 8, 8), tmp_path / "m.onnx")
    runner = ONNXRunner(path, intra_op_threads=2)
    assert runner.intra_op_threads == 2


def test_quantize_mixed_fp16_accuracy_retained_and_smaller(tmp_path, tiny_model):
    sample = torch.randn(1, 4, 64, 64)
    fp32_path = export_onnx(tiny_model, sample, tmp_path / "fp32.onnx")

    rng = np.random.default_rng(0)
    calibration = [rng.normal(size=(1, 4, 64, 64)).astype(np.float32) for _ in range(4)]
    quant_path = tmp_path / "mixed.onnx"
    report = quantize_mixed(fp32_path, quant_path, calibration,
                            fp16_ops=["Conv"], int8_ops=[])

    # MEASURED vs target -- assertions on facts we control here:
    assert report["measurement"] == "SIMULATED"
    assert report["fp16_applied"] and report["n_fp16_tensors"] > 0
    assert not report["int8_applied"]          # empty int8_ops must SKIP, not quantize-all
    assert report["quantized_bytes"] < report["fp32_bytes"]
    assert report["size_reduction_x"] > 1.0

    # Accuracy retained: quantized output within fp16 tolerance of fp32.
    r32 = ONNXRunner(fp32_path)
    rq = ONNXRunner(quant_path)
    x = rng.normal(size=(1, 4, 64, 64)).astype(np.float32)
    diff = np.abs(r32.run(x) - rq.run(x)).max()
    assert diff < 5e-2      # fp16-level drift, not silent degradation


def test_accuracy_delta_and_accept_rule():
    rng = np.random.default_rng(1)
    n_pos, n_neg = 200, 2000
    labels = np.zeros(n_pos + n_neg, dtype=bool)
    labels[:n_pos] = True

    fp32 = rng.normal(size=labels.shape)
    fp32[labels] += 2.0                       # good detector
    slightly_worse = fp32.copy()
    slightly_worse[labels] -= 0.03            # small degradation (<1% AUC)
    destroyed = rng.normal(size=labels.shape)  # silently degraded to noise

    d_ok = accuracy_delta(fp32, slightly_worse, labels)
    d_bad = accuracy_delta(fp32, destroyed, labels)

    assert d_ok["delta_auc"] > -0.05 and d_bad["delta_auc"] < -0.2
    assert accept_quantization(d_ok)          # <1% AUC loss accepted
    assert not accept_quantization(d_bad)     # silent INT8-style destruction rejected
