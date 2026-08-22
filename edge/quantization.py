"""3D.2 -- ONNX export + mixed FP16/INT8 quantization (plan.md §6.4).

MIXED, NEVER UNIFORM INT8: FP16 for covariance/statistics-sensitive stages
(RX matmuls, AE encoder, anything feeding a Cholesky) -- INT8 on a covariance
path destroys the conditioning and the detector degrades SILENTLY. INT8 only
for threshold-stage / late decoder convolutions whose output is about to be
binarized anyway. `accuracy_delta` makes the degradation measurable, and a
quantization costing >1% AUC is REJECTED by `accept_quantization`.

EXPORTER NOTE -- torch 2.13's default (dynamo) exporter requires
`onnxscript`, which is NOT installed in this environment (verified). Export
therefore goes through the legacy TorchScript exporter (`dynamo=False`),
which works. This is an environment fact, not a preference.

FP16 MECHANISM -- `onnxconverter_common` is not installed either, but
`onnxruntime.transformers.onnx_model.OnnxModel.convert_float_to_float16`
provides the same conversion (verified against a two-conv model before this
module was written: fp32-vs-fp16 max abs diff ~2.3e-4 on conv outputs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly.scoring import calibrate_threshold_for_recall

DEFAULT_TARGET_RECALL = 0.98          # §4.2's calibrated target -- not redefined here (3D.6 rule)
MAX_AUC_LOSS = 0.01                   # a quantization costing >1% AUC is rejected


# --------------------------------------------------------------------------- #
# export


def export_onnx(model: torch.nn.Module, sample_input: torch.Tensor, path: str | Path,
                *, opset: int = 17) -> Path:
    """Export a torch module to ONNX via the legacy exporter.

    The model is put in eval() mode first -- exporting with BatchNorm/dropout
    in train mode bakes training-time behaviour into the graph.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    try:
        torch.onnx.export(
            model, sample_input, str(path),
            opset_version=opset,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,   # legacy TorchScript exporter; default needs onnxscript (absent)
        )
    finally:
        if was_training:
            model.train()
    return path


# --------------------------------------------------------------------------- #
# calibration reader


class _ArrayCalibrationReader:
    """Minimal CalibrationDataReader over pre-collected input arrays."""

    def __init__(self, input_name: str, arrays: list[np.ndarray]):
        self._stream = ({input_name: np.asarray(a, dtype=np.float32)} for a in arrays)

    def get_next(self):
        return next(self._stream, None)


# --------------------------------------------------------------------------- #
# mixed quantization


def quantize_mixed(onnx_path: str | Path, out_path: str | Path,
                   calibration_data: list[np.ndarray], *,
                   fp16_ops: list[str], int8_ops: list[str]) -> dict:
    """Mixed FP16/INT8 quantization of an ONNX model.

    Order matters: INT8 static quantization runs FIRST (it rewrites weights of
    `int8_ops` node types to QInt8 with QDQ wrappers), then FP16 conversion
    runs over what remains, restricted via a block list to `fp16_ops` node
    types only. Nodes already quantized to INT8 are untouched by the FP16 pass.

    Returns a report dict (measured, never assumed): file sizes, which op
    types each precision actually landed on, and `"measurement": "SIMULATED"`.
    """
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.transformers.onnx_model import OnnxModel

    onnx_path = Path(onnx_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_model = onnx.load(str(onnx_path))
    input_name = src_model.graph.input[0].name

    # --- pass 1: INT8 static quantization restricted to int8_ops ----------
    # TRAP (deliberately avoided): quantize_static's op_types_to_quantize=None
    # means "quantize EVERYTHING" -- passing an empty list through as None
    # would silently produce uniform INT8, the exact degradation this module
    # exists to prevent. An empty int8_ops therefore SKIPS the INT8 pass.
    tmp_int8 = out_path.with_suffix(".int8.tmp.onnx")
    if int8_ops:
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(tmp_int8),
            calibration_data_reader=_ArrayCalibrationReader(input_name, calibration_data),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=list(int8_ops),
            per_channel=True,
        )
        int8_model = onnx.load(str(tmp_int8))
        has_qdq = any(n.op_type in ("QuantizeLinear", "DequantizeLinear")
                      for n in int8_model.graph.node)
        if not has_qdq:
            raise RuntimeError(
                f"INT8 pass requested for {sorted(int8_ops)} but no Q/DQ nodes "
                "landed in the graph -- quantize_static applied nothing. "
                "Reporting a requested-but-unapplied constraint would be the "
                "'plausible numbers that are simply wrong' failure mode."
            )
        # Under QDQ the original op keeps its type; presence of its QDQ wrapper
        # is what marks it as quantized, verified above via has_qdq.
        int8_present = sorted(set(int8_ops) & {n.op_type for n in int8_model.graph.node})
    else:
        int8_model = src_model
        import shutil
        shutil.copyfile(onnx_path, tmp_int8)
        int8_present = []

    int8_node_types = {n.op_type for n in int8_model.graph.node}

    # --- pass 2: FP16 for fp16_ops only ------------------------------------
    om = OnnxModel(int8_model)
    # Block every OP TYPE not requested for fp16, so conversion touches only
    # the statistics-sensitive subgraph (op_block_list is by op TYPE, which
    # keeps the rewritten graph topologically consistent -- a per-node block
    # list was tried first and produced a mis-sorted graph on this model).
    all_types = {n.op_type for n in om.model.graph.node}
    op_block_list = sorted(all_types - set(fp16_ops))
    om.convert_float_to_float16(keep_io_types=True, op_block_list=op_block_list)

    # The converter can leave inserted Cast nodes out of order (observed: the
    # graph-input cast landed after its first consumer), which onnx.checker
    # rejects. Sort before validating/saving.
    om.topological_sort()

    # convert_float_to_float16 generates boundary Cast nodes with derived
    # names; when ONE tensor feeds several consumers (every UNet skip
    # connection) it can emit duplicates, which onnx.load rejects outright
    # ("two nodes with same node name"). Node names are not edge identifiers
    # -- tensor names are -- so renaming is a safe repair.
    seen_names: dict[str, int] = {}
    for node in om.model.graph.node:
        if node.name in seen_names:
            seen_names[node.name] += 1
            node.name = f"{node.name}_dup{seen_names[node.name]}"
        else:
            seen_names[node.name] = 0
    om.topological_sort()

    # SSA repair: the converter emits one boundary Cast per (tensor, consumer),
    # and several of those casts share the same OUTPUT tensor name ("used as
    # output names multiple times"). Duplicates are identical casts of the
    # same input, so removing repeats is semantics-preserving: every consumer
    # already references the shared name, which the FIRST occurrence produces.
    seen_cast_keys: set[tuple[str, str]] = set()
    ssa_dupes = []
    for node in om.model.graph.node:
        if node.op_type != "Cast" or len(node.output) != 1:
            continue
        key = (node.input[0], node.output[0])
        if key in seen_cast_keys:
            ssa_dupes.append(node)      # identical cast already exists upstream
        else:
            seen_cast_keys.add(key)
    for node in ssa_dupes:
        om.model.graph.node.remove(node)

    onnx.checker.check_model(om.model)

    onnx.save(om.model, str(out_path))
    tmp_int8.unlink()

    # MEASURED, not inferred: op_type never changes under fp16 conversion, so
    # "Conv is fp16" is verified by counting FLOAT16 initializers in the graph.
    from onnx import TensorProto
    n_fp16_tensors = sum(1 for t in om.model.graph.initializer
                         if t.data_type == TensorProto.FLOAT16)

    fp32_bytes = onnx_path.stat().st_size
    quant_bytes = out_path.stat().st_size

    return dict(
        measurement="SIMULATED",
        fp32_bytes=fp32_bytes,
        quantized_bytes=quant_bytes,
        size_reduction_x=round(fp32_bytes / quant_bytes, 3) if quant_bytes else None,
        int8_op_types_requested=sorted(int8_ops),
        int8_applied=bool(int8_ops),
        int8_op_types_present=int8_present,
        fp16_op_types_requested=sorted(fp16_ops),
        n_fp16_tensors=n_fp16_tensors,
        fp16_applied=n_fp16_tensors > 0,
        n_calibration_samples=len(calibration_data),
        notes=[
            "INT8 static (QDQ, per-channel) applied only to requested op types.",
            "FP16 applied only to requested op types via op_block_list.",
            "Literature targets (~12x size, ~6x compute, >=99% accuracy retained) "
            "are TARGETS -- compare measured fields above, do not assume they were hit.",
        ],
    )


# --------------------------------------------------------------------------- #
# accuracy delta + accept/reject


def _metrics_from_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    """AUC, F1 and IoU at the §4.2 recall-calibrated operating point."""
    valid = ~np.isnan(scores)
    s, y = scores[valid].astype(np.float64), labels[valid].astype(bool)
    auc = float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else float("nan")
    thr, _fp = calibrate_threshold_for_recall(s, y, target_recall=DEFAULT_TARGET_RECALL)
    pred = s >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"roc_auc": auc, "f1": f1, "iou": iou}


def accuracy_delta(fp32_scores: np.ndarray, quantized_scores: np.ndarray,
                   labels: np.ndarray) -> dict:
    """AUC/F1/IoU deltas (quantized - fp32) at the same operating-point rule."""
    m32 = _metrics_from_scores(fp32_scores, labels)
    mq = _metrics_from_scores(quantized_scores, labels)
    return {
        "fp32": m32,
        "quantized": mq,
        "delta_auc": mq["roc_auc"] - m32["roc_auc"],
        "delta_f1": mq["f1"] - m32["f1"],
        "delta_iou": mq["iou"] - m32["iou"],
        "measurement": "SIMULATED",
    }


def accept_quantization(delta_report: dict, *, max_auc_loss: float = MAX_AUC_LOSS) -> bool:
    """A quantization costing more than `max_auc_loss` AUC is rejected."""
    return bool(delta_report["delta_auc"] >= -max_auc_loss)


def size_report(fp32_path: str | Path, quant_path: str | Path) -> dict:
    """Measured on-disk sizes -- the only honest 'size ratio' there is."""
    fp32_path, quant_path = Path(fp32_path), Path(quant_path)
    return dict(
        measurement="SIMULATED",
        fp32_bytes=fp32_path.stat().st_size,
        quantized_bytes=quant_path.stat().st_size,
        ratio=round(fp32_path.stat().st_size / quant_path.stat().st_size, 3),
    )


def report_to_jsonl(report: dict) -> str:
    return json.dumps(report, separators=(",", ":"), default=str)