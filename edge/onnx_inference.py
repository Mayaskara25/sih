"""3D.3 -- ONNX CPU inference (plan.md §6.4).

**CPUExecutionProvider ONLY.** onnxruntime 1.29.0 here is the CPU build; no
GPU provider exists in it and `onnxruntime-gpu` must NOT be added. The
provider list is asserted, not defaulted: if CUDAExecutionProvider ever
appears in this list something installed a GPU runtime underneath us and the
numbers stop being comparable.

Thread counts are EXPLICIT, never default -- the ORT default reads the host
core count, which makes every number incomparable across machines (and this
host has 8 cores; the target device class being simulated has fewer).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXRunner:
    """Thin wrapper enforcing CPU-only execution and explicit thread counts."""

    def __init__(self, model_path: str | Path, *, intra_op_threads: int = 1,
                 inter_op_threads: int = 1, graph_optimize: bool = True):
        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if not graph_optimize:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        # Explicit provider pinning -- never pass an empty/default list.
        available = ort.get_available_providers()
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(
                f"CPUExecutionProvider unavailable; providers={available}. "
                "Do NOT install onnxruntime-gpu to fix this."
            )
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.intra_op_threads = intra_op_threads

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def run(self, *input_arrays: np.ndarray) -> np.ndarray:
        """Run with inputs bound positionally to the session's input order.
        Returns the first output."""
        if len(input_arrays) != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} inputs {self.input_names}, "
                f"got {len(input_arrays)}"
            )
        feed = {name: np.asarray(a) for name, a in zip(self.input_names, input_arrays)}
        outputs = self.session.run(self.output_names, feed)
        return outputs[0]


def round_trip_check(torch_model, onnx_path: str | Path, sample_input,
                     *, atol: float = 1e-4) -> dict:
    """ONNX output must match torch within `atol=1e-4` (the same criterion
    §3A.6 sets). Returns a measured report; raises AssertionError on breach.
    """
    import torch

    runner = ONNXRunner(onnx_path)
    torch_model.eval()
    with torch.no_grad():
        ref = torch_model(sample_input)
    ref_np = ref.detach().cpu().numpy()
    out = runner.run(sample_input.detach().cpu().numpy())

    max_abs = float(np.max(np.abs(ref_np - out)))
    assert max_abs <= atol, (
        f"ONNX round-trip mismatch: max abs diff {max_abs:.3e} > atol {atol:.1e}"
    )
    return dict(measurement="SIMULATED", max_abs_diff=max_abs, atol=atol,
                intra_op_threads=runner.intra_op_threads,
                providers=runner.providers)