"""3D.6 -- edge benchmark harness (plan.md §6.4, "Definition of done").

Runs on ABU-Airport-1 (or a caller-supplied scene):
  1. `roi_vs_full_comparison` with global_rx as stage 1 and the trained
     LightUNet as stage 2 -> the <10%-pixels + 0.98-recall criterion,
     reported MET / NOT MET.
  2. Mixed FP16/INT8 quantization of the UNet -> MEASURED size/accuracy
     against the ~12x/~6x/>=99% literature target.

Every JSONL record written here is tagged `"measurement": "SIMULATED"`.
No Raspberry Pi exists; no number from this script is hardware evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.rx import global_rx                                   # noqa: E402
from edge.profiling import ProfileSink                             # noqa: E402
from edge.quantization import (                                    # noqa: E402
    accuracy_delta,
    accept_quantization,
    export_onnx,
    quantize_mixed,
)
from edge.roi_pipeline import roi_vs_full_comparison               # noqa: E402
from preprocessing.raster_loader import load_scene                 # noqa: E402
from segmentation.train_unet import LightUNet                      # noqa: E402

ABU_AIRPORT_1 = ROOT / "data" / "benchmark" / "abu" / "abu-airport-1.mat"
CKPT = ROOT / "experiments" / "seg_arch" / "unet_pretext.pt"
TRANSFORMER = ROOT / "experiments" / "seg_arch" / "reduce_bands_transformer.pkl"
OUT_DIR = ROOT / "experiments" / "edge_benchmarks"

# INT8 only where the output is about to be binarized or is covariance-safe;
# FP16 everywhere statistics-sensitive. The UNet's late decoder convs feed the
# 1x1 output conv whose logits get sigmoided + thresholded -> int8 candidates.
FP16_OPS = ["Conv", "MatMul", "Gemm"]      # conservative: all conv paths fp16
INT8_OPS: list[str] = []                   # measured first run keeps UNet fp16-only


def load_seg_model(in_channels: int = 30) -> LightUNet:
    model = LightUNet(in_channels=in_channels)
    model.load_state_dict(torch_load_strict(CKPT))
    model.eval()
    return model


def torch_load_strict(path: Path):
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


def run(scene_path: Path = ABU_AIRPORT_1, *, source: str = "abu",
        sink: ProfileSink | None = None) -> dict:
    import pickle

    cube, meta = load_scene(scene_path, source=source)
    with open(TRANSFORMER, "rb") as fh:
        transformer = pickle.load(fh)

    from scipy.io import loadmat
    gt = loadmat(str(scene_path))["map"].astype(bool)

    seg_model = load_seg_model()

    if sink is not None:
        from edge.profiling import profile_stage
        reports = {}
        with profile_stage("roi_vs_full", sink):
            reports["roi_vs_full"] = roi_vs_full_comparison(
                cube, meta, gt, global_rx, seg_model,
                transformer=transformer)
        sink.write_terminal("completed")
    else:
        reports = {"roi_vs_full": roi_vs_full_comparison(
            cube, meta, gt, global_rx, seg_model, transformer=transformer)}

    return reports


def run_quantization_benchmark(*, sink: ProfileSink | None = None) -> dict:
    """Export + mixed-quantize the UNet; measure size and score deltas on a
    synthetic patch set derived from the checkpoint's own input regime.

    Accuracy deltas here are measured on SYNTHETIC patches against the
    pretext-trained model -- a real detector-level delta needs the labelled
    scoring pipeline and is NOT claimed by this function.
    """
    import torch

    rng = np.random.default_rng(0)
    model = load_seg_model()
    sample = torch.randn(1, 30, 64, 64)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = export_onnx(model, sample, out_dir / "unet_fp32.onnx")
    calibration = [rng.normal(size=(1, 30, 64, 64)).astype(np.float32) for _ in range(8)]
    quant_report = quantize_mixed(fp32_path, out_dir / "unet_mixed.onnx",
                                  calibration, fp16_ops=FP16_OPS, int8_ops=INT8_OPS)

    # Measured score delta: same synthetic patches through torch vs quantized ONNX.
    from edge.onnx_inference import ONNXRunner
    runner = ONNXRunner(out_dir / "unet_mixed.onnx")
    patches = rng.normal(size=(32, 30, 64, 64)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(patches)).numpy()
    got = np.concatenate([runner.run(patches[i:i + 4]) for i in range(0, 32, 4)], axis=0)
    max_abs_diff = float(np.max(np.abs(ref - got)))
    quant_report["fp32_vs_quantized_max_abs_diff"] = max_abs_diff

    if sink is not None:
        from edge.profiling import ProfileRecord
        sink.write(ProfileRecord(
            stage="quantize_mixed_unet",
            wall_time_s=0.0, cpu_time_s=0.0, peak_rss_delta_mb=0.0, thread_count=1,
            extra={"report": quant_report},
        ))

    return quant_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(ABU_AIRPORT_1))
    ap.add_argument("--source", default="abu")
    ap.add_argument("--skip-roi", action="store_true")
    ap.add_argument("--skip-quant", action="store_true")
    args = ap.parse_args(argv)

    run_id = f"{uuid.uuid4().hex[:8]}"
    sink = ProfileSink(run_id)
    print(f"run_id={run_id}  scene={args.scene}")
    print("ALL NUMBERS SIMULATED -- constrained x86 host, not a Raspberry Pi.\n")

    ok = True
    if not args.skip_roi:
        reports = run(Path(args.scene), source=args.source, sink=sink)
        r = reports["roi_vs_full"]
        s2 = r["stage2"]
        acc = r["accept"]
        print("== ROI vs full ==")
        print(f"  stage-1 recall achieved : {r['recall_achieved']:.4f} "
              f"(target {r['target_recall']}, fp_rate {r['induced_fp_rate']:.4f})")
        print(f"  stage-2 pixels          : {s2['pixels_roi_path']}/{s2['pixels_total_scene']} "
              f"({100 * s2['fraction_processed_at_stage2']:.2f}% of scene)")
        if s2["latency_roi_s"] is not None:
            print(f"  latency ROI vs full     : {s2['latency_roi_s']:.3f}s vs "
                  f"{s2['latency_full_scene_s']:.3f}s ({s2['speedup_x']:.2f}x)")
        else:
            for note in s2["notes"]:
                print(f"  NOTE                    : {note}")
        bw = r["bandwidth"]
        print(f"  bandwidth ratio         : {bw['ratio_multiple']}x "
              f"(cube {bw['full_cube_bytes']} B -> GeoJSON {bw['geojson_bytes']} B)")
        verdict = "MET" if acc["criterion_met"] else "NOT MET"
        ok &= acc["criterion_met"]
        print(f"  ACCEPT CRITERION (<10% px @ >=0.98 recall): {verdict}\n")

    if not args.skip_quant:
        q = run_quantization_benchmark(sink=sink)
        print("== Quantization (measured, vs ~12x/~6x/>=99% target) ==")
        print(f"  size reduction          : {q['size_reduction_x']}x")
        print(f"  fp16 applied            : {q['fp16_applied']} "
              f"({q['n_fp16_tensors']} tensors)")
        print(f"  int8 applied            : {q['int8_applied']} {q['int8_op_types_present']}")
        print(f"  max abs diff vs fp32    : {q['fp32_vs_quantized_max_abs_diff']:.3e}")
        print(f"  notes                   : {'; '.join(q['notes'])}\n")

    print(f"JSONL: {sink.path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())