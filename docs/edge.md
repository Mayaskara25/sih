# Branch 3D — Edge / Systems: what was actually measured

**Status: implemented. Every number below is `SIMULATED` — there is no
Raspberry Pi and no instrumented hardware (plan.md §6.4, §9). No power figure
appears anywhere in this branch because no wattmeter exists; an estimated
wattage presented as measured would be a fabrication.**

## Host

AMD Ryzen 5 3550H, 8 logical cores, ~13 GB RAM, **no swap**, cgroup v2
(`cgroup2fs`), user-slice delegated controllers `cpu memory pids` only — no
`cpuset`. All verified on this machine before any code was written
(`docs/edge_branch_plan.md`'s environment table); nothing here was assumed.

## Modules built (in the plan's build order)

| module | what it does | tests |
|---|---|---|
| `edge/profiling.py` | `profile_stage` contextmanager (wall/CPU time, RSS delta, thread count), `ProfileSink` → `experiments/edge_benchmarks/{run_id}.jsonl` with a terminal record so truncated runs are detectable, `regression_check` at tol=0.15 | `tests/test_edge_profiling.py` |
| `edge/constrained_sim.py` | `run_constrained(cmd, cores, mem_mb, cpu_quota_pct)` via `systemd-run --user --scope` + `taskset`; **exit 137 reported as `OOM-killed at {mem_mb}MB`, not generic FAILED** | `tests/test_edge_constrained_sim.py` |
| `edge/streaming.py` | `StripPipeline.register/run` with per-stage halo (`lookahead_rows`) resolved recursively off `anomaly.streaming_rx._StripSource`; hard RSS ceiling raising `MemoryBudgetExceeded` from psutil sampling **plus the kernel's `ru_maxrss` high-water mark** | `tests/test_edge_streaming.py` |
| `edge/quantization.py` | `export_onnx` (legacy exporter — the dynamo one needs `onnxscript`, absent here), `quantize_mixed` (INT8 QDQ restricted to requested op types; FP16 via ORT transformers restricted by `op_block_list`), `accuracy_delta`, `accept_quantization` (>1% AUC loss rejected) | `tests/test_edge_quant_onnx.py` |
| `edge/onnx_inference.py` | `ONNXRunner`, CPUExecutionProvider only, explicit thread counts; `round_trip_check` at atol=1e-4 | same file as above |
| `edge/roi_pipeline.py` + `edge/benchmark.py` | `roi_vs_full_comparison` + runnable harness writing SIMULATED JSONL | `tests/test_edge_roi_pipeline.py`, `tests/test_edge_benchmark.py` |

## Measured results

### ROI vs full — ABU-Airport-1 (`python -m edge.benchmark`)

| quantity | measured |
|---|---|
| stage-1 recall at §4.2 threshold | **0.9861** (target 0.98 — met) |
| induced FP rate at that recall | 0.8782 |
| stage-2 pixel fraction | **3.69× the scene** (36 864 window-pixels vs 10 000 scene pixels) |
| bandwidth ratio | **73.65×** (8 200 000 B cube → 111 331 B GeoJSON) |
| **accept criterion (<10% px @ ≥0.98 recall)** | **NOT MET** |

Why it is not met: ABU-Airport-1 is 100×100. At patch=64 the full-scene grid
is 2×2 windows; calibrating to recall 0.98 flags 87.8% of background pixels,
and each flagged connected component's bbox expands to its own 64×64 window
(`segmentation/infer.py`'s expand-don't-pad rule). Scattered false positives
therefore cost MORE stage-2 windows than running the full grid. The cascade's
economics only work on scenes large relative to the patch size; this scene is
too small for the claim to hold, and the harness reports that rather than
tuning the criterion until it passes.

Stage-2 model latency is `None` with a recorded note: the pretext UNet's fitted
PCA transformer expects 184 harmonized bands and ABU ships 205 raw bands with
no wavelength array — it can neither be harmonized nor legally transformed
(plan.md D19 suspended UNet-on-ABU for exactly this reason). Pixel fractions
are geometry-only (window counts) and do not depend on the model.

### Quantization — LightUNet (`unet_pretext.pt`)

| quantity | measured | literature target |
|---|---|---|
| size reduction | **1.82×** (7 770 837 B → 4 277 782 B) | ~12× |
| fp16 tensors in graph | 30 (all Conv paths) | — |
| INT8 applied | **none** (none requested for this model — no binarization-safe subgraph identified yet) | — |
| max abs output diff vs fp32 | 2.16e-02 over 32 synthetic patches | — |

State vs target, honestly: the ~12×/~6×/≥99% figures are literature numbers
for uniform INT8 pipelines; this run measured an FP16-only conversion at
1.82×. The target was NOT hit and is not claimed. Uniform INT8 was rejected
by design (it destroys covariance conditioning silently); identifying which
UNet subgraphs are INT8-safe remains open work.

### Constrained execution

`run_constrained` wraps commands in
`systemd-run --user --scope -p MemoryMax=…M -p CPUQuota=N% --slice=sih3d.slice taskset -c …`
and reads the child's actual `memory.max`/`cpu.max` back from cgroupfs rather
than trusting the flags. Exit 137 is surfaced as `OOM-killed at {mem_mb}MB`.
A cgroup cap SIGKILLs silently on this machine (verified before coding:
exit 137, no traceback); nothing downstream may rely on catching MemoryError.

## What a throttled x86 core does NOT tell you about a Pi

Explicitly, since this branch exists to bound deployment behaviour:

1. **Nothing here measures energy or power.** No wattmeter exists. Any
   joules/watts figure quoted from these runs would be invented.
2. **x86 SIMD width and cache hierarchy are not ARM's.** NEON lanes,
   Cortex-A76/A72 memory latency, and the Pi's shared-bandwidth LPDDR are all
   different machines. A 4-core quota on Zen+ says little about 4 real A76s.
3. **Thermal envelope differs.** A Pi throttles under sustained load; this
   laptop does not at these duty cycles.
4. **I/O is incomparable.** SD-card read throughput dominates strip-streamed
   inference on a Pi; NVMe here makes streaming look free.
5. **The OS differs.** RPi OS ships different BLAS/OpenMP defaults; every
   number here pins threads explicitly precisely so cross-machine comparison
   stays possible later.

These runs are therefore useful for RELATIVE comparisons (ROI vs full, fp16
vs fp32, ceiling interactions) and regression guarding only.

## Environment facts discovered while building (recorded so nobody re-derives them)

* torch's default ONNX exporter needs `onnxscript` (not installed) → export
  uses `dynamo=False` (legacy TorchScript exporter).
* `onnxconverter_common` is absent, but ORT's vendored
  `OnnxModel.convert_float_to_float16` works — except it (a) emits duplicate
  node names and SSA-violating Cast outputs on UNet skip connections, and
  (b) can leave inserted Casts unsorted. `quantize_mixed` repairs both
  explicitly and validates with `onnx.checker` before saving.
* `op_types_to_quantize=None` in `quantize_static` means "quantize
  EVERYTHING" — an empty int8 op list must SKIP the pass, not fall through to
  None (this exact slip would have produced silent uniform INT8).
* `np.zeros` is lazy under Linux overcommit: an untouched 2.5 GB allocation
  never appears in RSS. The pipeline's ceiling check therefore also reads
  `ru_maxrss`, which survives frees.
