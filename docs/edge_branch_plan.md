# Branch 3D — Edge / Systems: implementation brief for an external agent

**Status: not started.** `edge/` contains one empty `__init__.py`.
Spec is `plan.md` §6.4 (read it, plus §0.3 and §9). This file is the executable brief:
what to build, what is already verified about the machine, and what will go wrong.

## The one rule that overrides everything

**There is no Raspberry Pi and no instrumented hardware.** Every number this branch produces
is `SIMULATED` and must be labelled so in the artefact itself, not only in prose. Phase 6
**Tier B is BLOCKED, not deferred** — do not write code against assumed hardware. Never report
power: no wattmeter exists, and an estimated wattage presented as measured is a fabrication
(`plan.md` §9, §13 rule 1).

## Environment — verified 2026-08-22 on this machine, not assumed

| fact | value | consequence |
|---|---|---|
| `onnx` | 1.22.0 | installed, no fetch needed |
| `onnxruntime` | 1.29.0 | CPU provider only; do **not** add `onnxruntime-gpu` |
| `psutil` | 7.2.2 | RSS sampling available |
| `torch` | 2.13.0+cu130 | export source; inference must not need CUDA (§0.3) |
| cgroup | **v2** (`cgroup2fs`) | `run_constrained` can use cgroups |
| root controllers | `cpuset cpu io memory hugetlb pids rdma misc dmem` | |
| **user-slice delegated controllers** | **`cpu memory pids` only — no `cpuset`** | Pin cores with `taskset`, cap memory/CPU with `systemd-run --user`. **See the cgroup section below — it is all verified, do not rediscover it.** |
| host | 8 logical cores, ~13 GB RAM, no swap | a 6 GB RSS ceiling has real headroom; the machine will OOM-kill rather than swap (this already happened once — `plan.md` D17) |

## cgroup v2 on this machine — everything `run_constrained` needs, verified 2026-08-22

Do not spend time discovering this. Every line below was run on this host and its output recorded.

### What works, and what does not

| mechanism | verdict | evidence |
|---|---|---|
| `systemd-run --user --scope -p MemoryMax=… -p CPUQuota=…` | **works, unprivileged — use this** | process read back `memory.max = 536870912`, `cpu.max = 50000 100000` |
| `taskset -c 0-3 <cmd>` | **works** | `taskset -c 0-3 nproc` → `4` |
| manual `mkdir` under `user@1000.service/` | works; delegated controllers `cpu memory pids` | `mkdir` OK, `cgroup.controllers` → `cpu memory pids` |
| **`cpuset` (core pinning via cgroup)** | **NOT AVAILABLE** | a user cgroup contains only `cpu.max`, `cpu.max.burst`, `memory.max` — there is **no `cpuset.cpus` file at all**, and writing it fails |
| catching the memory cap as `MemoryError` | **DOES NOT HAPPEN** | see below |

### The recommended call

```bash
systemd-run --user --scope -q     -p MemoryMax=8192M -p CPUQuota=100% --slice=sih3d.slice     taskset -c 0-3 <cmd>
```

`taskset` supplies the core pinning that `cpuset` cannot, and `systemd-run --user` supplies the
memory and CPU caps. Neither needs root. Note `CPUQuota=100%` means *one core's worth of time*,
not "unthrottled" — for 4 cores at full speed pass `CPUQuota=400%`, and make
`cpu_quota_pct` in the signature mean the systemd sense so the two do not silently disagree.

To read back what a child actually got (worth asserting in a test, rather than trusting the flags):

```python
cg = open("/proc/self/cgroup").read().split("::")[1].strip()
base = f"/sys/fs/cgroup{cg}"
mem  = open(f"{base}/memory.max").read().strip()   # bytes, or "max"
cpu  = open(f"{base}/cpu.max").read().strip()      # "<quota> <period>", or "max 100000"
```

### The trap that will actually cost you a day

**A cgroup memory cap does not raise `MemoryError`. It SIGKILLs the process, silently.**

Measured: a Python child under `MemoryMax=256M` allocating past the cap produced **exit code 137**
(128 + 9 = SIGKILL), **no traceback, no stdout, no stderr**. The `except MemoryError` branch never
ran.

Three consequences, all binding on 3D.1 and 3D.5:

1. `StripPipeline`'s `MemoryBudgetExceeded` **cannot** be implemented by catching an allocation
   failure. Sample RSS with `psutil` between stages and raise **before** the cap is reached. Set
   the cgroup cap *above* the pipeline's own ceiling so the pipeline's error fires first and the
   kernel's kill is the backstop, not the mechanism.
2. `run_constrained` must treat **exit 137 as a distinct, reported outcome** — "OOM-killed at
   `mem_mb`" — not as a generic non-zero exit. Returning `status="FAILED"` for it would lose
   exactly the fact the run existed to establish.
3. A silent kill mid-benchmark leaves a **truncated** JSONL file that parses fine. `ProfileSink`
   should write a terminal record on clean completion so a truncated run is detectable, rather
   than reading as a short-but-valid one.

This machine has **no swap**, so there is no gradual degradation before the kill — and the project
has already been OOM-killed once building the background pool (`plan.md` D17).

### Degrade explicitly, never silently

If a constraint cannot be applied — `cpuset` absent, `systemd-run` missing, cgroup v2 not mounted —
the run must **record which constraints were actually applied** in its output and, for a benchmark
run, fail rather than proceed. An unconstrained run that reports itself as constrained produces
plausible numbers that are simply wrong, which is this project's single most repeated failure mode
(D22.2, D24, D26, D28). Put the applied-constraints dict in every `experiments/edge_benchmarks/*.jsonl`
record next to `"measurement": "SIMULATED"`.

## Build order

`3D.4 profiling` → `3D.5 constrained_sim` → `3D.1 streaming` → `3D.2 quantization` →
`3D.3 onnx_inference` → `3D.6 roi_pipeline` + `benchmark`.

3D.4 and 3D.5 have **no dependencies** and are named in §11.1 as start-immediately.
Everything downstream reports through 3D.4, so building it first means the later modules are
instrumented from birth rather than retrofitted.

### 3D.4 `edge/profiling.py` — build first
`profile_stage(name, sink)` contextmanager recording wall time, CPU time, peak RSS delta
(`tracemalloc` + `psutil`), and thread count. `ProfileSink` appends JSONL to
`experiments/edge_benchmarks/{run_id}.jsonl`; **every record carries `"measurement": "SIMULATED"`**.
`regression_check(baseline, current, *, tol=0.15)` returns the list of stages that regressed
>15% against a committed baseline.

Set `onnxruntime` and BLAS thread counts **explicitly**, never by default — the default reads
the host core count, which makes every number incomparable across machines. Record the value
used in each JSONL record.

### 3D.5 `edge/constrained_sim.py`
`run_constrained(cmd, *, cores=4, mem_mb=8192, cpu_quota_pct=100)`. `taskset -c 0-3` for cores
(see the cgroup table above), cgroup v2 `memory.max` / `cpu.max` for the rest. Returns timings
tagged `measurement="SIMULATED"` and `host_cpu=<model>`.

This is a **regression guard and a relative comparison**. A throttled x86 core is not a
Cortex-A76. Nothing from this module may be presented as a Pi number.

### 3D.1 `edge/streaming.py`
`StripPipeline` with `register(name, fn, *, lookahead_rows=0)` and `run(scene_path, *, strip_rows=16)`.
The scheduler feeds overlapping strips so a stage needing *k* rows of context gets its halo
without the caller managing it. Hard RSS ceiling (default 6 GB) raising `MemoryBudgetExceeded`
**rather than swapping** — a pipeline that silently swaps invalidates every latency number
taken from it.

Reference implementation to match: `anomaly/streaming_rx.py`, which already streams a scene
and uses **float64 Welford/Chan accumulators**. `plan.md` **D24** records that the *reference*
`global_rx` was the imprecise side because it accumulated in float32 — so if a new streaming
stage disagrees with its batch equivalent, **suspect both**, and measure which one is wrong
before loosening a tolerance.

### 3D.2 `edge/quantization.py`
`export_onnx(model, sample_input, path, *, opset=17)` and
`quantize_mixed(onnx_path, out_path, calibration_data, *, fp16_ops, int8_ops)`.

**Mixed FP16/INT8, never uniform INT8.** FP16 for covariance/statistics-sensitive stages
(RX matmuls, AE encoder, anything feeding a Cholesky) — INT8 on a covariance path destroys the
conditioning and the detector **degrades silently**. INT8 only for threshold-stage and late
decoder convolutions whose output is about to be binarized anyway.

Literature target: ~12× size, ~6× compute, ≥99% FP32 accuracy retained. **State
measured-vs-target; do not assume the target was hit.** `accuracy_delta(fp32, quantized, labels)`
returns AUC/F1/IoU deltas, and a quantization costing >1% AUC is **rejected**.

The model to export is `experiments/seg_arch/unet_pretext.pt` (`segmentation/train_unet.py::LightUNet`,
`in_channels=30`), which is trained and on disk.

### 3D.3 `edge/onnx_inference.py`
`ONNXRunner`, **`CPUExecutionProvider` only**, explicit thread count. Round-trip test: ONNX
output matches torch within `atol=1e-4` (the same criterion §3A.6 sets).

### 3D.6 `edge/roi_pipeline.py` + `edge/benchmark.py`
`roi_vs_full_comparison(scene, detector, seg_model)` measures the branch's actual claim:
full-scene vs ROI-only latency, pixels processed at stage 2 vs total, % discarded by screening,
and bandwidth as full-cube bytes vs transmitted GeoJSON bytes.

**Accept:** on ABU-Airport-1, the ROI path processes **<10% of pixels at stage 2** *while
stage-1 recall meets §4.2's calibrated `target_recall = 0.98`*. This branch **does not define its
own recall floor** — two recall numbers in two sections is how a cascade quietly ships at the
weaker one. Use `anomaly/scoring.py::calibrate_threshold_for_recall`, which already returns
`(threshold, fp_rate)`. Report the bandwidth ratio as an explicit multiple.

## Reuse — do not reimplement

| need | use |
|---|---|
| detectors | `anomaly/{rx,local_rx,kernel_rx,crd,streaming_rx}.py` |
| threshold calibration | `anomaly.scoring.calibrate_threshold_for_recall` |
| segmentation inference | `segmentation/infer.py`, checkpoint `experiments/seg_arch/unet_pretext.pt` |
| band reduction | `preprocessing.harmonize.reduce_bands`; **fitted** transformer at `experiments/seg_arch/reduce_bands_transformer.pkl` — apply it, never refit (D15) |
| contracts | `core/contracts.py` — `validate_scene`, `validate_score_raster`, `validate_roi` |
| scene loading | `preprocessing/raster_loader.py::load_scene`; benchmark enumeration in `scripts/run_benchmark.py` |
| macro/micro pooling | `scripts/run_benchmark.py::pool` — an unlabelled pooled figure is banned |
| ROI extraction | `geospatial/polygonize.py::mask_to_rois`, `geospatial/geojson.py` |

## Conventions this repo enforces

- Detector signature is frozen (`CONTRIBUTING.md`): `f(cube, *, ...) -> [H,W] float32`, NaN in → NaN out positionally, never form an explicit inverse (`cho_factor`/`cho_solve`).
- Seeding is **explicit per function** (`seed: int`, `np.random.default_rng(seed)`). No global seeds.
- Tests live flat in `tests/`, named `test_<module>.py`. `skipif` anything needing `data/` — a fresh clone must stay green. **Write the failure test, not just the success test.**
- Rows with `status != "ok"` are kept, never dropped. A crashing stage is a finding.
- Dataset facts come from opening the file, never from a project page (`CLAUDE.md`).
- Pre-PR: `pytest -q`, `scripts/verify_had100.py`, `scripts/verify_benchmarks.py` — all must pass.

## Traps this project has already paid for — all in `plan.md`

1. **D22 / D22.2** — an *absolute* `reg * I` ridge is scale-blind. D22 crashed on 3/13 ABU scenes; D22.2 failed **silently**, pinning a +50σ outlier at rank 481. Build regularizers from the data (`reg * trace(S)/b * I`). **D28** is the same lesson a third time.
2. **D24** — float32 covariance accumulation. When a tolerance fails, **the reference is a suspect too**.
3. **D27.7** — an unweighted balanced-subsample PR-AUC read 0.87 where the truth was 0.34. Any subsampled metric needs `sample_weight`.
4. **Green tests prove very little here.** Six defects were found by integration with a fully passing suite, because every test asserted against data the model had already seen. Run the thing end to end on real scenes.
5. **A check that passes for the wrong reason is worse than one that fails** (D26). `import qgis` succeeded against an empty directory for exactly this reason.

## Definition of done

- `edge/` modules built with tests; full suite green.
- `experiments/edge_benchmarks/*.jsonl` exists, **every record tagged `SIMULATED`**.
- `roi_vs_full_comparison` run on ABU-Airport-1 with the <10%-pixels + 0.98-recall criterion reported as **met or not met**, not as a target.
- Quantization reports **measured** size/compute/accuracy against the ~12×/~6×/≥99% target.
- A short `docs/edge.md` recording what was measured, on what host, and — explicitly — what a constrained x86 core does **not** tell you about a Pi.
