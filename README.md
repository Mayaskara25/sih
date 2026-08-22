# sih — AI-Based Hyperspectral Anomaly Detection & Geospatial Semantic Mapping

Detects anomalies in hyperspectral imagery and turns them into georeferenced ROIs
(GeoJSON, openable in QGIS). Classical detectors (RX family, CRD) plus learned
segmentation, with an edge-deployment arm and a quantum-feature-map arm.

**Read `plan.md` before writing code.** It is the executable spec — architecture,
interfaces, and every design decision with its evidence. This README only gets you
running.

---

## Setup

**Python is pinned to 3.12.13.** Not a preference — `fiona==1.10.1` has no cp314
wheel, so 3.14 fails at install. See D1.

```bash
git clone git@github.com:Mayaskara25/sih.git && cd sih
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

The suite passes on a fresh clone. Tests needing datasets **skip** rather than fail,
so you can develop without downloading anything.

## Data is not in the repo

`data/` is gitignored — ~14 GB, and some of it is licence-restricted.

| dataset | size | needed for |
|---|---|---|
| HAD100 | 7.7 G | the background pool; 3B training |
| ABU / HYDICE / Indian Pines | ~50 M | scoring, Phase 2 dev |
| harmonized pool (generated) | 6.3 G | 3B |
| EnMAP L2A | ~3 G | Phase 5 Level 2 only |

See `docs/datasets.md` for what each contains and `docs/enmap_handover.md` for EnMAP
access. **You do not need any of it to contribute** to the branches listed in
`CONTRIBUTING.md`.

## Layout

```
core/contracts.py     the shared boundary — SceneMeta, validate_scene, ROIRecord
preprocessing/        raster_loader · normalize · harmonize · background_pool
anomaly/              rx · scoring
segmentation/         postfilter
geospatial/           polygonize · projections · geojson
pipeline/             run_pipeline
change_detection/ edge/ quantum/    empty — open for contribution
scripts/              verify_* and fetch_* — run these, do not skip them
docs/                 dataset facts, onboarding, EnMAP handover
```

## Run the pipeline

```bash
.venv/bin/python -m pipeline.run_pipeline \
    --scene data/benchmark/indian_pines/Indian_pines_corrected.mat \
    --source indian_pines --detector fused --out experiments/demo
```

`--detector` resolves through a registry, so swapping detectors is a config
edit rather than a code change (§4.1). Available: `global_rx`, `local_rx`,
`kernel_rx`, `crd`, `fused`. `streaming_rx` is deliberately **not** in that
registry — it takes a scene path rather than a cube, because its whole purpose
is never materializing the cube; call it directly.

Per-dataset detector parameters go through `--detector-params` as JSON, never
hardcoded (§3A.2 — HAD100's 64×64 patches want a smaller annulus than a
150×150 ABU scene, and carrying one to the other measurably reranks the
detectors).

## Run the demo

```bash
.venv/bin/python pipeline/demo.py --assert-offline
```

Runs the full §10 sequence on a real HAD100/AVIRIS scene and prints all eleven
steps. Takes about 15 seconds; writes GeoJSON plus `demo_summary.json` to
`experiments/demo/`. Useful flags: `--scene <path.hdr>` to pick a scene,
`--profile landcover`, `--target-recall 0.99`, `--out <dir>`.

**`--assert-offline` genuinely enforces.** It replaces `socket.socket` for the
whole inference stage, so any connection attempt raises `OfflineViolation`
rather than being counted and reported afterwards. §10 step 8 is explicit that
claiming offline operation without proving it is the weakest possible version
of this demo.

**Steps 10 and 11 print SKIPPED with a reason.** Change detection (3C) and the
quantum branch (3E) are P2 in §11.1 and were never built. The demo says so
rather than fabricating output.

## Run the benchmark

```bash
.venv/bin/python scripts/run_benchmark.py --datasets abu,hydice,had100
```

Writes `experiments/rx_vs_ae/{report.md,results.csv,results_pooled.csv}` and a
false-negative log to `experiments/cascade_recall_audit/`. Add `--strict` to
exit non-zero if any detector failed on any scene.

Every pooled figure is emitted twice: **scene-macro (primary)** and
pixel-micro (labelled). ABU's anomaly density spans a 32× range, so an
unlabelled "pooled" number is effectively a report on two scenes — §3A.10
bans one and this harness never produces one.

## The project's one rule

Dataset facts are either **verified against the files** or **documentation-only**,
and `docs/datasets.md` says which. Assume documentation is wrong until checked —
HAD100's project page was wrong in five ways (D11), ABU and HYDICE in three more
(D13), and DLR's own docs described a portal that had been dead five months (§8.0a).

`scripts/verify_had100.py` and `scripts/verify_benchmarks.py` exit non-zero on drift.
Run them after any dataset change.

## Credentials

Secrets live at `~/.config/sih/credentials.env` (mode 600), **outside this repo**.
`.env.example` lists variable names only. Never print that file — see `CLAUDE.md`.

```bash
.venv/bin/python scripts/check_credentials.py    # booleans only, never values
```

## Contributing

See **`CONTRIBUTING.md`** — what's open, what's claimed, and the one file nobody
edits alone.
