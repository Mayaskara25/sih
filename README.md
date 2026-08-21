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
  --scene data/benchmark/indian_pines/Indian_pines_corrected.mat
```

Writes score rasters, a mask, ROIs as GeoJSON, and a run manifest to
`experiments/phase2/`.

> Indian Pines has **no CRS and no wavelengths**. Its affine is synthesised and
> labelled `georef: "synthetic"` (D2). It verifies transform *plumbing*, never
> real-world accuracy. Real georeferencing is verified on HAD100 (D11.5, D15).

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
