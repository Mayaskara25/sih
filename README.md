# sih — AI-Based Hyperspectral Anomaly Detection & Geospatial Semantic Mapping

Finds anomalies in hyperspectral imagery and turns them into georeferenced polygons
you can open in QGIS. Classical detectors (RX family, CRD), learned segmentation on
ROI crops only, plus change-detection, edge-deployment and quantum research arms.

**New here? Read [`docs/results.md`](docs/results.md) first.** It's one page: what
works, what doesn't, and where the evidence for each claim lives. Four of our arms
missed their own accept criteria and we report all four — that page is the honest
summary, and it will save you from quoting a number that has a caveat attached.

**Writing code? Read `plan.md`.** It's the executable spec: architecture, interfaces,
and every design decision with the evidence behind it (D1–D35).

---

## Quick start (15 minutes, no datasets needed)

```bash
git clone git@github.com:Mayaskara25/sih.git && cd sih
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # expect: 545 passed
```

**Python must be 3.12.** Not a preference — `fiona==1.10.1` has no cp314 wheel, so
3.14 fails at install (D1). If `python3.12` isn't on your machine, install it before
anything else.

The suite passes on a fresh clone with **no data downloaded**. Tests that need
datasets *skip* rather than fail, so you can read, run and contribute immediately.

If you see fewer than 545 passing, something is wrong with your environment — say so
before working around it.

---

## Get the data

`data/` is gitignored (~22 GB, some licence-restricted). You do **not** need all of
it. Pick by what you want to do:

| I want to… | I need | size | how |
|---|---|---|---|
| run the demo | HAD100 | 7.7 G | manual, see below |
| run the benchmark | + ABU, HYDICE, Indian Pines | ~50 M | `scripts/fetch_hydice.py`; others manual |
| retrain segmentation | + harmonized pool | 5.9 G | `scripts/build_background_pool.py` (generates it) |
| redo Level 2 (EnMAP) | EnMAP L2A | 8.4 G | needs a DLR account — `docs/enmap_handover.md` |
| redo Level 3 (Sentinel-2) | Sentinel-2 L2A | 5 M | `scripts/fetch_sentinel2.py`, needs CDSE keys |

**HAD100** — the one you most likely want — is a manual download from
[the project repo](https://github.com/ZhaoxuLi123/HAD100). Unpack it to
`data/benchmark/had100/`. Then **verify it**:

```bash
.venv/bin/python scripts/verify_had100.py       # exits non-zero on drift
.venv/bin/python scripts/verify_benchmarks.py
```

Run those. HAD100's own project page turned out to be wrong about its archive in
**five** separate ways (D11), and ABU and HYDICE in three more (D13). The verify
scripts exist because we got burned; they take seconds and they will tell you if your
download is not what you think it is.

`docs/datasets.md` records, for every dataset, whether each fact was **verified
against the files** or is **documentation-only**. Don't promote a row between tiers
without opening the files.

---

## Run the demo

This is the fastest way to see the whole system work end to end.

```bash
.venv/bin/python pipeline/demo.py --assert-offline
```

~15 seconds on a real HAD100/AVIRIS scene. Prints all eleven pipeline steps and writes
GeoJSON plus `demo_summary.json` to `experiments/demo/`.

Useful flags: `--scene <path.hdr>` · `--profile landcover` · `--target-recall 0.99` ·
`--out <dir>`

**Two steps deliberately don't do what you might assume, and say so on screen:**

- **Step 10 runs on a SYNTHETIC-PAIRS construction.** There is no real bi-temporal
  hyperspectral pair on disk (we checked all 20 EnMAP scenes; none overlap at
  different dates), so t2 is *built* from the real scene — known misregistration,
  implanted targets, illumination gain. Every figure it prints carries that label and
  must be quoted with it.
- **Step 11 prints SKIPPED.** The quantum arm is built and run, but deliberately isn't
  a dependency of the operational pipeline. Its results live in `docs/experiments.md`.

**`--assert-offline` genuinely enforces.** It replaces `socket.socket` for the whole
inference stage, so a connection attempt raises `OfflineViolation` rather than being
counted and reported afterwards. Claiming offline operation without proving it is the
weakest version of that claim.

## Run the pipeline on one scene

```bash
.venv/bin/python -m pipeline.run_pipeline \
    --scene data/benchmark/indian_pines/Indian_pines_corrected.mat \
    --source indian_pines --detector fused --out experiments/scratch
```

Detectors: `global_rx`, `local_rx`, `kernel_rx`, `crd`, `fused`. Swapping one is a
config edit, not a code change (§4.1). `streaming_rx` is deliberately **not** in the
registry — it takes a scene *path*, not a cube, because its whole point is never
materializing the cube; call it directly.

Per-dataset detector parameters go through `--detector-params` as JSON, never
hardcoded. HAD100's 64×64 patches want a smaller annulus than a 150×150 ABU scene, and
carrying one to the other measurably reranks the detectors (§3A.2).

Large scenes: use `--window row_off,col_off,height,width`. A full EnMAP scene is
~1.3 GB as one float32 copy and will get OOM-killed on a 16 GB machine (D32).

## Run the benchmark

```bash
.venv/bin/python scripts/run_benchmark.py --datasets abu,hydice,had100
```

Writes `experiments/rx_vs_ae/{report.md,results.csv,results_pooled.csv}`. Add
`--strict` to exit non-zero if any detector failed on any scene.

Every pooled figure is emitted **twice**: scene-macro (primary) and pixel-micro
(labelled). ABU's anomaly density spans a 32× range, so an unlabelled "pooled" number
is effectively a report on two scenes. This harness never produces one.

## Run the desktop UI

```bash
.venv/bin/python -m ui.app
```

Pick a scene, choose a detector, run, export `coordinates.xlsx`. It's a thin
front-end over `run_pipeline` — it calls the same code the CLI does, so its numbers
can't diverge from the benchmark's. Anomaly detection only in v1.

---

## Look through the failures yourself

This is the part worth your time. Start with [`docs/results.md`](docs/results.md),
then check any claim against its artifact:

```bash
# fusion loses to its own best component (D25)
cat experiments/rx_vs_ae/fusion_weights.json | head -40

# every quantum arm loses to classical RX (D28, D29)
column -s, -t experiments/quantum_results/results_pooled.csv | cut -c1-120

# the ROI cascade misses its target; quantization misses its target (D31)
cat docs/edge.md

# change detection is all synthetic pairs (D30)
cat experiments/change_arms/report.md
```

Each of those has a decision note in `plan.md` (search `### D25`, `### D28`, …) giving
the measurement, the cause, and what may and may not be claimed as a result.

**If you find something wrong with one of them, that's a contribution.** Several of
these findings came from someone re-checking a result that had already "passed."

---

## QGIS

Five projects in `qgis/projects/` open the outputs against an OpenStreetMap basemap:

| project | shows |
|---|---|
| `phase2_verify.qgz` | affine plumbing (Indian Pines — georeferencing is **synthetic**, D2) |
| `demo_verify.qgz` | real georeferencing, HAD100 |
| `phase5_level2_verify.qgz` | Level 2 accept check, EnMAP, **closed** |
| `phase5_level3_*.qgz` | Level 3 change intervals, Sentinel-2 over Jewar |

Rebuild them with **system `python3`**, not the venv — PyQGIS ships with the system
`qgis` package and cannot be installed into the venv (D26):

```bash
python3 scripts/build_qgis_project.py
```

---

## Credentials

Secrets live at `~/.config/sih/credentials.env` (mode 600), **outside this repo**.
`.env.example` lists variable *names* only.

```bash
.venv/bin/python scripts/check_credentials.py    # booleans only, never values
```

**Never print that file** — not in a terminal, a log, a commit, or a chat with an AI
assistant. An agent that prints it leaks it into a transcript that outlives the
session. See `CLAUDE.md` for the full rule.

You only need credentials for EnMAP (DLR) and Sentinel-2 (CDSE). Everything else runs
without them.

---

## The project's one rule

**Assume documentation is wrong until you have opened the file.**

HAD100's project page was wrong in five ways (D11). ABU and HYDICE in three more
(D13). DLR's docs described a portal dead five months (§8.0a). Our own `plan.md`
called EnMAP "blocked" while 3.6 GB of it sat on disk (D32). Sentinel-2's band count
was wrong in our spec in two directions at once (D34).

Every one of those was found by opening a file that a document claimed to describe.

## Contributing

See **`CONTRIBUTING.md`** — what's open, what's claimed, and the one file nobody edits
alone.
