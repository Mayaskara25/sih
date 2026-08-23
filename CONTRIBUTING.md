# Contributing

## Push to a branch. Never to `main`.

```bash
git checkout -b <yourname>/<module>      # e.g. arun/kernel-rx
# work, commit
git push -u origin <yourname>/<module>
```

Open a PR. `main` stays green — the full pytest suite passes (400+ tests;
check with `.venv/bin/python -m pytest --collect-only -q | tail -1`) and both
verify scripts exit 0 at every commit on it. If your branch can't say the
same, it isn't ready.

---

## The one rule that keeps us from breaking each other

**`core/contracts.py` is the shared boundary. Do not edit it alone.**

It defines `SceneMeta`, `validate_scene()`, `ROIRecord`, `validate_roi()`,
`validate_score_raster()`, `validate_geojson()` — the interfaces every module hands
data across. They are enforced at **runtime**, so a wrong-shaped tensor raises
`ContractViolation` at the boundary instead of silently flowing three layers
downstream and producing plausible nonsense.

That is why the branches below can be written in parallel by people who never talk
to each other. It only holds while the contracts hold. If your module genuinely
needs a contract change, **open an issue first** — a change there touches everyone.

Everything else in the table below is a leaf module plugging into contracts that
already exist.

---

## What's landed, what's open

The dependency DAG is `plan.md` §11. Current state of the branches:

| branch | modules | spec | status |
|---|---|---|---|
| **3A harmonize + detectors** | `harmonize`, `local_rx`, `kernel_rx`, `crd`, `streaming_rx` | §3A | **landed** |
| **3B segmentation** | synth → datasets → train_unet → infer → postfilter | §3B | **landed** (one trainable arm; 3 suspended on O9/D21 — they raise informative errors by design) |
| **3C change detection** | registration → spectral_angle → temporal_difference → physics_fusion → cloud_mask → temporal_baseline → siamese_net | §3C | **landed** (`experiments/change_arms/report.md`; siamese underfits at its modest budget — scaling it is open) |
| **3D edge** | profiling · constrained_sim · streaming · quantization · onnx_inference · roi_pipeline · benchmark | §3D, §9 | **landed** (SIMULATED numbers only — §0.2: no hardware) |
| **3E quantum** | qiskit_basics → feature_map → comparison | §3E | **built** (plan.md D27; results owned by the quantum branch) |

Genuinely open right now:

- **Phase 5 §8.0 — partly done.** EnMAP L2A and Sentinel-2 are now **verified**
  (`scripts/verify_enmap.py`, `scripts/verify_sentinel2.py`, both exit non-zero on
  drift). Still documentation-only: **AVIRIS flightlines** and **USGS splib07**.
  The plan's rule stands: assume documentation is wrong until you open the file.
  Both verifications found the spec wrong — EnMAP in D32, Sentinel-2 twice in D34.
- **`fetch_enmap.py` — DONE 2026-08-23 (D36).** The DLR download leg works and is now
  reproducible: `scripts/fetch_enmap.py` searches, authenticates, downloads bounded by
  `--limit` with a size projection and disk check, verifies magic bytes, and writes
  `docs/enmap_fetch_manifest.json` with sha256 per asset. `--reconcile` retroactively
  indexed the 20 pre-existing scenes, so they are no longer unreproducible artifacts.
- **Fetchers — `fetch_sentinel2.py` exists** and works (CDSE), and is the model to
  follow: credentials via `core.credentials`, re-runnable, manifest-merging, bounded.
- **`scripts/fetch_speclib.py::ingest()`** is a deliberate stub (D21): a human
  must browser-download the archive first; the parser gets written against the
  real files. This is the cheapest unblock in the project.
- **Siamese budget scaling** (§3C.8 finding): the learned change arm underfits
  at 15 epochs / 48 crops — see `experiments/change_arms/report.md`.

Claim a branch by opening an issue titled after the module before you start.

---

## Before you open a PR

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_had100.py        # skip if you have no data/
.venv/bin/python scripts/verify_benchmarks.py
```

- **Tests for the failure, not just the success.** A test that proves the mechanism
  works is worth less than one that proves the mistake it prevents actually happens.
  See `tests/test_data_hygiene.py`: it doesn't just check `scene_groups()` returns
  labels, it demonstrates a naive patch split *really does* leak scenes.
- **Guard on content, not status codes.** Any fetch must go through
  `core.http_guard.assert_magic`. DLR returns HTTP 200 with an HTML login page; this
  has bitten the project three times.
- **`skipif` on anything needing `data/`.** A fresh clone must stay green.

## Facts vs. claims

If you assert something about a dataset — band count, dtype, wavelengths, no-data —
**you must have opened the file.** Numbers from a project page are documentation-only
and go in the lower tier of `docs/datasets.md`.

This is not pedantry. HAD100's page was wrong in five ways, ABU and HYDICE in three
more, and DLR documented a portal that had been dead for five months. Every one was
caught by opening files.

## Machine limits worth knowing

- **RAM is the binding constraint: ~13 GB, ~8.3 GB free, no swap.** The background
  pool is 6.29 GB — open it with `mmap_mode="r"`, never `np.load` it whole. An
  earlier build was OOM-killed with exit 137 and no traceback (D17).
- Target deployment is 4 GB. Model sizing in §2.1 assumes it.

## Never commit

Secrets (`~/.config/sih/credentials.env` lives outside the repo), anything under
`data/`, or `.venv/`. All are gitignored — keep it that way, and never `git add -f`
past it.
