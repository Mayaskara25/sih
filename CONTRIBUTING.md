# Contributing

## Push to a branch. Never to `main`.

```bash
git checkout -b <yourname>/<module>      # e.g. arun/kernel-rx
# work, commit
git push -u origin <yourname>/<module>
```

Open a PR. `main` stays green — 94 tests pass and both verify scripts exit 0 at
every commit on it. If your branch can't say the same, it isn't ready.

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

## What's open

The dependency DAG is `plan.md` §11. These branches are unblocked **now**
(`3A.harmonize` landed, so anything waiting on it is free):

| branch | modules | spec | notes |
|---|---|---|---|
| **3A detectors** | `local_rx`, `kernel_rx`, `crd`, `streaming_rx` | §3A | `global_rx` exists in `anomaly/rx.py` — follow its signature |
| **3C change detection** | `registration` → `spectral_angle` → `physics_fusion` | §3C | `change_detection/` is empty. `siamese_net` waits on 3B.synth — don't start it |
| **3D edge** | `profiling`, `constrained_sim` | §3D, §9 | `edge/` is empty. §11 marks these "start immediately". **Power is never reported** — no instrumented hardware |
| **3E quantum** | `qiskit_basics` → `feature_map` | §3E | `quantum/` is empty |

### Interfaces you must match

**Detectors** (`local_rx`, `kernel_rx`, `crd`, `streaming_rx`). Four people are
writing these in parallel, so the signature is not negotiable. Follow
`anomaly/rx.py::global_rx`:

```python
def <name>(cube: np.ndarray, *, <your params with defaults>) -> np.ndarray:
    """cube: [H, W, B] float32, NaN = nodata.
    returns:  [H, W] float32, NaN wherever the input pixel was NaN.
    """
```

Rules that apply to all four:
- **NaN in → NaN out, positionally.** Downstream scoring and masking rely on it.
- **Never form an explicit inverse.** `global_rx` uses `cho_factor`/`cho_solve`;
  covariance matrices here are ill-conditioned and `np.linalg.inv` will bite you.
- **Keyword-only tuning params, all with defaults**, so the eventual Phase 4
  registry can call any detector uniformly.
- `streaming_rx` additionally must hold float64 accumulators (Welford/Chan) — see
  §3A. Single-pass float32 covariance loses precision on 2 088 patches.

Higher raw scores mean more anomalous. Normalization is **not** your job — §3A's
scoring layer owns it (D3).

**Claimed / do not take:** `3B` (synth → datasets → train_unet) is the critical path
and actively being worked. So is anything under `preprocessing/`.

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
