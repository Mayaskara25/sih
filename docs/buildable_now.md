# What is buildable right now

**Status date: 2026-08-21.** Written because EnMAP L2A download access is blocked (PLAN.md O11)
and the natural question is how much of the project that stops. The answer is: **one sub-phase.**

This document is a snapshot. `plan.md` remains authoritative; where they disagree, plan.md wins.
Re-check O11 before trusting the "blocked" column.

---

## 1. On disk, verified against the files

| dataset | size | what was verified | verifier |
|---|---|---|---|
| **HAD100** | 7.7 G | 616 ENVI scenes; real UTM/WGS-84 `map info` on every one (D11.5); **2 088 usable 64×64 patches** (D11.2); non-monotonic wavelength arrays (D11.4) | `scripts/verify_had100.py` |
| **ABU** | 39 M | 13 scenes, 7 distinct band counts, 3 dtypes; 8 int16 scenes contain genuine negatives (D13.2) | `scripts/verify_benchmarks.py` |
| **HYDICE Urban** | 2.5 M | 80×100×**175**, 21 anomaly px, pinned by SHA256 | `scripts/fetch_hydice.py` |
| **Indian Pines** | 5.7 M | no CRS, no affine, **no wavelength array** (D13.1) | `scripts/verify_benchmarks.py` |

Credentials configured (`scripts/check_credentials.py` → exit 0): CDSE S3 keys, DLR username/password.

---

## 2. Two facts that make this stronger than it looks

**HAD100 *is* the background pool.** §2.2 states it directly: HAD100 is the **primary** pool and
EnMAP L2A is *"the extension for sensor diversity, not the starting point."* 3B's longest pole was
always the HAD100 download, and that is finished.

**HAD100 already contains AVIRIS.** D11.3 enumerates `aviris_ng` (425 bands) and AVIRIS-Classic
inside it. The plan's background pool is "AVIRIS-NG and EnMAP L2A" — the AVIRIS half is on disk.

---

## 3. What EnMAP actually blocks

**Phase 5 Level 2. Nothing else.**

Not Phase 1, 2, 3A, 3B, 3C, 3D, 3E, 4, Phase 5 L1, Phase 5 L3, or Phase 6 Tier A.

Level 2's headline job has already moved elsewhere: it was *"real georeferencing verified for the
first time"*, but D11.5 found genuine UTM/WGS-84 headers on all 616 HAD100 scenes, so D2 was
amended to do that check in **3A**. Phase 7's demo step 1 reads "EnMAP/**AVIRIS**", which HAD100
satisfies.

What is genuinely lost while O11 stands: **sensor diversity** in the background pool, and a second
independent georeferencing check. Both are enhancements to a working system, not foundations.

---

## 4. Build order

Critical path (§11): `contracts → Phase 2 → 3A.harmonize → 3B.synth → 3B training → Phase 4 → Phase 5`.
Every step below is available today.

1. **Phase 1–2 — walking skeleton**, on Indian Pines. **DONE, except QGIS (2026-08-21, D14.)**
   loader → normalize → global_rx → scoring → postfilter → polygonize → projections → geojson →
   run_pipeline: built, 54 tests pass, and the pipeline runs end to end on
   `Indian_pines_corrected.mat` (3 ROIs, `validate_geojson` green). `.venv` is bootstrapped
   against the full §1.2 lock and `.tooling/venv` is gone, per §1.1.
   Indian Pines has **no** CRS and **no** wavelengths (D13.1), so the synthetic-affine path of D2
   applies and Phase 2 verifies transform *plumbing* only — never real-world accuracy.
   **QGIS verify is the one piece not done** — no GUI in this environment, and `which qgis` still
   finds nothing (O4). A programmatic substitute (polygon bounds vs. `meta.transform`-derived
   pixel bounds, run against the real pipeline output) passed on all 3 ROIs, but that is not the
   written exit criterion — treat Phase 2 exit as unsigned until O4 clears.
   **D14 also found and fixed two spec bugs while building this:** §2.2's water-band indices
   target the wrong band count for the shipped Indian Pines cube (D14.1), and a hand-rolled ENVI
   `map info` parse silently dropped real rotation present on every HAD100 header checked —
   confirmed wrong against GDAL, now fixed by delegating to it (D14.2). **D14.2 matters for the
   next step**, since 3A.1 is where real HAD100 georeferencing is first claimed accurate.

2. **3A.harmonize**, on HAD100 **raw ENVI**. **DONE (2026-08-21, D15.)**
   `preprocessing/harmonize.py` built: `CANONICAL_WL`/`WATER_WINDOWS`/`RETAINED_BANDS`/
   `water_mask`/`sort_spectral_axis`/`coverage_ok`/`harmonize`. 16 new tests pass. Real NG (425)
   and Classic (224) raw scenes both harmonize to `shape[-1] == 184` and stack — the D11.3 join.
   `coverage_ok` reproduces the plan's measured figures exactly: **0/184** uncovered for both raw
   sensors, **43/184** uncovered for a reconstructed `band_select`-style axis, which `harmonize`
   then refuses to interpolate across (self-defending, per D11.6).
   Not `main.py`'s `band_select` output — D11.6: that leaves interior holes up to 276 nm and
   43/184 canonical bands uncovered. Raw ENVI has 0 gaps.
   Sort the spectral axis first: AVIRIS wavelength arrays are non-monotonic (D11.4) and `np.interp`
   neither requires nor checks ascending `xp`.
   Target grid: 211 canonical → 27 dropped by the inclusive water mask → **184 retained** (D9).
   **D15 also found and fixed a real numerical bug**: the first interpolation implementation used
   one weight-matrix matmul, which is mathematically right on clean input but wrong under NaN —
   `0.0 * NaN == NaN`, so a single bad band poisoned all 184 outputs for that pixel instead of
   just the ≤2 that reference it. Fixed with gather-based interpolation; regression-tested.
   **Also closed §3A.1's first-real-georeference check** (D2, D11.5): independently hand-derived
   the rotation-aware affine from raw `map info` text and confirmed it matches
   `raster_loader`'s GDAL-delegated transform on both a 33° and a 90° real header, then confirmed
   the full pixel→world→EPSG:4326 path lands within one pixel on both.
   **Scope amendment (D15):** `reduce_bands` (PCA/kPCA → `C=30`) stays deferred to §3B.3, now for
   the correct reason — it must fit on the train split only, not merely after pool assembly, or
   it leaks scoring-scene statistics into the representation. §3A's accept criterion no longer
   claims `C=30`; that criterion moved to §3B.3.

3. **3B — synth → datasets → train_unet**, on the 2 088 patches.
   Enforce scene-level, spectrum-level and crop-level leakage control (§3B).

4. **In parallel, no dependencies** (§11 "start immediately"):
   `3D.profiling`, `3D.constrained_sim`, `3E.qiskit_basics`.

5. **Phase 5 Level 1** — benchmark on HAD100 + ABU + HYDICE. These are the headline numbers.
   Pooling is **scene-macro-average** (primary); pixel-micro-average is secondary and must be
   labelled as such (§3B.8).

6. **Phase 6 Tier A** — simulated edge benchmarks. Power is **not** reported (§9).

---

## 5. Before Phase 5, prove these

§8.0 forbids writing Phase 5 code against an assumed spec.

- **Sentinel-2 retrieval is unproven.** CDSE S3 keys are *configured*, not *exercised*. Prove one
  retrieval before Level 3 code. Catalogue search needs no credentials at all (§8.0a).
- **EnMAP band/wavelength facts are unverified** and stay documentation-only in §15 regardless of
  whether O11 clears — catalogue metadata is not verification.
- Every fetch must validate **content, not status** — `core.http_guard.assert_magic`. DLR returns
  HTTP 200 with an HTML login page; this has now bitten the project three times (§8.0a).

---

## 6. If someone else downloads EnMAP for you

Nothing above waits on it, so this is not urgent.

1. **Check the EnMAP licence** on `geoservice.dlr.de/web/datasets/enmap` first. D12 rule 4 already
   forbids moving benchmark data around without checking terms.
2. Files arriving by a route the repo cannot reproduce must still be **verified and manifested**
   (SHA256 + size), on the model of `scripts/fetch_hydice.py`.
3. Still run the **DESIS** download test yourself — DESIS is subscribed on the same account, so it
   separates "EnMAP-specific entitlement bug" from "account-wide propagation failure". That
   distinction is what `eoc-ums-helpdesk@dlr.de` needs.
