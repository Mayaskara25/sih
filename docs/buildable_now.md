# What is buildable right now

**Status date: 2026-08-22.** (Superseded §4 below; §1-3 still hold.) Written because EnMAP L2A download access is blocked (PLAN.md O11)
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

## 4. Build order — STATUS as of 2026-08-22

**P0 is complete. A working prototype exists and runs end to end today.**
19 commits, 295 tests, 26 modules. `pipeline/demo.py` executes the full §10
sequence on a real HAD100/AVIRIS scene with real georeferencing.

| stage | state |
|---|---|
| Phase 1–2 walking skeleton | done (D14). QGIS eyeball still open (O4) — no GUI in this environment. |
| 3A.1 `harmonize` | done (D15). 425/224 native bands → 184 canonical. |
| 3B background pool | done (D17). `[2088, 64, 64, 184]`, 6.29 GB, memmap-built after an OOM kill. |
| 3B synth → train → infer | done (D19). **`unet_pretext` trained**: 40 epochs, best val 0.1243, 5.1 h on the GTX 1650. |
| 3A detectors | done. `global_rx` · `local_rx` · `kernel_rx` · `crd` · `streaming_rx` · `fused`. |
| Phase 4 | done. Registry (§4.1) + recall calibration (§4.2) + `roi_fusion` (§4.3). |
| Phase 5 Level 1 | done. Benchmark + `cascade_recall_audit`. |
| Phase 7 demo | done. Runs on HAD100, not EnMAP (O11). Steps 10–11 skip with a stated reason. |
| 3C change detection · 3D edge · 3E quantum | **not started — deliberately.** P1/P2 in §11.1. |

## 5. Six defects that execution found and review did not

All six were invisible to a test suite that was green throughout, and all six
came from **integration** — running two finished components against each other,
or one component against real data at scale. This is the single most useful
thing to know before adding to this repo.

| # | defect | why the tests missed it |
|---|---|---|
| D22 | `global_rx` raised `LinAlgError` on **3 of 13** ABU scenes | fixtures are unit-scale synthetic, where an absolute `reg=1e-6` is sensible — **the fixture's scale hid the bug the fixture existed to find** |
| D22.2 | `kernel_rx` ranked a **50σ outlier identically to a 2σ one** (481st of 900, both) | AUC alone looked plausible; only a magnitude-response check exposed it |
| D23 | `rois_to_geojson([])` crashed | zero ROIs is a *normal* outcome; nothing tested the benign case |
| D24 | `global_rx` accumulated covariance in **float32** | §3A.5 wrote that warning into the *streaming* spec and never applied it to the reference the streaming module is validated against |
| D25 | fusion **fails** §3A.9's accept criterion even after the prescribed sweep | needed a held-out split and an oracle-vs-fixed distinction to see |
| — | harness applied HAD100's 64×64 window to 150×150 ABU scenes | ranked `local_rx` **last of five**; §3A.2 warned about exactly this |

Two habits came out of this and are worth keeping:

1. **When a tolerance fails, the reference is a suspect too.** D24's natural
   fix was to loosen `rtol` until green. Measuring *which side* was wrong
   found a real bias in the more-trusted implementation.
2. **A regularizer built from the data is scale-safe; one added as a bare
   `reg * I` is not.** That single sentence predicts D22, D22.2 and clears
   `crd` (D22.3) without running a sweep.

## 6. What is still owed

**Nothing blocks a working prototype.** Remaining items are reporting and reach:

- **§3A.9 fusion weights** — the sweep ran (D25) and the criterion fails. Do
  not report fusion as beating its components; `crd` (0.9674 ABU macro) is the
  defensible headline.
- **D21 — the cheapest unblock in the project.** Two browser downloads, no
  account, restore `SPECTRA_POOLS["lib"]` and with it `unet_implanted_lib`,
  turning §3B.8's headline comparison from one arm into two. Run
  `scripts/fetch_speclib.py --check` for the current instructions.
- **O4 QGIS eyeball** — needs a human with a GUI.
- **O9 wavelength recovery** — would un-suspend three §3B.8 arms and the
  `index` fusion component on ABU/HYDICE at once.
