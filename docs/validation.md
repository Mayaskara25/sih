# Phase 5 Level 3 site selection — O5

Open question O5 (PLAN.md): *"Pick during Phase 5 from actual data availability; record the
selection rationale here."* This is that record. Written by the agent that built
`scripts/fetch_sentinel2.py` / `scripts/verify_sentinel2.py`, before any change-detection code
ran against the data (that is the next agent's job, not this document's).

**Reporting constraint (PLAN.md §8 Level 3, Roadmap §1.9/§9.7), restated because it binds this
whole page:** report observed physical change only — no conflict attribution, no causal claim, no
inference about intent from image differences. The site below is a civil infrastructure project.
Nothing here or in the eventual case study should be read as, or drift toward, anything else.

---

## Site chosen: Noida International Airport (Jewar), Uttar Pradesh

- **AOI centre:** 77.61° E, 28.17° N (28°10′12″N 077°36′36″E)
- **Sentinel-2 MGRS tile:** T43RGM (confirmed: every product used below falls in this one tile)
- **AOI window:** 6 km × 6 km box centred on the coordinate above, in the tile's own UTM 43N
  (EPSG:32643) grid — 300×300 px at the 20 m grid `scripts/fetch_sentinel2.py` reads
- **The event:** construction of a new international airport on agricultural/village land,
  starting from a publicly dated groundbreaking through to public operation. This is an
  **infrastructure development**, not a disaster or a contested area.

### Public, dated sources for the event (documentation-only — not independently re-verified
against a primary government record by this agent; cited so the second agent can)

- Foundation stone laid 25 November 2021 (Gulf News: *"India PM Modi lays foundation stone for
  Noida International Airport at Jewar"*)
- Construction work reported starting around August 2021 (Hindustan Times, *"Construction work
  begins at Noida International Airport, Jewar"*, 27 Aug 2021)
- Land acquisition: State Government acquired ~1,336 ha for Phase I (YEIDA project page); further
  ~441 ha from 18 villages acquired for the airport corridor (subsequent reporting)
- Phase 1 inaugurated / opened to traffic in 2026 (multiple 2026 reports converge on a
  March–June 2026 opening window; exact commissioning date **not independently verified here** —
  the second agent should pin this down from a primary source, e.g. AAI/YEIDA press release, if
  the report needs to cite it precisely)

**This agent did not verify these facts against a primary government or airport-authority
document** — they come from news reporting found via web search, which is the documentation-only
evidence tier this project applies to everything until a file is actually opened (PLAN.md §15's
rule, applied here to the *event*, not a dataset). What **is** independently verified is the
satellite record itself (below) — real Sentinel-2 files opened, real dates, real reflectance
values, real cloud statistics.

### Why this site

1. **Genuine, low-cloud Sentinel-2 coverage spanning the whole timeline, on one tile.** A live
   OData search (`scripts/fetch_sentinel2.py:search_products`) found near-zero-cloud L2A products
   over this point in every era needed:

   | era | date used | tile `cloudCover` | AOI clear fraction (from SCL) |
   |---|---|---|---|
   | pre-construction | 2020-10-16 | 0.0% | 1.000 |
   | mid-construction | 2022-03-30 | 0.0% | 1.000 |
   | near-complete | 2024-10-30 | 0.0023% | 1.000 |
   | post-opening | 2026-06-17 | 0.0% | 1.000 |

   All four are on tile **T43RGM**, and `scripts/verify_sentinel2.py`'s cross-product check
   confirms the geotransform is **byte-identical across all four dates** (fixed MGRS pixel grid,
   verified — not assumed): no co-registration step is needed for this AOI/tile before running
   `TemporalBaseline`. This was the single biggest discriminator versus other candidates — many
   sites had at least one era with only cloudy passes available.

2. **Dry-season-friendly.** Construction milestones here are not seasonal, so low-cloud dates
   exist year-round, unlike a monsoon-timed event (see rejected: flood, below).

3. **Unambiguous, unpolitically-framed physical change.** Bare agricultural/village land being
   graded, a runway and terminal footprint appearing, and the site later showing built
   infrastructure is about as close to "observed physical change with no room for causal or
   intent inference" as a real Indian case study gets. There is no conflict, no border, no
   security dimension to accidentally imply.

4. **Not Kashmir, not a disputed border area** — required by this task's brief independent of (3).

### What this agent rejected, and why

- **Kashmir / any disputed border area** — excluded outright per instructions, not evaluated
  further.
- **A dated flood (e.g., a specific monsoon flood event)** — floods are monsoon-season by
  definition, and monsoon is exactly when Sentinel-2 optical passes are most likely to be
  clouded out over the flood itself; getting a clear *during-flood* optical pass is unreliable
  (this is why flood mapping literature leans on Sentinel-1 SAR, not S2, for the event date
  itself). Rejected for the same reason O5/advisor guidance flagged: "a dated flood may have no
  clear pre/post pair." Not pursued further than this reasoning — no specific flood site was
  searched.
- **A reservoir drawdown/refill site** — a real, dry-season-friendly alternative that was
  considered but not pursued once the airport site's coverage came back clean on the first search;
  no specific reservoir was searched or rejected on data grounds, only de-prioritized once a
  working candidate was found, so it remains a reasonable fallback if the airport site turns out
  to have some other problem the second agent discovers.
- **A mining/quarry expansion site** — same status as the reservoir option: a plausible,
  dry-season-friendly category per the task brief, not pursued once the airport candidate's
  coverage and event-dating both came back strong on the first attempt.

No other specific candidate location was searched or rejected — the airport site was found,
checked, and accepted on the first attempt, so there is no longer list of rejected *locations* to
report, only rejected *categories* (above) and the reasoning for skipping them.

---

## What was fetched (see `data/raw/sentinel2/jewar_airport/manifest.json`)

4 products, 6 landcover-profile bands (B02/B03/B04/B8A/B11/B12) + SCL each, cropped to the AOI
window described above. **Total local bytes written: 4,701,576 B (~4.5 MB)** — see
`scripts/fetch_sentinel2.py`'s module docstring for the measured network-vs-local relationship
(a single-band windowed read transferred ~512 KB of actual network bytes for a 300×300 px window,
via 3 HTTP range requests against a 33.7 MB source JP2 — JP2 tile granularity, not something this
fetcher controls further). This is far under the "well under 5 GB" budget.

| era | product | sensing time (from `MTD_TL.xml`, per-tile) |
|---|---|---|
| pre_construction | `S2B_MSIL2A_20201016T052819_N0500_R105_T43RGM_20230413T162324.SAFE` | 2020-10-16T05:41:07.277248Z |
| mid_construction | `S2B_MSIL2A_20220330T052639_N0510_R105_T43RGM_20240522T115955.SAFE` | 2022-03-30T05:41:00.266099Z |
| near_complete | `S2A_MSIL2A_20241030T052941_N0511_R105_T43RGM_20241030T093050.SAFE` | 2024-10-30T05:41:04.025492Z |
| post_opening | `S2B_MSIL2A_20260617T052649_N0512_R105_T43RGM_20260617T104156.SAFE` | 2026-06-17T05:41:01.854886Z |

Individual per-era manifests (`manifest_2022.json`, `manifest_2024.json`, `manifest_2026.json`)
sit alongside the merged `manifest.json` as provenance for how each date was selected (separate
narrow-date-range `fetch_sentinel2.py` calls, one per era, each picking that era's
lowest-tile-cloud date — not a single wide-range call's chronological first four, which would
have clustered inside one era and shown no construction-timeline signal at all).

**Note for the second agent:** the `near_complete` product (2024-10-30) predates the reported
2026 opening, so the actual "before / during-construction / after-opening" bracket is really
`pre_construction` → `mid_construction` → `post_opening`, with `near_complete` as a fourth,
useful mid-to-late-construction data point rather than a true "after opening" one. Re-check the
airport's actual commissioning date against a primary source before writing any date-anchored
claim in the final report.

## Data quality facts relevant to the case study (see `docs/sentinel2_verified.json` for the full,
machine-written record)

- All 6 landcover bands are natively available pre-resampled to **20 m** by ESA's own L2A
  processor for 3 of them (B02/B03/B04, natively 10 m) and natively 20 m for the other 3
  (B8A/B11/B12) — **no on-the-fly resampling performed by this project's code**, verified against
  a real `MTD_MSIL2A.xml`'s `<Spectral_Information>/<RESOLUTION>` elements. NIR is **B8A** (864 nm
  nominal), not B08 — B08 has no 20 m copy in this product line.
- Wavelength centres are **not** the same across S2A/B/C: e.g. one product's B12 centre was
  492.7/559.8/664.6/864.7/1613.7/2202.4 nm (S2A) vs 492.3/559.0/665.0/864.0/1610.4/2185.7 nm (S2B),
  measured directly, not assumed — `scripts/verify_sentinel2.py` reports
  `wavelengths_uniform: False` across this exact 4-date set for that reason. This is expected and
  not a data-quality problem; `meta.wavelengths` is read per-product, never hardcoded.
- `BOA_ADD_OFFSET` is **-1000, uniform across all 13 metadata-defined bands**, on every product
  checked; `BOA_QUANTIFICATION_VALUE` is **10000**, also uniform across all four dates, despite
  the four products spanning **four different processing baselines** (05.00, 05.10, 05.11, 05.12
  — one per date, see `docs/sentinel2_verified.json`). Baseline 04.00 is where CDSE/ESA
  introduced `BOA_ADD_OFFSET`; all four products here are well past that, so its presence was
  expected, and it was still read from each file rather than assumed. **This
  project's fetcher/loader records these values on every file but does NOT apply them** — the
  cube stays as raw digital numbers (uint16), matching the EnMAP convention (D32: recorded, not
  auto-rescaled). **The second agent must apply `(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE`
  before computing ndvi/ndwi/nbr/bsi**, or every index will be wrong by a large, silent, constant
  offset. **Warning for that step:** with offset −1000 and quantification 10000, any raw DN below
  1000 becomes a *negative* reflectance — this is legitimate (SWIR over water routinely does this)
  rather than an error, and this AOI does contain some SCL class-6 (water) pixels. If the second
  agent's index code or `preprocessing/normalize.standardize` clips negatives or treats them as
  invalid, water pixels will silently break — check that path deliberately rather than assuming.
- `AOI_CLEAR_FRACTION_SCL` is 1.000 for all four dates (recomputed independently by
  `verify_sentinel2.py`, not just trusted from the fetcher's own tag) — the AOI itself is fully
  clear on every date used, not merely "tile average under threshold." Every band's
  `fill_fraction_per_band` (DN==0) is also 0.0 on all four dates (`docs/sentinel2_verified.json`)
  — the D32 fill-poisons-every-band-via-NaN hazard does not apply to this AOI/date set, checked
  rather than assumed.
- **A known trap that turned out NOT to apply here, worth stating so the second agent doesn't
  re-derive it:** new concrete (runway/apron) is a classic SCL cloud (class 8/9) false positive,
  which would zero out exactly the changed area in `cloud_mask`/`fuse_change_signals`. It did not
  happen on any of these four products — `AOI_CLEAR_FRACTION_SCL == 1.000` on the post-opening
  date means zero pixels landed in SCL classes {3,8,9,10} there.
- **Visual confirmation the AOI actually frames the airport** (not just a statistical check): a
  quick true-colour (B04/B03/B02) stretch of each date's stack, done as part of this site-selection
  check and not committed to the repo, shows an unambiguous progression — 2020-10-16 is ordinary
  farmland/villages with no runway visible; 2022-03-30 shows a clearly demarcated boundary with
  land inside it already cleared; 2024-10-30 shows a runway/taxiway footprint mid-construction
  (pale paving visible); 2026-06-17 shows a complete, operating airport (runway, taxiways, terminal
  buildings) squarely centred in the 6×6 km window. The SCL vegetation-class fraction also moves
  across the same four dates (24.3% → 8.6% → 4.8% → 14.5%) but is confounded by season (October vs
  March vs June) and should not be read as a change measurement on its own — the quicklook is the
  evidence that matters here, not this fraction.

---

## Level 3 case study — results (second agent, run end to end 2026-08-23)

Runner: `scripts/run_level3_case_study.py`. Outputs: `experiments/phase5_level3/`. QGIS:
`qgis/projects/phase5_level3_20201016_to_20220330.qgz` and
`phase5_level3_20220330_to_20260617.qgz`, built via `scripts/build_qgis_project.py` (system
python3, PyQGIS). Every number below is read from `experiments/phase5_level3/run_manifest.json`,
the per-date `*_stats.json` files, or a written GeoTIFF/GeoJSON opened directly — none are
estimates.

**Side effect of running the QGIS builder, noted so it isn't mistaken for an unrelated change:**
`build_qgis_project.py`'s `main()` rebuilds every project in one pass, so running it to add the two
Level 3 projects also regenerated `phase2_verify.qgz`, `demo_verify.qgz` and
`phase5_level2_verify.qgz` (all three now show as modified in `git status`). Their substantive
content is unchanged — the build log reports the same CRS and ROI counts as before this run
(`phase2_verify`: EPSG:32616, 3 ROIs; `demo_verify`: EPSG:32611, 3 ROIs; `phase5_level2_verify`:
EPSG:32643, 33 ROIs, matching `experiments/phase5_level2/level2_metrics.json`'s `n_rois: 33`) — so
the D26/D32 human sign-offs recorded against those files still stand; only the on-disk `.qgz`
bytes changed (a build-time artifact, e.g. an embedded regeneration timestamp), not what they show.

**Reporting constraint restated:** everything below is an observed physical measurement — dates,
areas, index values, pixel counts. No cause is asserted for any of it.

### Pipeline actually run

`fetch [pre-existing] → cloud mask (SCL) → co-register (verification only) → TemporalBaseline →
SAM + physics fusion → postfilter (landcover profile) → ROIs → GeoJSON → QGIS`, exactly PLAN.md
§8's line, using only `preprocessing/cloud_mask.py`, `preprocessing/registration.py`,
`change_detection/{spectral_angle,temporal_difference,physics_fusion,temporal_baseline}.py`,
`segmentation/postfilter.py`, `geospatial/{polygonize,projections,geojson}.py`. No detection
science was reimplemented; the only new formula-level code is a direct reuse of
`anomaly.scoring._INDEX_DEFINITIONS` for descriptive NDVI/NDWI/NBR/BSI rasters (see below for why
`spectral_index_score()` itself could not be called here).

### DN → reflectance (task rule 1)

`BOA_ADD_OFFSET` (−1000.0) and `BOA_QUANTIFICATION_VALUE` (10000.0) were read **per product** from
`docs/sentinel2_verified.json`, not hardcoded — confirmed identical across all four processing
baselines (05.00/05.10/05.11/05.12) but read independently for each. Reflectance was **not
clipped**: the 2024-10-30 secondary (S2A) product has 190 of 90,000 B02 (blue) pixels with negative
reflectance after rescaling (min = −0.0086, from
`experiments/phase5_level3/dates/2024-10-30_S2A_SECONDARY_stats.json`'s
`reflectance_range_per_band.B02`); all three primary S2B dates have zero negative pixels in every
band. This is the concrete evidence the offset was applied and not silently skipped or
clipped.

### Cloud masking

All four products' SCL carries only classes {4,5,6,7} (vegetation/bare/water/unclassified — no
cloud/shadow/cirrus classes at all in this AOI), matching `docs/sentinel2_verified.json`'s
`aoi_clear_fraction_scl_tag: 1.0` for every date. `preprocessing/cloud_mask.cloud_shadow_mask(cube,
meta, scl=scl)` was called for every date (required — `source=="sentinel2"` raises if `scl` is
omitted); the resulting mask is all-zero on every date, and the per-interval cloud union used by
`fuse_change_signals` is 0.0 for both primary intervals. **Limitation of this specific run, stated
plainly:** because this AOI has zero cloud pixels on all four dates, this run cannot demonstrate
the SCL mask actually zeroing a genuinely cloudy region — only that it runs correctly and returns
the expected all-clear answer on clear data. `cloud_mask.py`'s hyperspectral spectral-threshold
fallback path was not exercised and does not apply to Sentinel-2 by the module's own design (it
raises rather than falling back when `source=="sentinel2"`).

### Co-registration — verification, not correction

The geotransform was independently re-confirmed byte-identical across all four products
(`(20.0, 0.0, 753260.0, 0.0, -20.0, 3121800.0)`, EPSG:32643, re-derived here directly from
`load_scene`, not just trusted from `docs/sentinel2_verified.json`). `coregister_subpixel` was
still run on every relevant pair, as the verification the task asked for, and its **measured**
residual (phase-correlation shift between the two dates' band-averaged proxies, re-estimated after
the ECC refinement stage) was:

| pair | rmse_px |
|---|---|
| 2020-10-16 → 2022-03-30 | 0.180 |
| 2022-03-30 → 2026-06-17 | 0.112 |
| 2022-03-30 → 2024-10-30 (secondary) | 0.400 |
| 2024-10-30 → 2026-06-17 (secondary) | 0.206 |

All four are well under the module's 1.0 px `RegistrationFailure` limit. These are **not** exactly
zero, and that is expected and correctly interpreted: phase correlation measures apparent shift in
the scene *content* (which genuinely changed between dates — that is the whole point of the study),
not just geometric offset, so a small non-zero residual on a provably identical grid is the
expected signature of real content change, not misalignment. No warp was applied to the analysis
cubes — the verification confirmed none was needed, per the task's instruction, and the original,
un-warped reflectance cubes were used for every downstream step.

### Primary series (S2B-only): per-date index summary

| date | sensor | mean NDVI | median NDVI | mean BSI |
|---|---|---|---|---|
| 2020-10-16 | S2B | 0.446 | 0.421 | 0.014 |
| 2022-03-30 | S2B | 0.364 | 0.362 | 0.053 |
| 2026-06-17 | S2B | 0.299 | 0.240 | 0.061 |

Mean NDVI falls monotonically and mean BSI (bare-soil index) rises monotonically across the three
primary dates — consistent with a vegetated → bare/built progression. **This is confounded with
season and cannot be separated from it by this single time series**: 2020-10-16 is post-monsoon
(historically the greenest pass of the year for this cropland), 2022-03-30 is dry-season, and
2026-06-17 is pre-monsoon. A genuinely seasonal (non-construction) NDVI cycle at this site would
produce the same qualitative pattern. This is stated as the confound it is, not resolved by this
dataset — four single-year snapshots cannot distinguish a one-off land-use conversion from a
seasonal cycle on their own; the corroborating evidence is the dated public groundbreaking record
(documentation-tier, `docs/validation.md` above) and the visual quicklook, not the NDVI number
alone.

### Primary interval change products (accept-criteria numbers)

Threshold: 95th percentile of the rank-normalized fused SAM + physics-fusion score (default
component weights: sam 0.50, variance 0.20, entropy 0.15, coherence 0.15), chosen before running —
see the runner's module docstring for why 95.0 rather than Level 2's 99.0 (the `landcover` profile
has no `max_area_px` cap and a much lower `min_solidity`, i.e. it is built for large regions, and a
top-1% threshold works against that). Postfilter: `landcover` profile thresholds from
`configs/target_profile.yaml` (`min_area_px: 50, max_area_px: null, min_solidity: 0.05,
max_elongation: 20.0`).

| interval | ROIs (raw → kept) | area of kept ROIs (top-5% score selection) | mean SAM (rad) | mean \|Δ magnitude\| |
|---|---|---|---|---|
| 2020-10-16 → 2022-03-30 | 127 → 9 | 44.16 ha | 0.141 | 0.102 |
| 2022-03-30 → 2026-06-17 | 100 → 10 | 51.60 ha | 0.176 | 0.216 |

**These area numbers are NOT a measurement of "how much changed" and must not be read that way.**
`threshold_by_percentile(pct=95.0)` selects the top 5% of pixels by rank **on every input, by
construction** — 4,500 of the AOI's 90,000 px, i.e. 180 ha, are selected before morphology and the
postfilter run, regardless of whether the scene changed at all. Running the identical procedure on
two dates of genuinely unchanged farmland would still select ~180 ha and, after the same
`min_area_px≥50` gate, would still leave some tens of hectares of ROIs — percentile thresholding is
a **relative, within-scene selection of the highest-ranked pixels**, not an absolute measurement of
changed extent. 44.16 ha and 51.60 ha are what survived morphological cleanup and the postfilter
out of that guaranteed 180 ha selection, in each interval; **the difference between them (51.60 vs
44.16 ha) is not interpretable as "more change happened in the second interval"** — both start from
the same fixed 5% quota. What this method DOES support: identifying and localising the
highest-ranked-within-scene regions for the human check below; it does not by itself support any
area or magnitude comparison across intervals or across sites.

**All 208 dropped ROIs across both intervals failed on exactly one gate: `area_below_min` (<50
px).** Not one was dropped for low solidity or high elongation, despite the landcover profile
being deliberately permissive on both — the postfilter's only active role in this run was removing
morphological speckle, not shape filtering. The kept-ROI centroids cluster at 28.16–28.19° N,
77.59–77.63° E, i.e. inside the stated AOI (centred 28.17° N, 77.61° E) — a geometric/location
sanity check, not a claim about what is there.

**`meta.acquired` verification (D33), asserted programmatically, not assumed:** every feature in
both interval GeoJSONs carries `timestamp` equal to the resolved t2 acquisition time (e.g.
`2022-03-30T05:41:00Z` for the first interval, `2026-06-17T05:41:01Z` for the second) — confirmed
by reading the written GeoJSON files directly (not the run date, 2026-08-23). The runner raises if
any feature's timestamp disagrees; it did not raise. **Convention used, stated because C6 has no
T1/T2 timestamp field (frozen at 16 properties):** a change-interval feature's `timestamp` is the
**end** of the interval (`meta_t2.acquired`); the interval start is recorded in the GeoJSON
filename, the `run_manifest.json` interval record (`t1_date`/`t1_scene`), and the raster tags
(`T1_SCENE_ID`, `T2_SCENE_ID`, `REG_RMSE_PX`, added on top of `save_score_raster`'s C2 tags since
C2 does not define the C4 pair-fields either).

### Fused score vs. NDVI magnitude — measured, but does not resolve the season-vs-construction question

Spearman rank correlation between the fused change score and `|NDVI(t2) − NDVI(t1)|`, computed
over all 90,000 pixels of each interval:

| interval | ρ(fused, \|ΔNDVI\|) | ρ(fused, SAM) | ρ(fused, magnitude_difference) | ρ(SAM, magnitude_difference) |
|---|---|---|---|---|
| 2020-10-16 → 2022-03-30 | 0.684 | 0.789 | 0.666 | 0.624 |
| 2022-03-30 → 2026-06-17 | 0.735 | 0.798 | 0.538 | 0.604 |

The fused score correlates strongly (ρ≈0.68–0.74) with raw absolute NDVI change, and is dominated
by its SAM component (ρ≈0.79–0.80) more than by the difference-structure terms. SAM itself
correlates only moderately with the classical magnitude-difference baseline (ρ≈0.60–0.62), so SAM
is not simply reproducing brightness differencing.

**What this correlation does and does not show, stated carefully.** A high ρ between the fused
score and `|ΔNDVI|` is the EXPECTED signature of either cause, not just one of them: genuine
construction (vegetation → bare/built) produces a large NDVI drop, and so does an ordinary seasonal
vegetation cycle. Because both the thing this study wants to detect and the confound it is worried
about produce the same large-|ΔNDVI| signature, this correlation **cannot discriminate between
them** and is not evidence either way on whether physics fusion is or is not separating
construction from seasonal change at this site — that question needs the human/basemap check below
(or a same-season multi-year baseline this 4-date dataset does not have), not this statistic. What
the correlation DOES show, safely: the fused score tracks whole-scene vegetation-index magnitude
fairly closely (ρ≈0.68–0.74) rather than being independent of it, which is a real, reportable
property of this arm's behaviour even though it does not resolve the season-vs-construction
question either way.

### Negative result: the TemporalBaseline arm is not usable at n=2 epochs

`TemporalBaseline(window=2)` was built from the two earliest primary dates (2020-10-16,
2022-03-30) and scored against 2026-06-17 (`change_detection/temporal_baseline.py`). With exactly
2 baseline epochs, the median is the mean of the two values and the MAD is `|a − b| / 2` for every
pixel — a maximally coarse statistic, and any pixel whose two baseline values were nearly equal
produces a near-zero MAD, hitting the module's `1e-6` floor and exploding the z-score. Measured:
mean z = 101.2, median z = 5.4, max z ≈ 1.18×10⁵, and **68.1% of all pixels exceed z > 3** — the
canonical "outlier" threshold is crossed by more than two-thirds of the scene, which is not a
usable detection signal. The median (5.4) is a more honest single-number summary than the mean
here. This arm is included because PLAN.md §8 names `TemporalBaseline` explicitly in the Level 3
pipeline line, and it is reported exactly as it behaved: **not usable as a standalone detector at
this epoch count**, a limitation of having only 2–3 real epochs rather than of the module's
correctness. Outputs: `experiments/phase5_level3/temporal_baseline/`.

### Sensor confound — quantified

Same-sensor (S2B–S2B) vs cross-sensor (S2B–S2A or S2A–S2B) mean SAM, all four measured directly
from this run (not estimated):

| pair type | pair | mean SAM (rad) |
|---|---|---|
| same-sensor | 2020-10-16 → 2022-03-30 | 0.141 |
| same-sensor | 2022-03-30 → 2026-06-17 | 0.176 |
| cross-sensor | 2022-03-30 (S2B) → 2024-10-30 (S2A) | **0.174** |
| cross-sensor | 2024-10-30 (S2A) → 2026-06-17 (S2B) | **0.189** |

Cross-sensor pairs show higher mean SAM than same-sensor pairs, consistent with (not proof of) the
sensor-confound concern named in the task brief — measured band-centre gaps are **B11: 3.3 nm,
B12: 16.7 nm**. This comparison is **not controlled for time gap** (the intervals span 1.5, 4.2,
2.6 and 1.6 years respectively, all different), so it cannot isolate the sensor effect from real
change or from elapsed time alone; it is reported as suggestive corroboration of the design
decision, not as a clean measurement of the sensor effect in isolation. Per the task brief, the
2024-10-30 S2A date and both cross-sensor intervals above live only under
`experiments/phase5_level3/secondary_cross_sensor/`, are excluded from the primary accept-criteria
numbers, and are labelled `SECONDARY_CROSSSENSOR` in every filename, manifest entry, and this
document.

### A documentation-vs-code discrepancy found while building this

The task brief states all four `landcover` indices (ndvi, ndwi, nbr, bsi) are "computable from the
six bands present" — true of the **bands themselves**. It is not true, however, that
`anomaly.scoring.spectral_index_score()` can compute `bsi` on real Sentinel-2 data as shipped:
its SWIR1 lookup target is 1650 nm (a Landsat-style convention) with a fixed 15 nm tolerance
(`preprocessing/bands.py`'s `DEFAULT_TOL_NM`), and Sentinel-2's real B11 centre is ~1610–1614 nm —
36–40 nm away. Confirmed empirically:
`select_band(meta, 1650.0)` raises `ValueError: nearest band to 1650.0 nm is 1610.4 nm (39.6 nm
away, tol_nm=15.0)`. This is very likely a previously-unexercised gap — Sentinel-2 was excluded
from every earlier branch (PLAN.md: "Sentinel-2 is excluded from the background pool... appears
only in Phase 5 Level 3"), so this may be the first time this code path has been run against real
Sentinel-2 data. **This runner does not call `spectral_index_score()`** for that reason; the NDVI/
NDWI/NBR/BSI values reported above reuse the exact formulas in
`anomaly.scoring._INDEX_DEFINITIONS` directly, with bands selected by the verified, fixed 6-band
order documented in `docs/sentinel2_verified.json` rather than by `select_band`'s default-tolerance
search. `anomaly/scoring.py` itself was not modified — fixing its Sentinel-2 tolerance, if wanted,
is future work, not something this task's scope covers (it is shared by the `object` profile's
`ndbi`/`clay_ratio`, both validated on ABU/HAD100, so widening the tolerance there is not risk-free
and was left alone).

### What remains — a HUMAN step, not performed here, Level 3 is NOT self-certified accepted

PLAN.md §8 gives Level 3 no numeric accept criterion of its own (unlike Level 2's "polygon
centroids within 2 px of true position"); its pipeline line ends "...compared against known dated
events," which this document reads as requiring the same kind of human, QGIS-against-a-basemap
check Level 2 needed (`plan.md` D32) — **not performed by this agent, and this run does not claim
it as accepted.** Precise instructions for that human step:

1. Open `qgis/projects/phase5_level3_20201016_to_20220330.qgz` and
   `qgis/projects/phase5_level3_20220330_to_20260617.qgz` in QGIS with the OpenStreetMap basemap
   layer already included.
2. For each project, visually check that the flagged ROI polygons (red outline) sit on land that
   is visibly different between the two source dates against the basemap/known airport footprint —
   not scattered randomly across the AOI. **This eyeball check carries more weight here than it did
   for Level 2**: the reported hectare figures are a fixed top-5%-of-scene selection (see above),
   not an absolute change measurement, so this visual check is the only step in this run that can
   establish whether the flagged area actually sits on genuinely changed ground.
3. Cross-check the first interval's flagged area against the publicly reported 25 Nov 2021
   groundbreaking / Aug 2021 construction-start dates (`docs/validation.md` above,
   documentation-tier, not independently verified against a primary government source) — the
   interval brackets that date (2020-10-16 pre-dates it, 2022-03-30 post-dates it by ~4 months),
   so change consistent with early construction is the expected, checkable pattern, not a causal
   claim this document makes on its own.
4. Record the outcome (pass/fail, and why) in `plan.md` as a dated decision, the way D32 did for
   Level 2 — this agent was explicitly told not to edit `plan.md` itself.

Everything else the task asked for — the pipeline run, the measured numbers, the sensor/season
confound handling and quantification, the `meta.acquired` verification, the negative results — is
complete and is not blocked on this step. Only the visual/basemap sign-off is outstanding.

### Test suite

Full suite re-run after this work: `529 passed, 0 failed` (`pytest -q`, 250.7 s). Nothing in
`scripts/run_level3_case_study.py` is imported by any existing test, and no existing module was
modified except `scripts/build_qgis_project.py` (additive: one new `if` block appending the two
Level 3 QGIS builds to `main()`, nothing existing changed) — so this count reflects the
pre-existing suite's own health, not new coverage of this script. No test was added for the new
runner; the task asked to run the existing suite, not extend it, and this write-up's own numbers
(the D33 timestamp assertion, the registration residuals, the ROI/area counts) are the checks
specific to this run.
