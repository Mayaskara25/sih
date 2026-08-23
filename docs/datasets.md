# Datasets — provenance and verification status

Two tiers, deliberately. A row is **VERIFIED** only if its files were downloaded and every
variable loaded. A row is **DOCUMENTED** if its numbers come from a project page or paper and
nothing has been opened. HAD100 is why the distinction exists: its documentation was right
about the dataset and wrong about the archive in five separate ways (PLAN.md D11).

Re-derive the verified rows with `scripts/verify_had100.py` and `scripts/verify_benchmarks.py`.
Both exit non-zero on drift.

---

## VERIFIED — files opened, invariants asserted

### HAD100 — `data/benchmark/had100/`
Li et al., IEEE TGRS 2023, doi:10.1109/TGRS.2023.3258067 · [repo](https://github.com/ZhaoxuLi123/HAD100) · [site](https://zhaoxuli123.github.io/HAD100/)

- `HAD100.zip` — 3 754 846 290 B, sha256 `6f91035543b7bc7806ebc555cd5411d320f2810f0af8dfcf20fe4f331227c19f`
- **Raw:** 94 test scenes (425 bands) · 260 AVIRIS-NG background (425) · 262 AVIRIS-Classic background (224)
- **Unpacked by the repo's own `main.py`:** 100 test patches + 2 088 background patches, all 64×64
- Raw test scenes span **8 distinct spatial shapes up to 120×120** — not uniformly 64×64
- AVIRIS-Classic: mixed BIL/BIP interleave, mixed int16/float32, **wavelengths non-monotonic in 262/262 files**
- All 616 scenes carry real UTM/WGS-84 georeferencing; no-data sentinels `-9999.0` **and** `1e-34`
- Model input harmonizes from the **raw** ENVI cubes, not `main.py`'s band-selected output (D11.6)

### ABU — `data/benchmark/abu/`
[Kang et al.](http://xudongkang.weebly.com/data-sets.html) — 13 scenes, 4 airport / 4 beach / 5 urban

- **Seven distinct band counts:** 205 ×5, 191 ×2, 188 ×2, 193, 204, 207, and 102 (`abu-beach-4`, ROSIS)
- **Three dtypes:** int16 ×8, uint16 ×4, float64 ×1 — per-scene, never assume uniform
- Spatial: 100×100 ×11, 150×150 ×2
- Anomaly density **0.084 % – 2.72 %**, a 32× range — see the pooling rule in PLAN.md §3A.10
- int16 scenes contain genuine small negatives (min −50 … −1); `abu-beach-4` is float64 in [0,1], i.e. reflectance, while the rest are raw DN
- **Ships no wavelength array** (O8)

### HYDICE anomaly scene — `data/benchmark/hydice_urban_anomaly/`
[github.com/sxt1996/HYDICE](https://github.com/sxt1996/HYDICE)

- `HYDICE-urban.mat` — 2 544 597 B, sha256 `a998766a7180bcacaf5d2163d57857726d80b42f64b490083de85f072b593f4b`
- `README.md` — 1 201 B, sha256 `ffe0592be4ce8b2d8520220b1f6fb20e858c55a2209b9ef56273263b0f49d39f` (keep it; it is the provenance)
- **80 × 100 × 175** float64 in [0,1]; binary mask, **21 anomaly pixels in 10 connected components**
- Michigan, USA. **Ships no wavelength array** (O8)

> ### ⚠️ The Copperas Cove HYDICE is NOT this dataset and must never be substituted for it.
>
> A second, unrelated dataset is also called "HYDICE Urban": the **Copperas Cove, TX unmixing
> scene — 307 × 307, 210 → 162 bands, whose ground truth is six-endmember abundance maps rather
> than an anomaly mask.** It is a spectral-unmixing benchmark, cannot be scored against pixel
> masks, and will silently produce meaningless anomaly-detection numbers if swapped in.
>
> The two are indistinguishable by filename — the file we want is itself named
> `HYDICE-urban.mat`. **The committed SHA256 in `scripts/fetch_hydice.py` is the only reliable
> discriminator**, which is why that fetcher is pinned and asserts shape and anomaly-pixel count
> after download. If you searched "HYDICE" and found a 162-band file, you have the wrong one.
>
> The directory is named `hydice_urban_anomaly/` — not `hydice_urban/` — specifically to make
> the distinction visible in every path that mentions it.

### Indian Pines — `data/benchmark/indian_pines/`
[huggingface.co/datasets/danaroth/indian_pines](https://huggingface.co/datasets/danaroth/indian_pines)
(the `ehu.eus/ccwintco` URLs now return HTML error pages)

- `indian_pines_corrected` (145, 145, 200) uint16, range [955, 9604] — **raw DN, not reflectance**
- `indian_pines_gt` (145, 145) uint8, 16 classes + background
- **No CRS, no affine, no `map info`, no wavelengths** — one variable per file. This is the
  verified basis for PLAN.md D2's synthetic-affine design.
- Phase 2 wiring only. **Never used as an anomaly ground truth.**

### EnMAP L2A — `data/raw/enmap/` (2026-08-23, PLAN.md D32)

Promoted from documentation-only. **Eight complete L2A products** (~3.6 GB, 40 files) were found
already on local disk — see PLAN.md O11/D32 for how that fact sat alongside a "blocked" status
for two days. Re-derive with `scripts/verify_enmap.py`; writes `docs/enmap_verified.json`.

- `*-SPECTRAL_IMAGE_COG.TIF`: **224 bands, int16, nodata −32768.0, GSD 30 m** — uniform across
  all 8 products. **CRS is NOT uniform** — EPSG:32642 (×3) / 32643 (×3) / 32644 (×2), UTM zone
  follows scene longitude, expected and not a defect.
- Wavelengths ship only in the sidecar `*-METADATA.XML`/`.XML.XML` (inconsistently named — 1 of
  8 products uses the single suffix, 7 of 8 the double; both are on disk, both must be handled),
  never in the TIFF. **418.416–2445.30 nm, 224 points, strictly ascending, all finite,
  byte-identical (SHA-256) across all 8 products** — one wavelength grid, not eight.
- `GainOfBand` (the reflectance scale factor): **0.0001, uniform** across all 224 bands and all
  8 products. Recorded; **not applied** by the loader — `SceneMeta` carries no scale-factor
  field, and this project already leaves ABU's mixed radiometric scales unscaled, so applying it
  here alone would be an undocumented asymmetry, not a fix.
- `harmonize.coverage_ok == False` on all 8 (8/184 canonical bands uncovered) — reproduces
  PLAN.md D16 exactly, independently re-derived rather than assumed from that entry.
- **A fact D16 did not have, because D16 verified metadata only, never pixel data:** bands
  131–135 (1342.82–1390.48 nm, the edge of the 1350–1450 nm water-absorption window) are
  effectively 100% nodata, constant across all 8 products — not the same as the ~30% border
  nodata every band carries, and it silently zeroes a naive detector's entire valid-pixel set
  unless flagged (`preprocessing/raster_loader.py` now does, generically, for any `.tif` source).
- Valid (non-nodata) pixel fraction: **mean 0.740, range 0.697–0.765** across the 8 products.
- **Still documentation-only for this dataset:** licence/redistribution terms, the DLR Geoservice
  STAC's 5 000-product cap (a catalogue claim, not re-checked here), and whether the live
  download leg currently succeeds (PLAN.md O11 — explicitly untested by this entry; CLAUDE.md
  scopes credential/network checks out of this work).
- Phase 5 Level 2 ran on one product (a real, georeferenced windowed crop of it — PLAN.md D32);
  the human QGIS-against-basemap step is pending.

### Sentinel-2 L2A — `data/raw/sentinel2/jewar_airport/`

Copernicus Data Space Ecosystem, `eodata` S3 bucket (`https://eodata.dataspace.copernicus.eu`) —
access verified live 2026-08-23 (`scripts/verify_access.py::cdse`), 4 products opened
(`scripts/verify_sentinel2.py` → `docs/sentinel2_verified.json`). Site rationale in
`docs/validation.md` (O5).

- **Plan claim "~13 bands multispectral" is TRUE for the metadata definition** (13
  `<Spectral_Information>` entries, bandId 0–12, re-confirmed from a fresh `MTD_MSIL2A.xml` read)
  **but the L2A product actually delivers imagery for 12 bands** — B10 (cirrus, 1376.9 nm) has no
  image file in L2A at all; atmospheric correction removes it as a surface quantity. SCL/AOT/WVP/
  TCI are additional non-spectral auxiliary layers, not counted in either number.
- **Plan claim "10/20/60 m resolution mixing, must not assume one grid" is TRUE at the sensor
  level** (confirmed per-band from `<RESOLUTION>`: B02/B03/B04/B08 = 10 m, B05/B06/B07/B8A/B11/B12
  = 20 m, B01/B09/B10 = 60 m) **but FALSE for the `landcover` profile's actual band set** — all 6
  bands it needs (B02/B03/B04/B8A/B11/B12) ship pre-resampled to a single 20 m grid by ESA's own
  L2A processor, so this project's fetcher performs **zero on-the-fly resampling**. A different
  index set needing B08 (10 m-only) or B01/B09 (60 m-only) would still need real resampling.
- NIR for the landcover indices is **B8A** (864 nm nominal), not B08 — B08 has no 20 m copy.
- **`BOA_ADD_OFFSET` = -1000, uniform across all 13 metadata bands, on every one of 4 products
  checked; `BOA_QUANTIFICATION_VALUE` = 10000**, also uniform — both read from each product's own
  `MTD_MSIL2A.xml`, never assumed, despite the 4 products spanning 4 different processing
  baselines (05.00/05.10/05.11/05.12).
- `meta.acquired` source is `MTD_TL.xml`'s per-tile `SENSING_TIME`, **not**
  `MTD_MSIL2A.xml`'s `PRODUCT_START_TIME`/`DATATAKE_SENSING_START` (the whole datatake/strip's
  start) — measured to differ from the tile's own sensing time by ~12–14 minutes on every product
  checked. Using the datatake-level field would have been wrong, not just imprecise.
- Wavelength centres are **not constant across S2A/B/C**: e.g. B12 centre measured at 2202.4 nm
  (S2A) vs 2185.7 nm (S2B) on the same tile — read per-product from each `MTD_MSIL2A.xml`, never
  hardcoded.
- CRS/transform (pixel grid) is **byte-identical across all 4 dates fetched** (fixed MGRS tile
  grid, EPSG:32643) — confirmed, not assumed; no co-registration needed for this AOI/tile before
  `TemporalBaseline`.
- SCL band present at 20 m, values found restricted to the documented ESA PSD-15 class codes
  (0–11); AOI clear fraction recomputed independently from SCL (not just trusted from the
  fetcher's own tag) = **1.000 on all 4 dates used**.
- Cube dtype as fetched: **uint16, raw digital numbers — the offset/quantification above are
  recorded on every file but NOT applied**, matching the EnMAP convention (PLAN.md D32): a
  consumer must apply `(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE` before computing any
  reflectance-based index.
- Total local bytes for the 4-date, 6-band + SCL, 6×6 km AOI fetch: **4,701,576 B (~4.5 MB)** —
  see `scripts/fetch_sentinel2.py`'s docstring for the measured network-vs-local relationship
  (windowed JP2 reads via GDAL `/vsis3/`, not whole-tile downloads).
- **Still documentation-only:** licence/redistribution terms were not checked here.

---

## DOCUMENTED ONLY — nothing opened yet, treat every number as provisional

**Scheduled for file-opened verification as a Phase 5 prerequisite (PLAN.md §8.0).** Do not
write Phase 5 code against these specs until that runs. "Not yet flagged as a problem" is not
the same as "verified" — HAD100 sat in this tier for one draft.

| Dataset | Claimed | Needs checking |
|---|---|---|
| AVIRIS / AVIRIS-NG flightlines | 224 / 425 bands, NASA open portal | per-flightline band count and wavelength file availability |
| USGS splib07 | `doi:10.5066/F7RR1WDJ`, plus ECOSTRESS/ASTER | record format, resampling convention, wavelength grid, licence terms |

## Credentials

EnMAP (DLR / EOC UMS) and Sentinel-2 (Copernicus) require accounts. Real values live at
`~/.config/sih/credentials.env` (mode 600), **outside this repository**; `.env.example` here
lists the variable names. Load them with `core.credentials.require("cdse" | "dlr")` and check
configuration with `scripts/check_credentials.py`, which prints booleans and never values.

Copernicus uses **S3 keys** (`CDSE_S3_ACCESS_KEY` / `CDSE_S3_SECRET_KEY`), generated at
<https://eodata-s3keysmanager.dataspace.copernicus.eu/>. The account password is never stored:
catalogue *search* needs no credentials at all, and S3 covers the download leg while staying
revocable on its own. Keys carry a chosen expiry — pick a long one (PLAN.md O10). The Sentinel
Hub OAuth flow is **not** used; that dashboard was sunset 2026-03-20 and served the wrong API.

**Never print the credentials file.** See `CLAUDE.md`. Full rationale in PLAN.md §4.1b.

## What is buildable while EnMAP is blocked

**Correction, 2026-08-23 (PLAN.md D32): EnMAP L2A was never blocking Phase 5 Level 2 — 8 complete
products were on local disk, verified above, and Level 2 has been run.** The line below is left
as originally written; only this correction changes it. See [buildable_now.md](buildable_now.md)
— the critical path was unblocked regardless, and what actually still gates anything EnMAP-related
is the live DLR download leg (untested by this correction — PLAN.md O11), which matters only for
*acquiring more* EnMAP scenes or for O12/SpectralEarth, not for Phase 5 Level 2 itself.

<!-- BEGIN test_env.py version table (auto-generated) -->
## Environment

| package | version |
|---|---|
| numpy | 2.5.2 |
| scipy | 1.18.0 |
| sklearn | 1.9.0 |
| rasterio | 1.5.1 |
| geopandas | 1.1.4 |
| shapely | 2.1.2 |
| fiona | 1.10.1 |
| pyproj | 3.7.2 |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| onnx | 1.22.0 |
| onnxruntime | 1.29.0 |
| cv2 | 5.0.0.93 |
| skimage | 0.26.0 |
| matplotlib | 3.11.1 |
| qiskit | 2.5.2 |
| qiskit_aer | 0.17.2 |
| qiskit_machine_learning | 0.9.1 |
| h5py | 3.16.0 |
| yaml | 6.0.3 |
| tqdm | 4.70.0 |
| pandas | 3.0.5 |
| psutil | 7.2.2 |
| spectral | 0.25 |
| pytest | 9.1.1 |

<!-- END test_env.py version table -->
