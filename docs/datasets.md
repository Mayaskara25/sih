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

---

## DOCUMENTED ONLY — nothing opened yet, treat every number as provisional

**Scheduled for file-opened verification as a Phase 5 prerequisite (PLAN.md §8.0).** Do not
write Phase 5 code against these specs until that runs. "Not yet flagged as a problem" is not
the same as "verified" — HAD100 sat in this tier for one draft.

| Dataset | Claimed | Needs checking |
|---|---|---|
| EnMAP L2A | 224 bands, 30 m GSD, DLR Geoservice STAC, 5 000-product cap | band count, dtype, wavelength availability, tile structure, actual no-data handling |
| Sentinel-2 L2A | ~13 bands multispectral, Copernicus Data Space | band subset per product, resolution mixing (10/20/60 m), Phase 5 Level 3 only |
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

See [buildable_now.md](buildable_now.md) — the critical path is unblocked; EnMAP L2A access (PLAN.md O11) gates Phase 5 Level 2 only.

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
