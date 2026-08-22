# QGIS verification — how to close O4

O4 is the last thing gating the Phase 2 exit criterion. It needs a human with
a GUI; there is no QGIS in the dev environment (`which qgis` finds nothing).

**There are two different checks here and they answer different questions.**
Conflating them is the main way this goes wrong, so do them in this order.

---

## Check A — affine plumbing (§2.10, the one O4 names)

**Question:** do the polygons land on the pixels the detector actually fired
on? This is about the raster→world→GeoJSON transform chain being wired
correctly. It is **not** about real-world accuracy.

**Files** (already on disk, `experiments/phase2/`):

| layer | file |
|---|---|
| score raster | `Indian_pines_corrected_anom_norm.tif` |
| ROI polygons | `Indian_pines_corrected_rois.geojson` |
| binary mask | `Indian_pines_corrected_mask.tif` |

**Steps**

1. Install QGIS (`sudo pacman -S qgis` on Arch; or flatpak `org.qgis.qgis`).
2. `Layer → Add Layer → Add Raster Layer…` → `Indian_pines_corrected_anom_norm.tif`
3. Style it: double-click the layer → **Symbology** → Render type **Singleband
   pseudocolor** → colour ramp **Magma** → Min 0, Max 1.
4. `Layer → Add Layer → Add Vector Layer…` → `Indian_pines_corrected_rois.geojson`
5. Style it: **Symbology** → Simple fill → Fill style **No Brush**, stroke
   **red**, width 0.5. Then **Labels** → Single labels → Value `roi_id`.
6. Right-click the vector layer → **Zoom to Layer**.

**What passing looks like:** every red outline sits on top of a bright
(high-score) blob in the magma raster. No systematic offset, no flip, no
90° rotation, no polygons off the raster edge.

**What failing looks like, and what each means:**

| symptom | likely cause |
|---|---|
| polygons offset by a constant | affine translation term wrong |
| polygons mirrored vertically | row origin flipped (north-up vs south-up) |
| polygons rotated 90° | row/col swapped somewhere in `mask_to_rois` |
| polygons at a wildly different scale | pixel-size term wrong |

> **Do NOT add a basemap for this check and do not conclude anything about
> real-world position from it.** Indian Pines ships **no CRS and no affine**
> (D13.1), so the pipeline assigns a synthetic one (D2). The polygons will
> land at some arbitrary spot on Earth. **That is expected and is not a bug.**
> Any Indian Pines result carries `georef: "synthetic"` and §13 rule 7 forbids
> drawing a geospatial accuracy metric from it.

---

## Check B — real georeferencing (stronger, and now possible)

**Question:** do the polygons land in the right place *on Earth*?

This one is real, because the demo now runs on HAD100/AVIRIS, which ships
genuine UTM headers on all 616 scenes (D11.5) parsed via GDAL (D14.2).

```bash
.venv/bin/python pipeline/demo.py --assert-offline --out experiments/demo
```

**Files** (`experiments/demo/`):

| layer | file |
|---|---|
| score raster | `ang20170821t183707_100_anom_norm.tif` |
| ROI polygons | `ang20170821t183707_100_demo.geojson` |

Same styling steps as Check A, then **add a basemap**:
`Browser panel → XYZ Tiles → OpenStreetMap` (drag to the bottom of the layer
stack). If XYZ Tiles is empty, right-click it → New Connection, name
`OpenStreetMap`, URL `https://tile.openstreetmap.org/{z}/{x}/{y}.png`.

**What to expect:** the scene reports **EPSG:32611** (UTM zone 11N) and the
ROIs land near **-114.50, 51.44** — southern Alberta, Canada. The raster
should sit on real terrain, not in the ocean and not at 0°N 0°E.

**A cross-check worth doing, because it is independent of the basemap:** open
the GeoJSON attribute table and look at `area`. A 3-pixel ROI reports about
**26.5 m²**, which implies roughly **3 m** ground sample distance — AVIRIS-NG's
actual GSD. A transform that is wrong by a scale factor would have to be wrong
by a *plausible* factor to survive that test, which is a much stronger
statement than "the polygons look like they are in Canada."

---

## Regenerating the projects

```bash
python3 scripts/build_qgis_project.py        # system python3, NOT .venv/bin/python
```

Rebuilds both `.qgz` files from the pipeline's current outputs, fully styled.
Run it after any rerun of `run_pipeline.py` or `demo.py`.

**Use system `python3` deliberately.** PyQGIS cannot be installed into the
venv: it is not on PyPI, and `qgis/_core.so` is compiled against Python 3.14
while D1 pins the venv to 3.12.13 for `fiona`. Forcing it via `PYTHONPATH`
fails on `PyQt6.sip`. Nothing else in the repo imports `qgis`, so the split
costs nothing.

**A trap in this specific repo:** §2.10 requires output under `qgis/projects/`,
so a directory named `qgis/` sits at the repo root — and Python imports it as
a PEP 420 namespace package, shadowing the real PyQGIS. `import qgis`
**succeeds**; only `qgis.__file__ is None` gives it away. The script strips the
repo root from `sys.path` before importing for exactly this reason.

## Saving the project

§2.10 asks for `qgis/projects/phase2_verify.qgz`:

```bash
mkdir -p qgis/projects
```
Then `Project → Save As…` → `qgis/projects/phase2_verify.qgz`.

Save Check B as `qgis/projects/demo_verify.qgz` alongside it.

`.qgz` files are small and belong in git — check `.gitignore` does not exclude
them before committing.

---

## Reporting the result

If Check A passes, **O4 is closed and Phase 2 exit is signed off**. Say so in
`plan.md` §14 with the date, the QGIS version, and which of the 3 Indian Pines
ROIs you inspected.

If Check B passes, that is **stronger evidence than Phase 2 was ever going to
give** — real georeferencing verified against an independent basemap. It does
not formally close O4 (which names the Phase 2 artefact), but it is the check
that actually matters for the Phase 5 Level 2 claim, and it is worth recording
as a D-entry either way.

A programmatic affine check already passed on all 3 Indian Pines ROIs (D14) —
polygon bounds against `meta.transform`-derived pixel bounds. That is
**partial, not sufficient** evidence: it verifies the transform is applied
consistently, not that it is the right transform. The eyeball is what
distinguishes those two.
