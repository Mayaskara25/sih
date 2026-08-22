# QGIS verification — result, and how to re-run it

**O4 is CLOSED (D26, 2026-08-22, QGIS 4.2.1 "Belém do Pará"). Phase 2 exit is
signed off.** This document records what was verified, what was deliberately
*not* verified, and how to reproduce either check after a pipeline rerun.

Two checks. They answer **different questions**, and conflating them is the
main way this goes wrong — so they live in separate project files with
deliberately different setups.

| | Check A | Check B |
|---|---|---|
| project | `qgis/projects/phase2_verify.qgz` | `qgis/projects/demo_verify.qgz` |
| data | Indian Pines | HAD100 `ang20170821t183707_100` |
| CRS | EPSG:32616 (**synthetic**, D2) | EPSG:32611 (**real**, D11.5) |
| basemap | **none, on purpose** | OpenStreetMap |
| question | do polygons land on the right *pixels*? | do they land on the right *place on Earth*? |
| status | **PASS** | **PASS** |

---

## Running them

```bash
qgis qgis/projects/phase2_verify.qgz
qgis qgis/projects/demo_verify.qgz
```

Both open fully styled — magma 0–1 raster, red ROI outlines, `roi_id` labels
with halos, correct project CRS and a saved view extent. No Symbology work.

Three harmless startup messages, all safe to dismiss:

- **"Wayland session detected: user experience will be degraded"** — cosmetic.
  Use `QT_QPA_PLATFORM=xcb qgis …` if dialogs misbehave.
- **Master password / wallet** — QGIS's credential store for database
  connections. This project uses none. *Settings → Options → Authentication*
  to silence permanently.
- **DB Manager deprecation** — unrelated plugin notice.

---

## Check A — affine plumbing (§2.10)

**Question:** do the polygons land on the pixels the detector actually fired
on? This tests the raster→world→GeoJSON transform chain. It says **nothing**
about real-world accuracy.

**Result: PASS.** The magma raster renders with visible structure and the 3
ROIs sit on high-score pixels. No offset, mirror, rotation or scale error.

**If you re-run it and it fails,** the symptom names the cause:

| symptom | cause |
|---|---|
| polygons offset by a constant | affine translation term wrong |
| polygons mirrored vertically | row origin flipped (north-up vs south-up) |
| polygons rotated 90° | row/col swapped in `mask_to_rois` |
| polygons at a wildly different scale | pixel-size term wrong |

> **There is no basemap in this project, deliberately.** Indian Pines ships
> **no CRS and no affine** (D13.1), so the pipeline assigns a synthetic one
> (D2). The scene will land at an arbitrary spot on Earth. **That is expected
> and is not a bug.** Every Indian Pines result carries `georef: "synthetic"`,
> and §13 rule 7 forbids drawing any geospatial accuracy metric from it.

**Finding the ROIs.** They are **20, 11 and 5 pixels** in a 145×145 raster, so
at full-scene zoom they are small even at the deliberately-thick 1.4 outline
width. Right-click the ROI layer in the **Layers panel** (bottom-left, *not*
the map canvas) → **Zoom to Layer**. To make them unmistakable, open the
attribute table and `Ctrl+A` — selected features highlight yellow.

---

## Check B — real georeferencing

**Question:** do the polygons land in the right place *on Earth*?

Real, because HAD100 ships genuine UTM headers on all 616 scenes (D11.5),
parsed via GDAL (D14.2) rather than a hand-rolled `map info` reader.

**Result: PASS.** The scene lands on **Dogpound Creek, Alberta, Canada**,
matching the transform of the GeoTIFF's own bounds (−114.501…−114.499,
51.440…51.442) computed independently of QGIS. At 1:301 and 1:64 the outlines
trace the bright pixels **including the cross-shaped notches of the mask
boundary** — so the polygons follow the actual connected component, not its
bounding box.

**Three independent lines of evidence agree on the same affine:**

1. the ENVI header's own UTM `map info`, via GDAL;
2. OpenStreetMap placing the scene on a *named* real-world feature;
3. **a derived GSD** — a 3-pixel ROI reports 26.5 m², implying ~3 m ground
   sample distance, which matches AVIRIS-NG.

The third is the most useful, because it is **independent of the basemap**: a
transform wrong by a scale factor would have to be wrong by a *plausible*
factor to survive it. Check it yourself in the attribute table.

### What Check B does NOT establish

Phase 5 Level 2's accept criterion is *"polygon centroids for a **manually
identified feature** land within 2 pixels (~2 GSD) of its true position."*
That needs a target whose true position is independently known. These ROIs are
**unlabelled anomalies over scrubland** with no ground truth to compare
against.

So: **scene-level placement is verified; feature-level accuracy is not.**
Level 2 stays open on that criterion, and **no location-accuracy number may be
quoted from this check.**

---

## Regenerating the projects

```bash
python3 scripts/build_qgis_project.py        # system python3, NOT .venv/bin/python
```

Rebuilds both `.qgz` files from the pipeline's current outputs, fully styled.
Run it after any rerun of `run_pipeline.py` or `demo.py`. The `.qgz` files are
committed, so the styling is reproducible rather than living in one person's
session.

**Use system `python3` deliberately.** PyQGIS **cannot** be installed into the
venv, and this is structural, not a missing step: it is not on PyPI (it ships
with the `qgis` system package), and `qgis/_core.so` is compiled against
**Python 3.14** while D1 pins the venv to **3.12.13** for `fiona`. Forcing it
via `PYTHONPATH` fails on `PyQt6.sip` (verified). The only alternative would be
rebuilding QGIS against 3.12 — trading a real constraint for a cosmetic one.
Nothing else in the repo imports `qgis`, so the split costs nothing.

---

## Three traps this exercise found

All three produced **convincing false signals** — the reason they are written
down rather than just fixed.

**1. `import qgis` succeeds against an empty directory.** §2.10 requires output
under `qgis/projects/`, so a directory named `qgis/` sits at the repo root, and
Python imports it as a PEP 420 **namespace package**, shadowing the real
PyQGIS. `import qgis` returns cleanly; only `qgis.__file__ is None` reveals it.
That exact check was run, passed, and PyQGIS was reported "available in the
venv" — where it is not installed at all. `build_qgis_project.py` now strips
the repo root from `sys.path` before importing.

**2. An empty project CRS puts the basemap in the wrong country.** Every layer
was individually correct — raster EPSG:32611, ROIs EPSG:4326, OSM EPSG:3857 —
and `QgsProject.crs()` was empty, so on-the-fly reprojection misplaced the
tiles. The raster rendered, the polygons rendered, and OSM painted a detailed,
plausible **French town** underneath, which reads as *"the georeferencing is
broken"* rather than *"the project has no CRS."* The arithmetic that settles
it: read the scene's UTM easting/northing (673818, 5701837) as **Web Mercator**
metres and you get lon 6.05, lat 45.5 — Montmélian, France, exactly where the
tiles drew. The data was right the whole time.

**3. A `Null` default view extent gives a blank white canvas.** QGIS opens at
scale 1:1 near the origin while the data sits at easting ~500 000. Both layers
load and style correctly in the panel and the map is empty — indistinguishable
from a styling failure. The projects now save a 20 %-padded extent. If you ever
see this with any layer: right-click → **Zoom to Layer**.

The common thread, shared with D22/D24/D25 from the same day: **a check that
passes for the wrong reason is worse than one that fails.**
