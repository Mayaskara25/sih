#!/usr/bin/env python3
"""Build the PLAN.md §2.10 QGIS verification projects, styled, headlessly.

§2.10 asks for `qgis/projects/phase2_verify.qgz` loading the score raster
(magma, 0-1) plus the ROI GeoJSON (red outline, no fill, labelled by
`roi_id`). Doing that by hand through Symbology dialogs is fine once and
unreproducible forever; this builds it from the pipeline's own outputs so the
check can be re-created after any rerun.

Two projects, because they answer DIFFERENT questions (see
docs/qgis_verification.md):

  phase2_verify.qgz  -- Indian Pines. Affine PLUMBING only. Its georeferencing
                        is SYNTHETIC (D2/D13.1), so the polygons land at an
                        arbitrary spot on Earth by design. No basemap: a
                        basemap here invites exactly the wrong conclusion.

  demo_verify.qgz    -- HAD100/AVIRIS. REAL UTM georeferencing (D11.5), read
                        from the ENVI header via GDAL (D14.2). OpenStreetMap
                        included, because here the real-world position is the
                        thing being checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "qgis" / "projects"

# RUN THIS WITH SYSTEM python3, NOT .venv/bin/python.
#
# PyQGIS CANNOT be installed into the venv, and this is structural rather than
# a missing step. It is not on PyPI (it ships with the `qgis` system package),
# and `qgis/_core.so` is a compiled C++ extension built against **Python
# 3.14**, while D1 pins this venv to **3.12.13** for fiona. Pointing
# PYTHONPATH at the system site-packages gets as far as
# `ModuleNotFoundError: No module named 'PyQt6.sip'` -- verified 2026-08-22.
# The only way to have PyQGIS inside the venv would be to rebuild QGIS against
# 3.12, which trades a real constraint for a cosmetic one.
#
# So the split is deliberate: system python3 for QGIS project generation,
# .venv/bin/python for the pipeline. Nothing else in the repo imports qgis.
#
# And drop any repo-root entry from sys.path FIRST. §2.10 requires the output
# at `qgis/projects/`, so this repo contains a directory literally named
# `qgis/` -- which Python happily imports as a PEP 420 namespace package,
# shadowing the real PyQGIS. That failure is quiet and convincing: `import
# qgis` SUCCEEDS, and only `qgis.__file__ is None` gives it away. It briefly
# fooled me into reporting PyQGIS as available in the venv when it is not.
for _p in ("", str(ROOT), str(Path.cwd())):
    while _p in sys.path:
        sys.path.remove(_p)

from qgis.core import (  # noqa: E402
    QgsApplication, QgsColorRampShader, QgsFillSymbol, QgsPalLayerSettings,
    QgsProject, QgsRasterLayer, QgsRasterShader, QgsReferencedRectangle,
    QgsSingleBandPseudoColorRenderer,
    QgsTextBufferSettings, QgsTextFormat, QgsVectorLayer, QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor, QFont  # noqa: E402


def _magma_0_1(layer: QgsRasterLayer) -> None:
    """Singleband pseudocolor, magma, min 0 max 1 -- §2.10's exact ask."""
    stops = [(0.0, "#000004"), (0.25, "#51127c"), (0.5, "#b73779"),
             (0.75, "#fc8961"), (1.0, "#fcfdbf")]
    ramp = QgsColorRampShader(0.0, 1.0)
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(
        [QgsColorRampShader.ColorRampItem(v, QColor(c), f"{v:.2f}") for v, c in stops])
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    r = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
    # Set the classification bounds EXPLICITLY. Without them the layer has no
    # computed band statistics yet and the legend renders its min/max labels
    # as "nan / nan" -- the canvas is correct but the legend reads like the
    # raster is empty, which is alarming and wrong. The C2 contract already
    # guarantees _anom_norm is in [0, 1], so there is nothing to compute.
    r.setClassificationMin(0.0)
    r.setClassificationMax(1.0)
    layer.setRenderer(r)


def _red_outline(layer: QgsVectorLayer, label_field: str = "roi_id") -> None:
    layer.setRenderer(layer.renderer())
    # Thick outline ON PURPOSE. The Phase 2 ROIs are 5, 11 and 20 PIXELS in a
    # 145x145 raster (measured), so at full-scene zoom a hairline outline is
    # invisible and reads as "the vector layer failed to load". Legibility of
    # the check beats cartographic restraint here.
    layer.renderer().setSymbol(QgsFillSymbol.createSimple(
        {"color": "0,0,0,0", "outline_color": "255,0,0,255",
         "outline_width": "1.4", "style": "no"}))
    if label_field in [f.name() for f in layer.fields()]:
        s = QgsPalLayerSettings()
        # Label with the ROI NUMBER, not the whole roi_id. roi_id is
        # "{scene_id}:{branch}:{NNNN}", and scene_id carries the full product
        # name -- so labelling the raw field printed ~76-character strings like
        # "S2B_MSIL2A_20220330T052639_N0510_R105_T43RGM_..._stack:change:0012"
        # across every ROI, which made the map unreadable at any zoom (observed
        # on the Level 2 and Level 3 projects; recorded in plan.md D32).
        # Everything after the last colon is the zero-padded index; the full
        # roi_id is still one click away in the attribute table, so no
        # information is lost. An id with no colon falls through unchanged.
        s.fieldName = f"'#' || regexp_replace(\"{label_field}\", '^.*:', '')"
        s.isExpression = True
        s.enabled = True
        fmt = QgsTextFormat()
        fmt.setFont(QFont("Sans", 9))
        fmt.setColor(QColor("#ffffff"))
        buf = QgsTextBufferSettings()      # halo: white-on-magma is unreadable
        buf.setEnabled(True)
        buf.setSize(1.0)
        buf.setColor(QColor("#000000"))
        fmt.setBuffer(buf)
        s.setFormat(fmt)
        s.placement = QgsPalLayerSettings.Placement.OverPoint
        layer.setLabeling(QgsVectorLayerSimpleLabeling(s))
        layer.setLabelsEnabled(True)


def build(name: str, raster: Path, vector: Path, *, basemap: bool, note: str) -> Path | None:
    if not raster.exists() or not vector.exists():
        print(f"  SKIP {name}: missing {raster.name if not raster.exists() else vector.name}")
        return None
    proj = QgsProject.instance()
    proj.clear()

    # SET THE PROJECT CRS EXPLICITLY, before adding layers.
    #
    # Without it QgsProject has an EMPTY crs and on-the-fly reprojection
    # misbehaves: the layers are each individually correct (raster EPSG:32611,
    # ROIs EPSG:4326, OSM EPSG:3857) and the basemap still lands in the wrong
    # country. The failure is quiet and extremely convincing -- the raster
    # renders, the polygons render, and OSM draws a detailed, plausible town
    # underneath them, so it reads as "the georeferencing is wrong" rather
    # than "the project has no CRS".
    #
    # The arithmetic that gives it away: this scene sits at UTM 11N easting
    # 673818, northing 5701837, which is Alberta, Canada. Read those same two
    # numbers as WEB MERCATOR metres and you get lon 6.05, lat 45.5 --
    # Montmelian, France, which is exactly where the basemap drew. The
    # basemap was being fetched for the projected coordinates taken at face
    # value. The raster was right the whole time.

    rl = QgsRasterLayer(str(raster), f"{raster.stem} (anomaly score)")
    if not rl.isValid():
        print(f"  SKIP {name}: raster invalid"); return None
    _magma_0_1(rl)
    proj.setCrs(rl.crs())          # the scene's own CRS is the project's CRS

    vl = QgsVectorLayer(str(vector), f"{vector.stem} (ROIs)", "ogr")
    if not vl.isValid():
        print(f"  SKIP {name}: vector invalid"); return None
    _red_outline(vl)

    # Order matters: basemap bottom, raster middle, polygons on top.
    if basemap:
        url = ("type=xyz&url=https://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png"
               "&zmax=19&zmin=0")
        bm = QgsRasterLayer(url, "OpenStreetMap", "wms")
        if bm.isValid():
            proj.addMapLayer(bm)
        else:
            print(f"  note: basemap layer would not load (offline?) -- continuing without")
    proj.addMapLayer(rl)
    proj.addMapLayer(vl)
    proj.setTitle(note)

    # SAVE A DEFAULT VIEW EXTENT. Without this the project stores a Null
    # extent and QGIS opens at scale 1:1 somewhere near the origin -- while
    # the data sits at UTM eastings around 500 000. The result is a blank
    # white canvas with both layers correctly loaded and styled in the panel,
    # which reads exactly like "the styling failed" and is in fact a camera
    # position problem. Padded 20% so the ROIs are not flush against the edge.
    ext = rl.extent()
    ext.scale(1.2)
    proj.viewSettings().setDefaultViewExtent(
        QgsReferencedRectangle(ext, rl.crs()))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.qgz"
    proj.write(str(path))
    print(f"  wrote {path.relative_to(ROOT)}   crs={rl.crs().authid() or 'NONE'}  "
          f"rois={vl.featureCount()}")
    return path


def main() -> int:
    QgsApplication.setPrefixPath("/usr", True)
    app = QgsApplication([], False)
    app.initQgis()
    try:
        p2 = ROOT / "experiments" / "phase2"
        build("phase2_verify",
              p2 / "Indian_pines_corrected_anom_norm.tif",
              p2 / "Indian_pines_corrected_rois.geojson",
              basemap=False,
              note=("Phase 2 affine PLUMBING check (PLAN.md 2.10). Indian Pines "
                    "georeferencing is SYNTHETIC (D2) -- do NOT read real-world "
                    "position from this project."))
        d = ROOT / "experiments" / "demo"
        rasters = sorted(d.glob("*_anom_norm.tif"))
        vectors = sorted(d.glob("*_demo.geojson"))
        if rasters and vectors:
            build("demo_verify", rasters[0], vectors[0], basemap=True,
                  note=("HAD100/AVIRIS REAL georeferencing check. UTM header read "
                        "via GDAL (D14.2). Basemap included because real-world "
                        "position IS the question here."))
        else:
            print("  SKIP demo_verify: run `python pipeline/demo.py` first")

        l2 = ROOT / "experiments" / "phase5_level2"
        l2_rasters = sorted(l2.glob("*_anom_norm.tif"))
        l2_vectors = sorted(l2.glob("*_rois.geojson"))
        if l2_rasters and l2_vectors:
            build("phase5_level2_verify", l2_rasters[0], l2_vectors[0], basemap=True,
                  note=("Phase 5 Level 2 (PLAN.md O11/D32): EnMAP L2A, REAL "
                        "georeferencing (COG GeoTIFF CRS/transform, D-note). This is "
                        "the QGIS-against-basemap human check the accept criterion "
                        "requires -- manually identify a feature (coastline, road, "
                        "field boundary, built structure) near a ROI centroid and "
                        "confirm it lands within ~2 pixels (~60 m at 30 m GSD) of "
                        "its true position on OpenStreetMap. NOT yet performed by a "
                        "human as of this write -- do not treat this project file's "
                        "existence as the check having passed."))
        else:
            print("  SKIP phase5_level2_verify: run "
                  "`python -m pipeline.run_pipeline --source enmap ...` first")

        l3 = ROOT / "experiments" / "phase5_level3" / "intervals"
        l3_pairs = sorted(zip(sorted(l3.glob("*_change_norm.tif")),
                               sorted(l3.glob("*_change_rois.geojson"))))
        for raster, vector in l3_pairs:
            name = "phase5_level3_" + raster.name.replace("_change_norm.tif", "").replace("-", "")
            build(name, raster, vector, basemap=True,
                  note=("Phase 5 Level 3 (PLAN.md §8): Sentinel-2 S2B-only change interval "
                        f"{raster.stem.replace('_change_norm', '')}, Jewar Airport AOI, REAL "
                        "georeferencing (EPSG:32643, from the GeoTIFF header). SAM + physics-"
                        "fusion change score, magma 0-1. Basemap included so a human can check "
                        "flagged regions against real-world features -- this is the QGIS check "
                        "Level 3's accept criteria still needs (see docs/validation.md); this "
                        "project's existence does NOT mean that check has been performed."))
        if not l3_pairs:
            print("  SKIP phase5_level3_*: run `python scripts/run_level3_case_study.py` first")
    finally:
        app.exitQgis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
