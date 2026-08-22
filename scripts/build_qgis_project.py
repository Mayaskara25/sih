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

# RUN THIS WITH SYSTEM python3, NOT .venv/bin/python. PyQGIS ships as an
# Arch system package under /usr/lib/python3.14/site-packages and is not pip
# installable into the venv; the venv genuinely does not have it.
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
        s.fieldName = label_field
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
    finally:
        app.exitQgis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
