"""§12 test_pipeline_e2e.py -- full run on a fixture, validate_geojson green."""
import numpy as np
import rasterio

from core.contracts import validate_geojson
from pipeline.run_pipeline import run_pipeline


def test_pipeline_runs_end_to_end_on_32x32_fixture(tmp_path):
    rng = np.random.default_rng(0)
    h, w, b = 32, 32, 10
    cube = rng.normal(size=(h, w, b)).astype(np.float32)
    cube[20:23, 20:23] += 40.0   # implanted anomaly blob

    scene_path = tmp_path / "fixture_scene.tif"
    transform = rasterio.Affine(10.0, 0, 400_000.0, 0, -10.0, 4_000_000.0)
    with rasterio.open(
        scene_path, "w", driver="GTiff", height=h, width=w, count=b, dtype="float32",
        crs="EPSG:32616", transform=transform,
    ) as ds:
        ds.write(np.moveaxis(cube, -1, 0))

    out_dir = tmp_path / "out"
    manifest = run_pipeline(
        scene=scene_path, source="enmap", detector="global_rx", threshold_pct=99.0,
        profile="object", out_dir=out_dir,
    )

    geojson_path = out_dir / f"{scene_path.stem}_rois.geojson"
    assert geojson_path.exists()
    validate_geojson(geojson_path)

    assert (out_dir / "run_manifest.json").exists()
    assert manifest["n_rois"] >= 1
    for stage_name in ("load", "detector", "geojson"):
        assert manifest["timings_s"][stage_name] >= 0.0
