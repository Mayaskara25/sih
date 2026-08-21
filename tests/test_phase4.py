"""PLAN.md Phase 4 -- the detector registry (§4.1) and recall-first
threshold calibration (§4.2).

§4.1's accept criterion is specifically that swapping `detector: global_rx ->
fused` is a CONFIG edit and still passes `validate_geojson`. That is the real
test of whether the frozen contracts held: if integration turns out to need a
rewrite, a contract was violated somewhere and that is the bug to find.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from anomaly.scoring import calibrate_threshold_for_recall
from core.contracts import validate_geojson
from pipeline.run_pipeline import DETECTORS, PATH_DETECTORS, run_pipeline

SCENE = Path("data/benchmark/indian_pines/Indian_pines_corrected.mat")
needs_scene = pytest.mark.skipif(not SCENE.exists(), reason="Indian Pines not downloaded")


# ---------------------------------------------------------------- §4.1 ----

def test_registry_contains_the_phase3_detectors():
    for name in ("global_rx", "local_rx", "kernel_rx", "crd", "fused"):
        assert name in DETECTORS, f"{name} missing from the Phase 4 registry"
        assert callable(DETECTORS[name])


def test_streaming_rx_is_declared_path_only_not_silently_absent():
    """streaming_rx takes a PATH, not a cube (§3A.5), so it cannot satisfy the
    cube->score contract the registry resolves against. It must be *named* as
    an exception rather than quietly missing, or the next reader assumes it
    was forgotten and 'fixes' it into DETECTORS, where it would fail at call
    time with a confusing TypeError."""
    assert "streaming_rx" in PATH_DETECTORS
    assert "streaming_rx" not in DETECTORS


def test_unknown_detector_raises_with_the_available_names():
    with pytest.raises(ValueError, match="unknown detector"):
        run_pipeline(scene=SCENE, source="indian_pines", detector="does_not_exist",
                     threshold_pct=99.0, profile="object", out_dir=Path(tempfile.mkdtemp()))


@needs_scene
def test_path_detector_refused_by_the_cube_pipeline():
    with pytest.raises(ValueError, match="takes a scene path"):
        run_pipeline(scene=SCENE, source="indian_pines", detector="streaming_rx",
                     threshold_pct=99.0, profile="object", out_dir=Path(tempfile.mkdtemp()))


@needs_scene
@pytest.mark.parametrize("detector,params", [
    ("global_rx", None),
    ("fused", {"local_rx": {"outer": 15, "inner": 3, "n_components": 12}}),
])
def test_detector_swap_is_config_only_and_geojson_stays_valid(detector, params, tmp_path):
    """§4.1 accept criterion, verbatim."""
    manifest = run_pipeline(scene=SCENE, source="indian_pines", detector=detector,
                            threshold_pct=99.0, profile="object", out_dir=tmp_path,
                            detector_params=params)
    validate_geojson(Path(manifest["outputs"]["geojson"]))
    assert manifest["detector"] == detector


@needs_scene
def test_detector_params_are_recorded_in_the_manifest(tmp_path):
    """§3A.2 requires per-dataset detector params to come from config, never
    hardcoded. A param that is not in the manifest is not reproducible."""
    params = {"outer": 15, "inner": 3, "n_components": 12}
    manifest = run_pipeline(scene=SCENE, source="indian_pines", detector="local_rx",
                            threshold_pct=99.0, profile="object", out_dir=tmp_path,
                            detector_params=params)
    assert manifest["detector_params"] == params


# ---------------------------------------------------------------- §4.2 ----

def _separable_scores(n_neg=9900, n_pos=100, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.normal(0, 1, n_neg), rng.normal(4, 1, n_pos)])
    labels = np.concatenate([np.zeros(n_neg, bool), np.ones(n_pos, bool)])
    return scores, labels


@pytest.mark.parametrize("target", [0.90, 0.95, 0.98, 1.0])
def test_calibrated_threshold_achieves_at_least_the_target_recall(target):
    scores, labels = _separable_scores()
    thr, _fp = calibrate_threshold_for_recall(scores, labels, target_recall=target)
    achieved = float((scores[labels] >= thr).mean())
    assert achieved >= target - 1e-9, f"target {target}, achieved {achieved}"


def test_threshold_is_the_LOWEST_meeting_the_target_not_merely_one_that_works():
    """§4.2's rule is 'pick the LOWEST threshold achieving target_recall'.
    A conservative implementation that returns a higher threshold would also
    satisfy the recall assertion above while quietly discarding the
    over-triggering the cascade design depends on -- so check tightness:
    nudging the threshold up must drop recall below target."""
    scores, labels = _separable_scores()
    thr, _ = calibrate_threshold_for_recall(scores, labels, target_recall=0.98)
    pos = np.sort(scores[labels])[::-1]
    just_above = float(pos[pos > thr].min()) if (pos > thr).any() else thr + 1e-6
    assert float((scores[labels] >= just_above).mean()) < 0.98


def test_higher_recall_costs_a_higher_false_positive_rate():
    """The FP rate is returned, not just logged, so the compute cost of the
    recall target is explicit rather than hidden (§4.2)."""
    scores, labels = _separable_scores()
    rates = [calibrate_threshold_for_recall(scores, labels, target_recall=t)[1]
             for t in (0.90, 0.98, 1.0)]
    assert rates == sorted(rates), f"FP rate must be non-decreasing in recall, got {rates}"
    assert rates[0] < rates[-1]


def test_nan_scores_are_excluded_not_treated_as_zero():
    """NaN is nodata repo-wide. Treating it as 0.0 would place every nodata
    pixel below any real threshold and silently count it as a confident
    negative, deflating the reported FP rate."""
    scores, labels = _separable_scores()
    poisoned = scores.copy()
    poisoned[labels] = np.where(np.arange(labels.sum()) < 5, np.nan, scores[labels])
    thr, fp = calibrate_threshold_for_recall(poisoned, labels, target_recall=0.98)
    assert np.isfinite(thr) and np.isfinite(fp)


def test_no_positives_raises_rather_than_returning_a_meaningless_threshold():
    scores, labels = _separable_scores()
    with pytest.raises(ValueError, match="no valid positive"):
        calibrate_threshold_for_recall(scores, np.zeros_like(labels))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_target_recall_raises(bad):
    scores, labels = _separable_scores()
    with pytest.raises(ValueError, match="target_recall"):
        calibrate_threshold_for_recall(scores, labels, target_recall=bad)


def test_shape_mismatch_raises():
    scores, labels = _separable_scores()
    with pytest.raises(ValueError, match="shape mismatch"):
        calibrate_threshold_for_recall(scores, labels[:-1])


# ------------------------------------------------- empty-ROI regression ----

def test_zero_rois_writes_a_valid_empty_featurecollection(tmp_path):
    """A scene with no ROIs is a NORMAL outcome, not an error (D23).

    `gpd.GeoDataFrame([], geometry="geometry")` raises `ValueError: Unknown
    column geometry`, so the empty case needs its own path. Found when
    local_rx(outer=15) on Indian Pines produced 211 thresholded pixels of
    which morphological opening removed every one.
    """
    import rasterio.crs
    import rasterio.transform

    from core.contracts import SceneMeta
    from geospatial.geojson import rois_to_geojson

    meta = SceneMeta(
        scene_id="empty_scene",
        crs=rasterio.crs.CRS.from_epsg(32616),
        transform=rasterio.transform.from_origin(0, 0, 10, 10),
        wavelengths=None,
        bad_bands=np.zeros(10, dtype=bool),
        gsd_m=10.0,
        source="indian_pines",
        georef="synthetic",
    )
    out = rois_to_geojson([], meta, tmp_path / "empty.geojson")
    validate_geojson(out)

    import json as _json
    data = _json.loads(out.read_text())
    assert data["type"] == "FeatureCollection"
    assert data["features"] == []


@needs_scene
def test_pipeline_survives_a_detector_config_that_finds_nothing(tmp_path):
    """The end-to-end version of the above: the pipeline must not crash on
    its most benign possible input. Phase 5 L1 runs 113 scenes unattended and
    Phase 7 runs live, so 'found nothing' must be a result, not a traceback."""
    manifest = run_pipeline(
        scene=SCENE, source="indian_pines", detector="local_rx", threshold_pct=99.0,
        profile="object", out_dir=tmp_path,
        detector_params={"outer": 15, "inner": 3, "n_components": 12})
    validate_geojson(Path(manifest["outputs"]["geojson"]))
    assert manifest["n_rois"] == 0
