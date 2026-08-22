"""Tests for edge/roi_pipeline.py + edge/benchmark.py (3D.6).

Uses synthetic scenes so a fresh clone stays green (`skipif` anything needing
data/); the real ABU-Airport-1 acceptance run is exercised separately by
`test_edge_benchmark_abu` when the file is present.
"""
from __future__ import annotations

import affine
import numpy as np
import pytest
import rasterio
import rasterio.crs
import torch
import torch.nn as nn

from anomaly.rx import global_rx
from core.contracts import SceneMeta
from edge.roi_pipeline import (
    DEFAULT_TARGET_RECALL,
    MAX_STAGE2_PIXEL_FRACTION,
    _count_full_scene_pixels,
    _count_stage2_pixels,
    roi_vs_full_comparison,
)


class TinySeg(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Conv2d(in_ch, 1, 3, padding=1)

    def forward(self, x):
        return self.net(x)


def _meta(scene_id="synthetic"):
    return SceneMeta(
        scene_id=scene_id,
        crs=rasterio.crs.CRS.from_epsg(32616),
        transform=affine.Affine(10.0, 0, 100.0, 0, -10.0, 200.0),
        wavelengths=None,
        bad_bands=np.zeros(6, dtype=bool),
        gsd_m=10.0,
        source="abu",
        georef="synthetic",
    )


def _scene_with_targets(h=96, w=96, b=6, seed=0, n_targets=4):
    rng = np.random.default_rng(seed)
    cube = rng.normal(loc=100.0, scale=5.0, size=(h, w, b)).astype(np.float32)
    gt = np.zeros((h, w), dtype=bool)
    for _ in range(n_targets):
        r, c = rng.integers(8, h - 12), rng.integers(8, w - 12)
        cube[r:r + 4, c:c + 4] += rng.uniform(40, 80)      # bright compact targets
        gt[r:r + 4, c:c + 4] = True
    return cube, gt


def _fitted_pca(cube, n_components=3):
    from sklearn.decomposition import PCA

    flat = cube.reshape(-1, cube.shape[-1])
    pca = PCA(n_components=n_components).fit(flat)
    return pca


def test_count_helpers():
    bboxes = [(0, 0, 64, 64), (64, 0, 96, 32)]   # second clips at scene edge 96x96
    assert _count_stage2_pixels((96, 96, 5), [(0, 0, 64, 64)], patch=64) == 64 * 64
    # overlapping windows deduplicated
    both = _count_stage2_pixels((128, 128, 5), bboxes * 2, patch=64)
    assert both == _count_stage2_pixels((128, 128, 5), bboxes, patch=64)
    assert _count_full_scene_pixels((96, 96, 5), patch=64) == 2 * 2 * 64 * 64


def test_roi_vs_full_report_structure_and_accept_logic():
    cube, gt = _scene_with_targets()
    meta = _meta()
    pca = _fitted_pca(cube)
    seg = TinySeg(in_ch=3).eval()

    report = roi_vs_full_comparison(
        cube, meta, gt, global_rx, seg,
        patch=32, batch=4, transformer=pca, n_components=3)

    assert report["measurement"] == "SIMULATED"
    for key in ("stage1_latency_s", "recall_achieved", "threshold", "induced_fp_rate",
                "stage2", "bandwidth", "accept", "n_rois"):
        assert key in report

    s2 = report["stage2"]
    assert s2["pixels_total_scene"] == 96 * 96
    assert s2["fraction_processed_at_stage2"] <= 1.0
    assert s2["fraction_discarded_by_screening"] == pytest.approx(
        1.0 - s2["fraction_processed_at_stage2"])

    bw = report["bandwidth"]
    assert bw["full_cube_bytes"] == cube.nbytes
    assert bw["geojson_bytes"] > 0
    assert bw["ratio_multiple"] > 1.0          # GeoJSON must be smaller than the cube

    acc = report["accept"]
    expected_met = bool(s2["fraction_processed_at_stage2"] < MAX_STAGE2_PIXEL_FRACTION
                        and report["recall_achieved"] >= DEFAULT_TARGET_RECALL)
    assert acc["criterion_met"] == expected_met
    assert isinstance(expected_met, bool)


def test_roi_path_processes_fewer_pixels_than_full_on_clean_scene():
    """On a mostly-background synthetic scene the screening stage must discard
    the vast majority of pixels -- the structural claim of the cascade."""
    cube, gt = _scene_with_targets(seed=7)
    meta = _meta()
    pca = _fitted_pca(cube)
    seg = TinySeg(in_ch=3).eval()

    report = roi_vs_full_comparison(
        cube, meta, gt, global_rx, seg,
        patch=32, batch=4, transformer=pca, n_components=3)

    assert (report["stage2"]["pixels_roi_path"]
            < report["stage2"]["pixels_full_scene"])
