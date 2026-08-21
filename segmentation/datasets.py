"""§3B.3 -- PyTorch Dataset wrappers over synthetic (training) and real
(scoring-only) data. D7's governing rule: train on synthetic, score on real,
never both on the same data -- enforced structurally in RealSegDataset, not
just documented.

Splitting is by SOURCE SCENE via GroupShuffleSplit keyed on
preprocessing.background_pool.scene_groups() -- the one sanctioned mechanism
(D11.5: the four corner crops of one scene overlap by up to ~62px per axis,
so a patch-index split leaks near-duplicates across train/val).
tests/test_data_hygiene.py demonstrates the leak this prevents.

reduce_bands (PCA/kPCA, RETAINED_BANDS -> C=30) is fit HERE, on a bounded
random subsample of the TRAIN split's pixels streamed from the memmap
(D15, D17) -- never on the whole pool, never loaded fully into RAM.

PCA does not normalize its output scale -- HAD100 patches are raw
radiance-scale DN, not reflectance, so the reduced C=30 cube comes out with
components in the thousands. Found the hard way: training under AMP fp16
produced NaN loss from epoch 1, traced to activations overflowing fp16's
range on that unnormalized input. Both dataset classes standardize
(per-band zero-mean/unit-variance, preprocessing/normalize.py) the reduced
cube before it reaches the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from preprocessing.background_pool import BackgroundPatch, scene_groups
from preprocessing.harmonize import RETAINED_BANDS, harmonize, reduce_bands
from preprocessing.normalize import standardize
from preprocessing.raster_loader import load_scene
from segmentation.synth import implant_targets, pseudo_anomaly_patch

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "processed" / "had100_background_pool.npy"
MANIFEST_PATH = ROOT / "data" / "processed" / "had100_background_manifest.csv"
HAD100_ROOT = ROOT / "data" / "benchmark" / "had100" / "HAD100"

N_COMPONENTS = 30

# Verbatim from HAD100/main.py's crop_dict (18 keys, 6 with 2 crops each =>
# 94 + 6 = 100 test patches, D11.2). Not re-derived as a judgement call --
# an upstream change to main.py shows up here as a mismatch, not silently
# absorbed. (row, col) offsets, 0-based, into the RAW (pre-harmonize) scene.
HAD100_TEST_CROP_DICT: dict[str, list[tuple[int, int]]] = {
    'ang20170821t183707_102': [(0, 15)],
    'ang20170821t183707_124': [(0, 0)],
    'ang20170821t195229_21': [(99 - 63, 99 - 63)], 'ang20170825t173426_14': [(0, 0)],
    'ang20170825t173426_17': [(0, 99 - 63)],
    'ang20170908t225309_32': [(0, 0), (99 - 63, 80 - 63)], 'ang20170908t225309_34': [(99 - 63, 0)],
    'ang20170908t225309_36': [(0, 0), (10, 99 - 63)], 'ang20170908t225309_37': [(0, 0), (10, 99 - 63)],
    'ang20170908t225309_39': [(0, 79 - 63)], 'ang20171012t194435_23': [(0, 0)],
    'ang20180227t082814_15': [(0, 0), (99 - 63, 99 - 63)],
    'ang20180227t082814_17': [(0, 99 - 63), (99 - 63, 5)],
    'ang20191004t185054_22': [(0, 0)], 'ang20191004t185054_27': [(0, 3)],
    'ang20210614t141018_32': [(0, 80 - 63)],
    'ang20210614t141018_37': [(10, 0), (99 - 63, 99 - 63)], 'ang20210614t141018_38': [(99 - 63, 0)],
}


def load_manifest_records(manifest_path: Path = MANIFEST_PATH) -> list[BackgroundPatch]:
    df = pd.read_csv(manifest_path)
    return [BackgroundPatch(**row) for row in df.to_dict("records")]


def train_val_scene_split(*, manifest_path: Path = MANIFEST_PATH,
                           val_fraction: float = 0.2, seed: int = 0) -> dict:
    """The ONE sanctioned split: GroupShuffleSplit keyed on scene_groups(),
    never on patch/array index directly (D11.5)."""
    records = load_manifest_records(manifest_path)
    groups = scene_groups(records)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(records)), groups=groups))

    train_scene_ids = set(groups[train_idx].tolist())
    val_scene_ids = set(groups[val_idx].tolist())
    assert train_scene_ids.isdisjoint(val_scene_ids), "scene leaked across the split"

    return dict(
        train_array_indices=np.array([records[i].array_index for i in train_idx]),
        val_array_indices=np.array([records[i].array_index for i in val_idx]),
        train_scene_ids=train_scene_ids, val_scene_ids=val_scene_ids,
    )


def fit_reduce_bands_transformer(*, pool_path: Path = POOL_PATH,
                                  train_array_indices: np.ndarray,
                                  n_components: int = N_COMPONENTS,
                                  n_sample_pixels: int = 200_000, seed: int = 0,
                                  method: str = "pca") -> tuple[object, int]:
    """Fits reduce_bands' transformer on a bounded random subsample of
    TRAIN-split pixels, streamed from the memmap one patch at a time --
    never loads the whole 6.29 GB pool into RAM (D17), and never touches a
    val/eval patch (D15: fitting outside the train split leaks scoring-scene
    statistics into the representation every model shares).

    Returns (transformer, n_pixels_actually_sampled).
    """
    if len(train_array_indices) == 0:
        raise ValueError("train_array_indices is empty")

    pool = np.load(pool_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    pixels_per_patch = max(1, n_sample_pixels // len(train_array_indices))

    samples = []
    for idx in train_array_indices:
        patch = np.asarray(pool[int(idx)])
        flat = patch.reshape(-1, patch.shape[-1])
        valid_idx = np.where(~np.any(np.isnan(flat), axis=-1))[0]
        if len(valid_idx) == 0:
            continue
        take = min(pixels_per_patch, len(valid_idx))
        chosen = rng.choice(valid_idx, size=take, replace=False)
        samples.append(flat[chosen])
    del pool

    fit_pixels = np.concatenate(samples, axis=0).astype(np.float32)
    dummy_cube = fit_pixels[:4].reshape(2, 2, RETAINED_BANDS)
    _discard, transformer = reduce_bands(
        dummy_cube, n_components=n_components, method=method, fit_on=fit_pixels)
    return transformer, fit_pixels.shape[0]


class SyntheticSegDataset(Dataset):
    """implanted OR pretext, flag-selected. Draws raw 184-band patches from
    the cached background pool (mmap, D17), applies synth.py's perturbation,
    then reduce_bands -- transform-only, via the pre-fitted transformer,
    NEVER refit here (that would be exactly the leak D15 exists to prevent).
    """

    def __init__(self, *, mode: str, array_indices: np.ndarray, transformer,
                 n_components: int = N_COMPONENTS, pool_path: Path = POOL_PATH,
                 seed: int = 0, target_spectra: np.ndarray | None = None,
                 spectrum_ids: list[str] | None = None,
                 n_targets: int = 2, n_regions: int = 3):
        if mode not in ("implanted", "pretext"):
            raise ValueError(f"mode must be 'implanted' or 'pretext', got {mode!r}")
        if mode == "implanted" and target_spectra is None:
            raise ValueError("mode='implanted' requires target_spectra (see synth.load_target_spectra)")

        self.mode = mode
        self.array_indices = np.asarray(array_indices)
        self.transformer = transformer
        self.n_components = n_components
        self.pool_path = Path(pool_path)
        self.seed = seed
        self.target_spectra = target_spectra
        self.spectrum_ids = spectrum_ids
        self.n_targets = n_targets
        self.n_regions = n_regions
        self._pool = None   # opened lazily so DataLoader worker processes each open their own mmap

    def _pool_array(self) -> np.ndarray:
        if self._pool is None:
            self._pool = np.load(self.pool_path, mmap_mode="r")
        return self._pool

    def __len__(self) -> int:
        return len(self.array_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.array_indices[i])
        background = np.array(self._pool_array()[idx])   # copies ONE 64x64x184 patch out of the mmap
        item_seed = (self.seed * 1_000_003 + idx) % (2**31 - 1)

        if self.mode == "pretext":
            cube, mask = pseudo_anomaly_patch(background, n_regions=self.n_regions, seed=item_seed)
        else:
            cube, mask, _meta = implant_targets(
                background, self.target_spectra, n_targets=self.n_targets,
                seed=item_seed, spectrum_ids=self.spectrum_ids)

        reduced, _ = reduce_bands(cube, n_components=self.n_components, fit_on=self.transformer)
        normed = standardize(reduced)   # PCA output isn't scale-normalized; see module docstring
        patch = np.moveaxis(normed, -1, 0).astype(np.float32)   # [C,H,W]
        return torch.from_numpy(patch), torch.from_numpy(mask[None].astype(np.float32))


class RealSegDataset(Dataset):
    """abu | hydice_urban_anomaly | had100 -- EVAL ONLY (D7). had100/test is
    the only source with a wavelength array (D13.4/O8), so it is the only
    one a learned model can be scored on; abu/hydice_urban_anomaly raise a
    named error pointing at the classical detectors (3A) instead of silently
    producing an unregistered result.
    """

    def __init__(self, *, source: str, split: str, transformer,
                 n_components: int = N_COMPONENTS, had100_root: Path = HAD100_ROOT):
        assert split == "eval", (
            "RealSegDataset is scoring-only. Training on real GT violates D7.")
        if source in ("abu", "hydice_urban_anomaly"):
            raise NotImplementedError(
                f"RealSegDataset(source={source!r}): no wavelength array to harmonize onto "
                "the canonical grid (D13.4/O8). Learned models score on HAD100 only; use the "
                "classical detectors (anomaly/rx.py, anomaly/crd.py) for this source instead.")
        if source != "had100":
            raise ValueError(f"unknown source {source!r}, expected 'abu', 'hydice_urban_anomaly', or 'had100'")

        self.transformer = transformer
        self.n_components = n_components
        self.had100_root = Path(had100_root)
        self._items = self._enumerate_test_crops()

    def _enumerate_test_crops(self) -> list[tuple[str, int, int, str]]:
        items = []
        target_dir = self.had100_root / "data" / "aviris_ng_target"
        for hdr_path in sorted(target_dir.glob("*.hdr")):
            name = hdr_path.stem
            crops = HAD100_TEST_CROP_DICT.get(name, [(0, 0)])
            for i, (row0, col0) in enumerate(crops):
                save_name = f"{name}_{i + 1}" if len(crops) > 1 else name
                items.append((name, row0, col0, save_name))
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        scene_name, row0, col0, _save_name = self._items[i]
        hdr_path = self.had100_root / "data" / "aviris_ng_target" / f"{scene_name}.hdr"
        cube, meta = load_scene(hdr_path, source="had100")
        harmonized, _new_meta = harmonize(cube, meta)
        patch_184 = harmonized[row0:row0 + 64, col0:col0 + 64]

        gt_path = self.had100_root / "gt" / "aviris_ng_gt" / f"{scene_name}.mat"
        gt = loadmat(gt_path)["gt"]
        mask = (gt[row0:row0 + 64, col0:col0 + 64] > 0).astype(np.float32)

        reduced, _ = reduce_bands(patch_184, n_components=self.n_components, fit_on=self.transformer)
        normed = standardize(reduced)
        patch = np.moveaxis(normed, -1, 0).astype(np.float32)
        return torch.from_numpy(patch), torch.from_numpy(mask[None])
