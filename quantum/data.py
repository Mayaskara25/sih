"""PLAN.md 3E -- the frozen HAD100-flightline train/val/test split and pixel
sampling for the whole quantum branch. Four downstream agents (VQC, QAE, quantum
kernel, classical-vs-quantum) call `build_split` and must all get the SAME split,
which is why it is cached to disk rather than re-derived per call.

FLIGHTLINE, NOT PATCH, IS THE SPLIT UNIT. HAD100's 94 test patches come from only
10 AVIRIS-NG flightlines (counted from *.hdr filenames, e.g. "ang20170821t183707_91"
-> flightline "ang20170821t183707"); several patches from the SAME flightline are
near-duplicate crops of the same flight pass. Splitting by PATCH would leak a
flightline's spectral/atmospheric signature across train and test -- the same
scene-leakage failure mode tests/test_data_hygiene.py exists to catch for the
main pipeline's patch splits. FLIGHTLINE_SPLIT is fixed and committed, never
revisited, on the same discipline as experiments/rx_vs_ae/fusion_weights.json's
tune/report split.

Counts verified against the files (2026-08-22, `ls .../aviris_ng_target/*.hdr |
... | uniq -c`): ang20170821t183707=38, ang20171012t194435=12,
ang20170908t225309=10, ang20170821t195229=8, ang20210614t141018=7,
ang20191004t185054=6, ang20170825t173426=5, ang20180227t082814=4,
ang20191004t215336=3, ang20191027t204454=1. Sum 94, matching the total patch
count. Split sums: train 38+10+5+4=57, val 8+1=9, test 12+7+6+3=28 -- matches
this module's brief exactly, so the brief's counts are FILE-VERIFIED, not
documentation-only (CONTRIBUTING.md's evidence standard).

NO aviris_target/ (AVIRIS-Classic) directory exists under HAD100 -- the test set
is AVIRIS-NG only (also file-verified). This module therefore only reads
data/aviris_ng_target, unlike scripts/run_benchmark.py::_had100_scenes which
loops over both subdirs defensively; the loading BODY (spectral for the ENVI
cube, scipy.io for the .mat ground truth, µm->nm rescale) is reused unchanged
from that function -- see _load_scene -- restructured only to load one .hdr at
a time so scenes can be selected by FLIGHTLINE rather than a global
head-of-glob `limit`. scripts/run_benchmark.py itself is never imported or
modified (CONTRIBUTING.md: don't touch scripts/run_benchmark.py).
"""
from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio.crs
import rasterio.transform

from core.contracts import SceneMeta
from preprocessing.harmonize import RETAINED_BANDS, harmonize
from quantum.feature_map import QuantumFeatureTransformer, fit_transformer

ROOT = Path(__file__).resolve().parents[1]
HAD100 = ROOT / "data" / "benchmark" / "had100" / "HAD100"
_NG_DATA = HAD100 / "data" / "aviris_ng_target"
_NG_GT = HAD100 / "gt" / "aviris_ng_gt"
CACHE_DIR = ROOT / "experiments" / "quantum_results"

# Dummy geo fields, same convention scripts/run_benchmark.py uses for HAD100 (it
# ships no real CRS/transform -- SceneMeta requires SOME value, not None, and
# nothing in this branch's pixel-classification path reads them).
_DUMMY_CRS = rasterio.crs.CRS.from_epsg(4326)
_DUMMY_TRANSFORM = rasterio.transform.from_origin(0, 0, 1, 1)

FLIGHTLINE_SPLIT: dict[str, tuple[str, ...]] = {
    "train": ("ang20170821t183707", "ang20170908t225309", "ang20170825t173426",
              "ang20180227t082814"),                                          # 57 patches
    "val": ("ang20170821t195229", "ang20191027t204454"),                      # 9 patches
    "test": ("ang20171012t194435", "ang20210614t141018", "ang20191004t185054",
              "ang20191004t215336"),                                          # 28 patches
}

_FLIGHTLINE_RE = re.compile(r"^(ang\d{8}t\d{6})_\d+$")


def flightline_of(scene_id: str) -> str:
    """'ang20170821t183707_91' -> 'ang20170821t183707'. Raises ValueError on
    anything that doesn't match the ang<8-digit-date>t<6-digit-time>_<patch>
    HAD100/AVIRIS-NG naming convention -- silently falling back to e.g.
    rsplit('_', 1) would mis-split any id that ever gains a second underscore,
    and mis-splitting a flightline id is exactly the leak FLIGHTLINE_SPLIT
    exists to prevent.
    """
    m = _FLIGHTLINE_RE.match(scene_id)
    if not m:
        raise ValueError(
            f"scene_id {scene_id!r} does not match ang<date>t<time>_<patch> "
            "(HAD100/AVIRIS-NG naming convention)")
    return m.group(1)


def stratified_share(total_cap: int, n_flightlines: int) -> int:
    """Ceiling per-flightline row budget when total_cap rows are split evenly
    across n_flightlines groups: ceil(total_cap / n_flightlines). NO leftover
    redistribution -- an underfilled small flightline's spare share is NOT
    handed to a bigger one, because that reintroduces exactly the imbalance
    stratification exists to prevent (ang20170821t183707 has 38 of HAD100's 94
    patches; redistribution would let it eat any slack). Exposed publicly so
    callers and tests can compute the expected per-flightline cap independently
    of _stratified_cap_per_flightline's internals.
    """
    if n_flightlines <= 0:
        return 0
    return -(-total_cap // n_flightlines)                                    # ceil div


def _iter_hdr_by_flightline(flightlines: list[str]) -> dict[str, list[Path]]:
    """.hdr paths under aviris_ng_target, grouped by flightline, sorted within
    each group. Only groups for flightlines actually requested are returned.
    """
    groups: dict[str, list[Path]] = {fl: [] for fl in flightlines}
    for hdr in sorted(_NG_DATA.glob("*.hdr")):
        fl = flightline_of(hdr.stem)
        if fl in groups:
            groups[fl].append(hdr)
    return groups


def _load_scene(hdr_path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """One HAD100 AVIRIS-NG scene: (scene_id, cube[H,W,B] float32, gt[H,W] bool,
    wl[B] float64 nm). Body is scripts/run_benchmark.py::_had100_scenes' per-file
    logic verbatim (spectral for the ENVI cube via .hdr, scipy.io for the .mat
    ground truth, micrometre->nm rescale when header wavelengths are < 100) --
    reused rather than reinvented, just split out to load ONE file so callers
    can select scenes by flightline instead of a global glob-order `limit`.
    """
    import scipy.io as sio
    import spectral

    gt_path = _NG_GT / f"{hdr_path.stem}.mat"
    img = spectral.open_image(str(hdr_path))
    cube = np.asarray(img.load(), dtype=np.float32)
    wl = None
    if getattr(img, "bands", None) is not None and img.bands.centers:
        wl = np.asarray(img.bands.centers, dtype=np.float64)
        if wl.max() < 100:                                   # some ENVI headers use micrometres
            wl = wl * 1000.0
    gm = sio.loadmat(gt_path)
    key = next(k for k in gm if not k.startswith("__"))
    gt = gm[key].astype(bool)
    return hdr_path.stem, cube, gt, wl


def test_scene_ids() -> list[str]:
    """The 28 test-split scene ids (patch-level), for a runner to loop over --
    e.g. `for scene_id in test_scene_ids(): score_scene_natural(arm, scene_id, split)`.
    Touches disk (globs aviris_ng_target); callers needing a data/-free path
    should use FLIGHTLINE_SPLIT['test'] (flightline ids) instead.
    """
    groups = _iter_hdr_by_flightline(list(FLIGHTLINE_SPLIT["test"]))
    return sorted(hdr.stem for hdrs in groups.values() for hdr in hdrs)


# --------------------------------------------------------------- scene sampling --

@dataclass
class _SceneRecord:
    scene_id: str
    flightline: str
    split: str
    X_bg: np.ndarray            # [nb, RETAINED_BANDS] float64, harmonized, no NaN rows
    X_an: np.ndarray            # [na, RETAINED_BANDS] float64


def _build_scene_records(rng: np.random.Generator, *, n_bg_per_scene: int,
                          max_anom_per_scene: int, limit_scenes: int | None
                          ) -> list[_SceneRecord]:
    """Load every scene needed by FLIGHTLINE_SPLIT (optionally capped to the
    first `limit_scenes` patches PER FLIGHTLINE, for a fast smoke path that
    still touches every split -- a global head-of-glob cap would instead load
    88 of 94 patches before reaching the LAST flightline in sorted order, see
    module docstring's cumulative counts, defeating "fast"). Each scene is
    harmonized once and per-scene-capped background/anomaly pixel rows are
    drawn (n_bg_per_scene / max_anom_per_scene) -- these per-scene caps are
    what keeps one huge patch (up to 134 anomaly pixels seen in this data)
    from dominating its own scene's contribution before flightline-level
    stratification even runs.
    """
    all_flightlines = [fl for fls in FLIGHTLINE_SPLIT.values() for fl in fls]
    groups = _iter_hdr_by_flightline(all_flightlines)

    records: list[_SceneRecord] = []
    for split_name, flightlines in FLIGHTLINE_SPLIT.items():
        for fl in flightlines:
            hdrs = groups.get(fl, [])
            if limit_scenes is not None:
                hdrs = hdrs[:limit_scenes]
            for hdr in hdrs:
                scene_id, cube, gt, wl = _load_scene(hdr)
                meta = SceneMeta(scene_id=scene_id, crs=_DUMMY_CRS, transform=_DUMMY_TRANSFORM,
                                  wavelengths=wl, bad_bands=np.zeros(cube.shape[-1], dtype=bool),
                                  gsd_m=1.0, source="had100", georef="real")
                cube_h, _ = harmonize(cube, meta)
                flat = cube_h.reshape(-1, cube_h.shape[-1]).astype(np.float64)
                valid = ~np.any(np.isnan(flat), axis=-1)
                gt_flat = gt.reshape(-1)

                bg_idx = np.flatnonzero(valid & ~gt_flat)
                an_idx = np.flatnonzero(valid & gt_flat)
                if bg_idx.size > n_bg_per_scene:
                    bg_idx = rng.choice(bg_idx, size=n_bg_per_scene, replace=False)
                if an_idx.size > max_anom_per_scene:
                    an_idx = rng.choice(an_idx, size=max_anom_per_scene, replace=False)

                records.append(_SceneRecord(scene_id=scene_id, flightline=fl, split=split_name,
                                             X_bg=flat[bg_idx], X_an=flat[an_idx]))
    return records


def _pool_by_flightline(records: list[_SceneRecord], split_name: str, attr: str
                         ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Concatenate one class ('X_bg' or 'X_an') across all scenes of one
    flightline within one split -> {flightline: (X [n, B], scene_id [n] str)}.
    """
    by_fl: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for rec in records:
        if rec.split != split_name:
            continue
        X = getattr(rec, attr)
        if X.shape[0] == 0:
            continue
        xs, sids = by_fl.setdefault(rec.flightline, ([], []))
        xs.append(X)
        sids.append(np.full(X.shape[0], rec.scene_id))
    return {fl: (np.concatenate(xs), np.concatenate(sids)) for fl, (xs, sids) in by_fl.items()}


def _assemble_split(records: list[_SceneRecord], split_name: str, cap: int | None,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool a split's scenes by flightline, apply `cap` (None -> no cap, used
    for val) as a stratified per-flightline row budget, then concatenate into
    (X_raw [N, RETAINED_BANDS] float64, y [N] uint8, scene_id [N] str).

    Stratification caps EACH FLIGHTLINE to a `divmod(cap, n_flightlines)` share
    (the first `cap % n_flightlines` flightlines get one extra row) -- this is
    the form that satisfies BOTH properties a plain `ceil(cap/n)`-per-flightline
    cap does not: every flightline's share is still <= stratified_share(cap,
    n_flightlines) (so ang20170821t183707's 38-of-94 patches still can't
    dominate), AND the shares SUM TO EXACTLY `cap` rather than up to
    `n * ceil(cap/n)` -- a plain ceil-per-flightline share with no
    redistribution overshoots `cap` on every non-divisible split (e.g.
    max_train_total=601 over 4 flightlines: ceil(601/4)=151 per flightline,
    4*151=604 > 601). Each flightline's own share is further split into a
    background half and an anomaly half (ceil/floor), with the anomaly half
    floored at 1 (borrowed from the background half) whenever that flightline
    actually has anomaly rows available -- otherwise a share of 1 always
    rounds to 1 background / 0 anomaly, and a small enough cap can silently
    zero out every anomaly row in the split. No cap-exceeding, no cross-
    flightline redistribution: an underfilled flightline's spare share is
    simply unused, never handed to a bigger flightline.
    """
    flightlines = sorted({r.flightline for r in records if r.split == split_name})
    bg_pool = _pool_by_flightline(records, split_name, "X_bg")
    an_pool = _pool_by_flightline(records, split_name, "X_an")

    if cap is not None and flightlines:
        n = len(flightlines)
        base, rem = divmod(cap, n)
        shares = {fl: base + (1 if i < rem else 0) for i, fl in enumerate(flightlines)}
        for fl in flightlines:
            share = shares[fl]
            share_bg = -(-share // 2)
            share_an = share - share_bg
            if share_an == 0 and share > 0 and an_pool.get(fl, (np.zeros((0,)),))[0].shape[0] > 0:
                share_an, share_bg = 1, share - 1
            if fl in bg_pool:
                X, sid = bg_pool[fl]
                if X.shape[0] > share_bg:
                    keep = rng.choice(X.shape[0], size=share_bg, replace=False)
                    bg_pool[fl] = (X[keep], sid[keep])
            if fl in an_pool:
                X, sid = an_pool[fl]
                if X.shape[0] > share_an:
                    keep = rng.choice(X.shape[0], size=share_an, replace=False)
                    an_pool[fl] = (X[keep], sid[keep])

    X_list, y_list, sid_list = [], [], []
    for fl in flightlines:
        if fl in bg_pool:
            X, sid = bg_pool[fl]
            X_list.append(X); y_list.append(np.zeros(X.shape[0], dtype=np.uint8)); sid_list.append(sid)
        if fl in an_pool:
            X, sid = an_pool[fl]
            X_list.append(X); y_list.append(np.ones(X.shape[0], dtype=np.uint8)); sid_list.append(sid)

    if not X_list:
        return (np.zeros((0, RETAINED_BANDS), dtype=np.float64),
                np.zeros((0,), dtype=np.uint8), np.zeros((0,), dtype="<U32"))
    return np.concatenate(X_list), np.concatenate(y_list), np.concatenate(sid_list)


# --------------------------------------------------------------------- QuantumSplit --

@dataclass(frozen=True)
class QuantumSplit:
    X_train: np.ndarray         # [N, n_features] float64, angles in [0, pi]
    y_train: np.ndarray         # [N] uint8, 1 = anomaly
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scene_test: np.ndarray      # [N_test] str, patch scene_id -- for scene-macro pooling
    n_features: int
    transformer: object         # QuantumFeatureTransformer, fit on TRAIN only (D15)
    seed: int


def _cache_paths(seed: int, n_features: int) -> tuple[Path, Path]:
    stem = f"split_seed{seed}_f{n_features}"
    return CACHE_DIR / f"{stem}.npz", CACHE_DIR / f"{stem}.transformer.pkl"


def _try_load_cache(cache_npz: Path, cache_pkl: Path, build_params: dict) -> QuantumSplit | None:
    """Any mismatch, missing file, or load error is treated as a cache MISS
    (returns None -> caller rebuilds) rather than raised -- a stale or
    differently-parameterized cache silently feeding four downstream agents a
    split they didn't ask for is a worse failure mode than a slower rebuild.
    """
    if not (cache_npz.exists() and cache_pkl.exists()):
        return None
    try:
        with np.load(cache_npz, allow_pickle=False) as z:
            for k, v in build_params.items():
                if int(z[f"param_{k}"]) != v:
                    return None
            with open(cache_pkl, "rb") as f:
                transformer = pickle.load(f)
            return QuantumSplit(
                X_train=z["X_train"], y_train=z["y_train"],
                X_val=z["X_val"], y_val=z["y_val"],
                X_test=z["X_test"], y_test=z["y_test"],
                scene_test=z["scene_test"],
                n_features=int(z["n_features"]), transformer=transformer, seed=int(z["seed"]))
    except Exception:
        return None


def _save_cache(cache_npz: Path, cache_pkl: Path, split: QuantumSplit, build_params: dict) -> None:
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        X_train=split.X_train, y_train=split.y_train,
        X_val=split.X_val, y_val=split.y_val,
        X_test=split.X_test, y_test=split.y_test,
        scene_test=split.scene_test,
        n_features=split.n_features, seed=split.seed,
        **{f"param_{k}": v for k, v in build_params.items()},
    )
    np.savez(cache_npz, **save_kwargs)
    with open(cache_pkl, "wb") as f:
        pickle.dump(split.transformer, f)


def build_split(*, n_features: int = 8, n_bg_per_scene: int = 40,
                 max_anom_per_scene: int = 40, max_train_total: int = 600,
                 max_test_total: int = 600, seed: int = 0,
                 limit_scenes: int | None = None, rebuild: bool = False) -> QuantumSplit:
    """Build (or reload) the frozen-flightline HAD100 pixel split for the
    quantum branch. Reused by four downstream agents (VQC, QAE, quantum
    kernel, classical-vs-quantum) -- cached to
    experiments/quantum_results/split_seed{seed}_f{n_features}.npz (+ a sibling
    .transformer.pkl for the fitted QuantumFeatureTransformer, which an .npz
    cannot hold) so rebuilding from raw ENVI on every call is avoided.

    CACHING RULES (deliberately conservative -- a wrong cache hit silently
    hands four agents the wrong split, which is worse than a slow rebuild):
      - limit_scenes is not None -> the cache is BYPASSED entirely, neither
        read nor written. A `limit_scenes=2` smoke-path split must never be
        written to the same path a full build reads, and must never overwrite
        a good full build either.
      - rebuild=True -> the on-disk cache is not READ (always rebuilds from
        raw ENVI), but IS still written afterward (when limit_scenes is None),
        so this is the escape hatch for a stale/corrupt cache, not a one-off
        no-cache run -- use limit_scenes for that.
      - n_bg_per_scene / max_anom_per_scene / max_train_total / max_test_total
        are stored in the cache and checked on load; a mismatch is a MISS
        (silently rebuilds), not an error -- see _try_load_cache.

    Fits reduce_bands + the angle range (fit_transformer, quantum/feature_map.py)
    on TRAIN-split pixels ONLY, then reuses that fitted QuantumFeatureTransformer
    to transform val and test WITHOUT refitting (D15) -- see
    QuantumFeatureTransformer.transform for why out-of-train-range val/test
    pixels are CLIPPED to [0, pi] rather than left to alias past it.

    Per-scene caps (n_bg_per_scene, max_anom_per_scene) balance within a scene;
    max_train_total / max_test_total then subsample STRATIFIED ACROSS
    FLIGHTLINES (stratified_share), so ang20170821t183707's 38-of-94 patches
    cannot dominate the pooled budget. val has no total cap (only the
    per-scene caps apply) -- FLIGHTLINE_SPLIT['val'] is only 9 patches from 2
    flightlines, small enough that a further cap isn't needed.

    TWO DIFFERENT TEST POPULATIONS, NOT INTERCHANGEABLE. X_test/y_test/
    scene_test (this function's return) is capped and roughly class-balanced
    -- for FITTING and threshold work, e.g. picking an operating point. It is
    NOT a reporting population: at small `limit_scenes` it can cover only a
    subset of the 28 test scenes (a scene-macro pool over it silently omits
    whichever scenes lost the stratified draw). REPORTED numbers must go
    through test_scene_ids() + score_scene_natural, which scores every test
    scene at its own natural (not capped/balanced) prevalence with an explicit
    reweighting. Four downstream agents publishing off two different
    populations under the same "test AUC" label would be exactly the kind of
    incomparable-numbers mistake this split exists to prevent.
    """
    build_params = dict(n_bg_per_scene=n_bg_per_scene, max_anom_per_scene=max_anom_per_scene,
                         max_train_total=max_train_total, max_test_total=max_test_total)
    cache_npz, cache_pkl = _cache_paths(seed, n_features)

    if limit_scenes is None and not rebuild:
        cached = _try_load_cache(cache_npz, cache_pkl, build_params)
        if cached is not None:
            return cached

    rng = np.random.default_rng(seed)
    records = _build_scene_records(rng, n_bg_per_scene=n_bg_per_scene,
                                    max_anom_per_scene=max_anom_per_scene,
                                    limit_scenes=limit_scenes)

    X_train_raw, y_train, _ = _assemble_split(records, "train", max_train_total, rng)
    X_val_raw, y_val, _ = _assemble_split(records, "val", None, rng)
    X_test_raw, y_test, scene_test = _assemble_split(records, "test", max_test_total, rng)

    transformer = fit_transformer(X_train_raw, n_features=n_features)
    X_train = transformer.transform(X_train_raw).astype(np.float64)
    X_val = (transformer.transform(X_val_raw).astype(np.float64) if X_val_raw.shape[0]
             else np.zeros((0, n_features), dtype=np.float64))
    X_test = (transformer.transform(X_test_raw).astype(np.float64) if X_test_raw.shape[0]
              else np.zeros((0, n_features), dtype=np.float64))

    split = QuantumSplit(
        X_train=X_train, y_train=y_train.astype(np.uint8),
        X_val=X_val, y_val=y_val.astype(np.uint8),
        X_test=X_test, y_test=y_test.astype(np.uint8),
        scene_test=scene_test, n_features=n_features, transformer=transformer, seed=seed)

    if limit_scenes is None:
        _save_cache(cache_npz, cache_pkl, split, build_params)

    return split


# ------------------------------------------------------------- natural-prevalence scoring --

def _natural_prevalence_sample(gt_flat: np.ndarray, valid: np.ndarray, *,
                                max_bg_per_scene: int, seed: int
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Index selection + reweighting for score_scene_natural: ALL valid anomaly
    pixels plus a random background sample capped at max_bg_per_scene, plus a
    per-row weight that recovers natural-prevalence PR-AUC from that
    anomaly-enriched subsample. Pulled out of score_scene_natural so it is
    testable on a synthetic population without loading a scene from disk or
    paying a real arm's per-sample scoring cost.

    COST, NOT JUST CORRECTNESS, IS WHY THIS SAMPLES AT ALL. Scoring every pixel
    of all 28 test patches is ~200k pixels at ~0.4% anomaly prevalence; VQC's
    SamplerQNN forward costs ~17 ms/sample, so full-population scoring is ~1h
    PER quantum arm -- not affordable across four arms x 28 scenes. ROC-AUC is
    prevalence-invariant, so a plain subsample doesn't distort it, but PR-AUC is
    NOT prevalence-invariant -- an unweighted anomaly-enriched subsample reports
    a PR-AUC for a roughly-balanced population, not the true ~0.4%-prevalence
    one, and INFLATES it (measured on a synthetic 200k-pixel, 0.4%-prevalence
    population with 4800 scored pixels: full-population PR-AUC 0.3395,
    weighted-subsample 0.3259, unweighted-subsample 0.8695 -- unweighted more
    than doubles the number, in the flattering direction. D27.7).

    Returns (idx, weight): idx are positions into the scene's FULL flat pixel
    array (gt_flat.shape == valid.shape) to actually score -- np.flatnonzero
    over the full-length valid/gt_flat masks, NOT re-indexed into flat[valid].
    A caller must index the same full-length array (flat[idx], gt_flat[idx]),
    same as score_scene_natural does; indexing flat[valid][idx] instead is
    silently wrong on any scene with invalid pixels. weight[i] = 1.0 for an
    anomaly row, n_bg_total / n_bg_sampled for a background row.
    average_precision_score(labels, scores, sample_weight=weight) on
    (idx, weight) is then the natural-prevalence PR-AUC estimate; calling it
    WITHOUT sample_weight on the same subsample is not.
    """
    an_idx = np.flatnonzero(valid & gt_flat)
    bg_idx = np.flatnonzero(valid & ~gt_flat)

    rng = np.random.default_rng(seed)
    n_bg_total = bg_idx.size
    n_bg_take = min(max_bg_per_scene, n_bg_total)
    bg_take = (rng.choice(bg_idx, size=n_bg_take, replace=False)
               if n_bg_take else np.zeros((0,), dtype=bg_idx.dtype))

    idx = np.concatenate([an_idx, bg_take])
    weight = np.ones(idx.size, dtype=np.float64)
    if n_bg_take:
        weight[an_idx.size:] = n_bg_total / n_bg_take
    return idx, weight


def score_scene_natural(arm, scene_id: str, split: QuantumSplit, *,
                         max_bg_per_scene: int = 400, seed: int = 0
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores one TEST-split scene at NATURAL prevalence -> (scores, labels,
    sample_weight).

    Takes ALL anomaly pixels plus a random background sample of at most
    max_bg_per_scene, and weights the background back up by
    n_background_total / n_background_sampled, so
    average_precision_score(y, s, sample_weight=w) is the natural-prevalence
    PR-AUC rather than the balanced one (D27.7). Scoring every pixel instead
    would cost ~1h per quantum arm for a number this recovers to within 0.014
    from 4800 pixels (see _natural_prevalence_sample for the measurement).

    `arm` is any object with .score(X) -> [N] higher-is-more-anomalous. The
    pixels are transformed with `split.transformer` -- fit on TRAIN, NEVER
    refit here (D15).

    Raises ValueError if scene_id's flightline is not in FLIGHTLINE_SPLIT['test']:
    scoring a train/val scene "at natural prevalence" and reporting it is
    exactly the tune/report leak this repo's split discipline exists to
    prevent (see experiments/rx_vs_ae/fusion_weights.json's tune/report note).
    """
    fl = flightline_of(scene_id)
    if fl not in FLIGHTLINE_SPLIT["test"]:
        raise ValueError(
            f"score_scene_natural: {scene_id!r} (flightline {fl!r}) is not in the "
            f"TEST split {FLIGHTLINE_SPLIT['test']} -- scoring a train/val scene at "
            "natural prevalence and reporting it is exactly the leak the tune/report "
            "split discipline exists to prevent")

    hdr_path = _NG_DATA / f"{scene_id}.hdr"
    if not hdr_path.exists():
        raise FileNotFoundError(f"score_scene_natural: no HAD100 scene at {hdr_path}")

    _, cube, gt, wl = _load_scene(hdr_path)
    meta = SceneMeta(scene_id=scene_id, crs=_DUMMY_CRS, transform=_DUMMY_TRANSFORM,
                      wavelengths=wl, bad_bands=np.zeros(cube.shape[-1], dtype=bool),
                      gsd_m=1.0, source="had100", georef="real")
    cube_h, _ = harmonize(cube, meta)
    flat = cube_h.reshape(-1, cube_h.shape[-1]).astype(np.float64)
    valid = ~np.any(np.isnan(flat), axis=-1)
    gt_flat = gt.reshape(-1)

    idx, weight = _natural_prevalence_sample(gt_flat, valid, max_bg_per_scene=max_bg_per_scene,
                                              seed=seed)
    X = split.transformer.transform(flat[idx])
    scores = np.asarray(arm.score(X))
    labels = gt_flat[idx].astype(np.uint8)
    return scores, labels, weight


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    split = build_split(n_features=args.n_features, seed=args.seed,
                         limit_scenes=args.limit_scenes, rebuild=args.rebuild)
    print(f"train: X{split.X_train.shape} y{split.y_train.shape} "
          f"(anomaly frac {split.y_train.mean():.4f})" if split.y_train.size else "train: empty")
    print(f"val:   X{split.X_val.shape} y{split.y_val.shape}")
    print(f"test:  X{split.X_test.shape} y{split.y_test.shape} scenes={len(set(split.scene_test))}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
