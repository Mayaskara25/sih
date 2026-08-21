"""§3B.1/3B.2 -- synthetic training data. D7's governing rule: train on
synthetic, score on real, never both on the same data.

Two branches:
  implant_targets      -- linear-mixing implantation of real/library target
                           spectra into HAD100 background patches.
  pseudo_anomaly_patch  -- self-supervised pretext task, zero real target
                           spectra, the honest zero-prior comparison arm.

Both operate on the 184-band canonical grid (harmonize()'s output) -- band
reduction to C=30 happens later, in segmentation/datasets.py, behind the
train/eval split (D15).
"""
from __future__ import annotations

import numpy as np

# Which real datasets a spectra pool's provenance traces back to. "lib" carries
# no dataset identity (library endmembers); "abu_real"/"hyd_real" do, which is
# exactly why scoring a model trained on them against that same dataset would
# be spectrum-level leakage (§3B.1) even with zero real ABU/HYDICE *scenes* in
# training.
SPECTRA_POOLS = {
    "lib": ("usgs_splib07", "ecostress_aster"),
    "abu_real": ("abu",),
    "hyd_real": ("hydice_urban_anomaly",),
}


def load_target_spectra(pools: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """[K, RETAINED_BANDS] on the canonical grid, plus a per-spectrum
    provenance tag (matching SPECTRA_POOLS[pool]).

    "lib" and "abu_real"/"hyd_real" both raise today, for different reasons
    -- see each branch below. Neither is a bug to route around; both are
    named, verified gaps (2026-08-21 plan review), not silent failures.
    """
    all_spectra: list[np.ndarray] = []
    all_provenance: list[str] = []

    for pool in pools:
        if pool not in SPECTRA_POOLS:
            raise ValueError(f"unknown spectra pool {pool!r}, expected one of {sorted(SPECTRA_POOLS)}")

        if pool == "lib":
            raise FileNotFoundError(
                "SPECTRA_POOLS['lib'] (USGS splib07 + ECOSTRESS/ASTER) has not been fetched -- "
                "no scripts/fetch_speclib.py exists and nothing is on disk. Build that fetcher "
                "first (PLAN.md §1.6); this is a separate task, not a harmonize()/synth.py bug.")

        if pool in ("abu_real", "hyd_real"):
            raise NotImplementedError(
                f"SPECTRA_POOLS[{pool!r}] is suspended pending O9 (2026-08-21 plan review): "
                "ABU and HYDICE ship no wavelength array (D13.4), so target spectra harvested "
                "from their ground-truth masks cannot be harmonized onto the 184-band canonical "
                "grid needed to implant them into HAD100 backgrounds. This mirrors the existing "
                "suspension of unet_lodo_abu/unet_lodo_hyd (§3B.8) for the identical reason, "
                "applied consistently to the training-data side rather than only the scoring "
                "side. Will un-suspend automatically if O9 recovers per-scene wavelengths.")

    return (
        np.concatenate(all_spectra, axis=0) if all_spectra else np.empty((0, 0), dtype=np.float32),
        all_provenance,
    )


def _random_blob_mask(rng: np.random.Generator, h: int, w: int, area_px: int,
                       *, forbidden: np.ndarray | None = None, max_steps: int = 2000) -> np.ndarray:
    """A single connected region of up to area_px pixels via a random walk,
    avoiding `forbidden` (already-occupied) pixels where possible. May return
    fewer than area_px pixels if growth stalls against the patch edge or
    forbidden region -- size_range_px is a range, not an exact target.
    """
    mask = np.zeros((h, w), dtype=bool)
    seed_r = seed_c = None
    for _ in range(50):
        r0, c0 = int(rng.integers(0, h)), int(rng.integers(0, w))
        if forbidden is None or not forbidden[r0, c0]:
            seed_r, seed_c = r0, c0
            break
    if seed_r is None:
        return mask   # patch is fully occupied; caller treats an empty mask as "skip"

    mask[seed_r, seed_c] = True
    frontier = [(seed_r, seed_c)]
    count = 1
    steps = 0
    deltas = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
    while count < area_px and frontier and steps < max_steps:
        steps += 1
        pick = int(rng.integers(0, len(frontier)))
        r, c = frontier[pick]
        dr, dc = deltas[int(rng.integers(0, 4))]
        nr, nc = r + dr, c + dc
        if (0 <= nr < h and 0 <= nc < w and not mask[nr, nc]
                and (forbidden is None or not forbidden[nr, nc])):
            mask[nr, nc] = True
            frontier.append((nr, nc))
            count += 1
        elif rng.random() < 0.15:
            frontier.pop(pick)   # dead end -- prune so the walk doesn't stall forever
    return mask


def implant_targets(background: np.ndarray, target_spectra: np.ndarray, *,
                     n_targets: int, abundance_range: tuple[float, float] = (0.1, 1.0),
                     size_range_px: tuple[int, int] = (1, 40), shape: str = "blob",
                     seed: int, spectrum_ids: list[str] | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Linear mixing model: m = a*t + (1-a)*s.
    t = target spectrum, s = the background pixel actually at that location,
    a = abundance fraction swept over abundance_range.

    a -> 0.1 gives faint sub-pixel targets; a -> 1.0 gives fully-visible ones.
    Sweeping a IS the difficulty control.

    Masks are exact and free: the implant location is CHOSEN, not annotated.
    Returns (cube, mask[C3], meta) where meta records per-target
    (abundance, spectrum_id, centroid, area_px) for the ablation curve.
    """
    if shape != "blob":
        raise ValueError(f"unsupported shape {shape!r}, only 'blob' is implemented")
    if target_spectra.ndim != 2 or target_spectra.shape[0] == 0:
        raise ValueError(f"target_spectra must be [K>=1, B], got shape {target_spectra.shape}")
    if target_spectra.shape[-1] != background.shape[-1]:
        raise ValueError(
            f"target_spectra band count {target_spectra.shape[-1]} != background "
            f"band count {background.shape[-1]} -- both must be on the same canonical grid")

    rng = np.random.default_rng(seed)
    h, w, _b = background.shape
    cube = background.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    occupied = np.zeros((h, w), dtype=bool)
    targets_meta = []

    for _ in range(n_targets):
        area_px = int(rng.integers(size_range_px[0], size_range_px[1] + 1))
        blob = _random_blob_mask(rng, h, w, area_px, forbidden=occupied)
        if not blob.any():
            continue

        spectrum_idx = int(rng.integers(0, target_spectra.shape[0]))
        t = target_spectra[spectrum_idx]
        a = float(rng.uniform(*abundance_range))

        rows, cols = np.where(blob)
        s = cube[rows, cols]
        mixed = a * t[None, :] + (1.0 - a) * s
        cube[rows, cols] = mixed.astype(cube.dtype)
        mask[blob] = 1
        occupied |= blob

        targets_meta.append(dict(
            abundance=a,
            spectrum_id=(spectrum_ids[spectrum_idx] if spectrum_ids is not None else spectrum_idx),
            centroid=(float(rows.mean()), float(cols.mean())),
            area_px=int(blob.sum()),
        ))

    meta = dict(n_targets_requested=n_targets, n_targets_placed=len(targets_meta),
                targets=targets_meta, seed=seed)
    return cube, mask, meta


_PERTURBATIONS = ("shift", "scale", "swap", "noise")


def pseudo_anomaly_patch(background: np.ndarray, *, n_regions: int,
                          perturbation: str = "mixed", seed: int,
                          swap_pool: np.ndarray | None = None,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Self-supervised pretext task: real patch vs. synthetic pseudo-anomaly
    patch. Trains on the background pool ONLY -- no target library, which is
    what makes it the honest zero-prior comparison arm.

    Region shape: random-walk blob (see implant_targets' shape="blob").
    Spectral perturbation, deliberately NOT a real target spectrum:
      "shift" : additive per-band offset drawn from the scene's own band std
      "scale" : multiplicative gain
      "swap"  : substitute a spectrum from elsewhere in the pool if
                swap_pool is given (a genuinely distant scene), else from a
                spatially distant pixel within this same patch
      "noise" : structured (band-correlated) noise injection
      "mixed" : sample one of the above per region
    """
    if perturbation != "mixed" and perturbation not in _PERTURBATIONS:
        raise ValueError(f"unknown perturbation {perturbation!r}")

    rng = np.random.default_rng(seed)
    h, w, b = background.shape
    cube = background.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    occupied = np.zeros((h, w), dtype=bool)
    band_std = np.nanstd(background, axis=(0, 1))
    band_std = np.where(np.isfinite(band_std) & (band_std > 0), band_std, 1.0)

    max_area = max(4, (h * w) // 8)
    for _ in range(n_regions):
        area_px = int(rng.integers(4, max_area + 1))
        blob = _random_blob_mask(rng, h, w, area_px, forbidden=occupied)
        if not blob.any():
            continue

        kind = _PERTURBATIONS[int(rng.integers(0, len(_PERTURBATIONS)))] \
            if perturbation == "mixed" else perturbation
        rows, cols = np.where(blob)
        vals = cube[rows, cols]

        if kind == "shift":
            new_vals = vals + rng.normal(scale=band_std, size=(1, b))
        elif kind == "scale":
            new_vals = vals * float(rng.uniform(0.5, 2.0))
        elif kind == "swap":
            if swap_pool is not None and len(swap_pool) > 0:
                donor = swap_pool[int(rng.integers(0, len(swap_pool)))]
                dr, dc = int(rng.integers(0, donor.shape[0])), int(rng.integers(0, donor.shape[1]))
                donor_spectrum = donor[dr, dc]
            else:
                dr, dc = int(rng.integers(0, h)), int(rng.integers(0, w))
                donor_spectrum = cube[dr, dc]
            new_vals = np.broadcast_to(donor_spectrum, vals.shape).copy()
        elif kind == "noise":
            noise = rng.normal(scale=band_std * 0.5, size=vals.shape)
            kernel = np.ones(5) / 5.0
            noise = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 1, noise)
            new_vals = vals + noise
        else:
            raise ValueError(f"unknown perturbation {kind!r}")

        cube[rows, cols] = new_vals.astype(cube.dtype)
        mask[blob] = 1
        occupied |= blob

    return cube, mask
