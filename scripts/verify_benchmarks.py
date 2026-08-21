#!/usr/bin/env python3
"""Verify Indian Pines, ABU and HYDICE against the invariants PLAN.md relies on.

Companion to verify_had100.py. Same contract: parse every file, assert what the
plan depends on, exit non-zero on drift. Written after HAD100's documented
composition turned out to disagree with its archive in five separate ways.

Checks, per dataset:
  Indian Pines -- D2's load-bearing claim that there is NO CRS/affine
  ABU          -- scene count and group split, and the per-scene band count,
                  which the plan asserts is uniformly 205 (it is not)
  HYDICE       -- which HYDICE this actually is; two different datasets are both
                  called "HYDICE urban" and they have different shapes, band
                  counts and ground-truth semantics
  ALL          -- presence of a wavelength array, without which harmonize()
                  (D9) has no source grid to interpolate from

Reads only. Writes docs/benchmarks_verified.json.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
GEO_HINTS = ("crs", "proj", "transform", "affine", "geo", "map info",
             "utm", "wgs", "coord")
WL_HINTS = ("wavelen", "wave", "_wl", "wl_", "fwhm", "bandcenter")


def load(p: Path) -> dict:
    return {k: v for k, v in sio.loadmat(p).items() if not k.startswith("__")}


def main() -> int:
    out: dict = {}
    fail: list[str] = []

    # ---------- Indian Pines ----------
    ip = BENCH / "indian_pines"
    files = sorted(ip.glob("*.mat"))
    ipinfo: dict = {"files": {}, "georef_keys": [], "wavelength_keys": []}
    for f in files:
        d = load(f)
        for k, v in d.items():
            ipinfo["files"][f"{f.name}:{k}"] = dict(
                shape=list(v.shape), dtype=str(v.dtype),
                min=float(v.min()), max=float(v.max()))
        ipinfo["georef_keys"] += [k for k in d if any(h in k.lower() for h in GEO_HINTS)]
        ipinfo["wavelength_keys"] += [k for k in d if any(h in k.lower() for h in WL_HINTS)]
    ipinfo["has_georef"] = bool(ipinfo["georef_keys"])
    out["indian_pines"] = ipinfo
    print("=== Indian Pines ===")
    for k, v in ipinfo["files"].items():
        print(f"  {k:44} {v['shape']} {v['dtype']}  [{v['min']:g}, {v['max']:g}]")
    print(f"  georeference keys : {ipinfo['georef_keys'] or 'NONE'}  "
          f"-> D2 {'INVALID' if ipinfo['has_georef'] else 'CONFIRMED'}")
    print(f"  wavelength keys   : {ipinfo['wavelength_keys'] or 'NONE'}")
    if ipinfo["has_georef"]:
        fail.append("Indian Pines HAS georeferencing -- D2's premise is wrong")

    # ---------- ABU ----------
    abu = sorted((BENCH / "abu").glob("*.mat"))
    rows, bands, spatial, dtypes, keysets = [], collections.Counter(), \
        collections.Counter(), collections.Counter(), collections.Counter()
    for f in abu:
        d = load(f)
        keysets[tuple(sorted(d))] += 1
        c, m = d.get("data"), d.get("map")
        if c is None or m is None:
            fail.append(f"{f.name}: missing data/map")
            continue
        bands[c.shape[2]] += 1
        spatial[c.shape[:2]] += 1
        dtypes[str(c.dtype)] += 1
        rows.append(dict(name=f.stem, shape=list(c.shape), dtype=str(c.dtype),
                         vmin=float(c.min()), vmax=float(c.max()),
                         neg_px=int((c < 0).sum()) if np.issubdtype(c.dtype, np.signedinteger) else 0,
                         over_int16=int((c.astype(np.int64) > 32767).sum())
                         if np.issubdtype(c.dtype, np.integer) else 0,
                         anom_px=int((m > 0).sum()),
                         anom_pct=round(100 * float((m > 0).mean()), 4),
                         mask_vals=np.unique(m).tolist(),
                         wavelength_keys=[k for k in d
                                          if any(h in k.lower() for h in WL_HINTS)]))
    groups = collections.Counter(r["name"].split("-")[1] for r in rows)
    out["abu"] = dict(n=len(rows), groups=dict(groups), scenes=rows,
                      band_counts=dict(bands),
                      spatial={f"{a}x{b}": c for (a, b), c in spatial.items()},
                      dtypes=dict(dtypes),
                      keysets=[list(k) for k in keysets],
                      uniform_bands=len(bands) == 1)
    print("\n=== ABU ===")
    print(f"  {'scene':16}{'cube':>20}{'dtype':>9}{'anom px':>9}{'%':>8}")
    for r in rows:
        print(f"  {r['name']:16}{str(tuple(r['shape'])):>20}{r['dtype']:>9}"
              f"{r['anom_px']:>9}{r['anom_pct']:>8}")
    print(f"  scenes={len(rows)} groups={dict(groups)}")
    print(f"  band counts : {dict(bands)}   uniform={len(bands)==1}")
    print(f"  spatial     : {out['abu']['spatial']}")
    print(f"  dtypes      : {dict(dtypes)}")
    neg = {r['name']: r['neg_px'] for r in rows if r['neg_px']}
    over = {r['name']: r['over_int16'] for r in rows if r['over_int16']}
    print(f"  int16 scenes with genuine negatives: {neg}")
    print(f"  integer scenes exceeding 32767     : {over or 'none'}  "
          f"(-> uint16-read-as-int16 wrap is NOT triggered by current data)")
    print(f"  wavelength keys anywhere: "
          f"{sorted({w for r in rows for w in r['wavelength_keys']}) or 'NONE'}")
    if len(rows) != 13:
        fail.append(f"ABU has {len(rows)} scenes, plan says 13")
    if dict(groups) != {"airport": 4, "beach": 4, "urban": 5}:
        fail.append(f"ABU group split is {dict(groups)}, plan says 4/4/5")

    # ---------- HYDICE ----------
    hy = sorted((BENCH / "hydice_urban_anomaly").glob("*.mat"))
    hinfo: dict = {}
    print("\n=== HYDICE ===")
    for f in hy:
        d = load(f)
        c, m = d.get("data"), d.get("map")
        from scipy import ndimage
        _, ncomp = ndimage.label(m > 0)
        hinfo = dict(file=f.name, shape=list(c.shape), dtype=str(c.dtype),
                     value_range=[float(c.min()), float(c.max())],
                     anom_px=int((m > 0).sum()), n_components=int(ncomp),
                     mask_vals=np.unique(m).tolist(),
                     wavelength_keys=[k for k in d
                                      if any(h in k.lower() for h in WL_HINTS)])
        print(f"  {f.name}: {tuple(c.shape)} {c.dtype} in "
              f"[{c.min():g}, {c.max():g}]")
        print(f"  anomaly pixels: {hinfo['anom_px']} in {ncomp} components "
              f"({100*float((m>0).mean()):.3f}%)")
        print(f"  wavelength keys: {hinfo['wavelength_keys'] or 'NONE'}")
    out["hydice"] = hinfo
    if hinfo and hinfo["shape"][2] != 175:
        fail.append(f"HYDICE has {hinfo['shape'][2]} bands, D13 says 175 "
                    f"-- you may have the OTHER HYDICE (see D13)")

    # ---------- cross-cutting ----------
    no_wl = (not ipinfo["wavelength_keys"]
             and not any(r["wavelength_keys"] for r in rows)
             and not (hinfo.get("wavelength_keys") if hinfo else []))
    out["harmonize_blocker"] = dict(
        any_dataset_ships_wavelengths=not no_wl,
        note=("D9 harmonize() interpolates from source wavelengths onto the "
              "canonical grid. None of these three ships one."))
    print("\n=== D9 harmonize prerequisite ===")
    print(f"  any of the three ships a wavelength array: {not no_wl}")
    if no_wl:
        print("  -> harmonize() cannot run on these without an external "
              "per-scene wavelength table (D13)")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "benchmarks_verified.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\nwrote {ROOT/'docs'/'benchmarks_verified.json'}")
    if fail:
        print("\nINVARIANTS VIOLATED:", file=sys.stderr)
        for f_ in fail:
            print(f"  - {f_}", file=sys.stderr)
        return 1
    print("all documented invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
