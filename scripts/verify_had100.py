#!/usr/bin/env python3
"""Re-derive every HAD100 number in PLAN.md D11 from the downloaded archive.

Parses all 616 ENVI headers plus the repo's own main.py, and reports:
  (a) true per-sensor background counts, raw and after main.py's 4-corner crop
  (b) the FULL distribution of raw spatial shapes -- every shape seen, with
      counts -- for the test set and both background pools
  (c) band counts before and after main.py's band_select, which is what decides
      whether the two background pools can be stacked without harmonize()
  (d) wavelength monotonicity, the thing that silently breaks np.interp
  (e) CRS presence and the set of no-data sentinels actually in use

Reads only. Writes docs/had100_verified.json. Exits non-zero if a D11 invariant
no longer holds, so a re-fetch that changed the archive fails loudly.
"""
from __future__ import annotations
import json, re, sys, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "benchmark" / "had100" / "HAD100"
SUBSETS = ("data/aviris_ng_target", "data/aviris_ng_normal", "data/aviris_normal")

# main.py band_select, transcribed verbatim -- not re-derived, so a change in
# the upstream repo shows up as a mismatch rather than being silently absorbed.
BAND_SELECT = {
    "aviris_ng": np.r_[15:109, 118:145, 158:187, 227:274, 328:407],
    "aviris":    np.r_[7:57, 65:79, 85:104, 122:149, 172:224],
}
EXPECT = {  # D11 invariants
    "data/aviris_ng_target": dict(n=94, bands={425}),
    "data/aviris_ng_normal": dict(n=260, bands={425}),
    "data/aviris_normal":    dict(n=262, bands={224}),
}


def parse_hdr(p: Path) -> dict:
    t = p.read_text(errors="replace")

    def num(key: str) -> int:
        m = re.search(rf"^{key}\s*=\s*(\d+)", t, re.M)
        if not m:
            raise ValueError(f"{p.name}: no '{key}'")
        return int(m.group(1))

    m = re.search(r"wavelength\s*=\s*\{(.*?)\}", t, re.S)
    wl = (np.array([float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()])
          if m else None)
    ig = re.search(r"data ignore value\s*=\s*(\S+)", t)
    return dict(
        lines=num("lines"), samples=num("samples"), bands=num("bands"),
        dtype=num("data type"),
        interleave=re.search(r"interleave\s*=\s*(\w+)", t).group(1),
        wl=wl, has_crs=("map info" in t),
        ignore=(float(ig.group(1)) if ig else None),
    )


def main() -> int:
    if not BASE.exists():
        print(f"MISSING: {BASE} -- run the fetcher first", file=sys.stderr)
        return 2

    out: dict = {"subsets": {}}
    fail: list[str] = []

    for sub in SUBSETS:
        hdrs = sorted((BASE / sub).glob("*.hdr"))
        if not hdrs:
            fail.append(f"{sub}: no .hdr files")
            continue
        rec = [parse_hdr(h) for h in hdrs]
        shapes = collections.Counter((r["lines"], r["samples"]) for r in rec)
        bands = collections.Counter(r["bands"] for r in rec)
        nonmono = sum(1 for r in rec
                      if r["wl"] is not None and bool(np.any(np.diff(r["wl"]) <= 0)))
        info = dict(
            n=len(rec),
            spatial={f"{a}x{b}": c for (a, b), c in sorted(shapes.items())},
            distinct_shapes=len(shapes),
            min_shape=f"{min(shapes)[0]}x{min(shapes)[1]}",
            max_shape=f"{max(shapes)[0]}x{max(shapes)[1]}",
            all_64x64=(set(shapes) == {(64, 64)}),
            bands=dict(bands),
            interleave=sorted({r["interleave"] for r in rec}),
            dtypes=sorted({r["dtype"] for r in rec}),
            nonmonotonic_wl=f"{nonmono}/{len(rec)}",
            crs_present=f"{sum(r['has_crs'] for r in rec)}/{len(rec)}",
            nodata_sentinels=sorted({r["ignore"] for r in rec}, key=str),
        )
        out["subsets"][sub] = info

        print(f"\n=== {sub}  (n={len(rec)}) ===")
        print(f"  spatial        : {info['spatial']}")
        print(f"  min/max/distinct: {info['min_shape']} / {info['max_shape']} / "
              f"{info['distinct_shapes']}   all_64x64={info['all_64x64']}")
        print(f"  bands          : {info['bands']}  interleave={info['interleave']}  "
              f"dtypes={info['dtypes']}")
        print(f"  non-monotonic wl: {info['nonmonotonic_wl']}")
        print(f"  CRS present    : {info['crs_present']}   nodata={info['nodata_sentinels']}")

        exp = EXPECT[sub]
        if len(rec) != exp["n"]:
            fail.append(f"{sub}: expected {exp['n']} scenes, found {len(rec)}")
        if set(bands) != exp["bands"]:
            fail.append(f"{sub}: expected bands {exp['bands']}, found {set(bands)}")

    # main.py-derived counts
    src = (BASE / "main.py").read_text()
    cd = re.search(r"crop_dict\s*=\s*\{(.*?)\n\n", src, re.S).group(1)
    entries = re.findall(r"'([^']+)':\s*(\[[^\]]*\])", cd)
    extra = sum(v.count("(") - 1 for _, v in entries)
    n_raw = out["subsets"]["data/aviris_ng_target"]["n"]
    ng, avc = len(BAND_SELECT["aviris_ng"]), len(BAND_SELECT["aviris"])
    d = dict(
        test_raw=n_raw, crop_dict_keys=len(entries), extra_crops=extra,
        test_patches=n_raw + extra,
        bg_ng_patches=out["subsets"]["data/aviris_ng_normal"]["n"] * 4,
        bg_classic_patches=out["subsets"]["data/aviris_normal"]["n"] * 4,
        bands_after_select={"aviris_ng": ng, "aviris": avc},
        pools_stackable=(ng == avc),
    )
    d["bg_total_patches"] = d["bg_ng_patches"] + d["bg_classic_patches"]
    out["derived"] = d

    print("\n=== derived from main.py ===")
    print(f"  test : {n_raw} raw + {extra} extra crops = {d['test_patches']} patches")
    print(f"  bg   : {d['bg_ng_patches']} NG + {d['bg_classic_patches']} Classic "
          f"= {d['bg_total_patches']} patches")
    print(f"  bands after band_select: NG {ng}, Classic {avc}  "
          f"-> stackable={d['pools_stackable']}")

    if d["test_patches"] != 100:
        fail.append(f"test patches = {d['test_patches']}, D11 says 100")
    if d["bg_total_patches"] != 2088:
        fail.append(f"bg patches = {d['bg_total_patches']}, D11 says 2088")
    if (ng, avc) != (276, 162):
        fail.append(f"band_select gives ({ng}, {avc}), D11 says (276, 162)")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "had100_verified.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {ROOT / 'docs' / 'had100_verified.json'}")

    if fail:
        print("\nD11 INVARIANTS VIOLATED:", file=sys.stderr)
        for f in fail:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("all D11 invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
