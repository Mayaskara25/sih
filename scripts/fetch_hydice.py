#!/usr/bin/env python3
"""Fetch the HYDICE *anomaly* scene, pinned by SHA256.

WHY THIS FILE IS PINNED AND THE OTHERS ARE NOT YET:
Two different, incompatible datasets are commonly called "HYDICE urban".

  THIS ONE (Michigan anomaly scene)     -- 80 x 100 x 175 float64 in [0,1],
      binary anomaly mask, 21 target pixels in 10 connected components.
      Correct for anomaly detection. What this project uses.

  THE OTHER (Copperas Cove, TX)         -- 307 x 307, 210 -> 162 bands,
      ground truth is SIX-ENDMEMBER ABUNDANCE MAPS, not an anomaly mask.
      It is an unmixing benchmark. It CANNOT be scored against pixel masks
      and MUST NOT be substituted here. See PLAN.md D13.3.

They are impossible to tell apart by filename -- the file we want is itself
named "HYDICE-urban.mat". The checksum is the only reliable discriminator, so
it is committed rather than fetched.
"""
from __future__ import annotations
import hashlib, sys, urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "data" / "benchmark" / "hydice_urban_anomaly"
FILES = {
    "HYDICE-urban.mat": (
        "https://raw.githubusercontent.com/sxt1996/HYDICE/main/HYDICE-urban.mat",
        "a998766a7180bcacaf5d2163d57857726d80b42f64b490083de85f072b593f4b",
        2544597,
    ),
    "README.md": (  # provenance: states the Michigan origin and the 175-band count
        "https://raw.githubusercontent.com/sxt1996/HYDICE/main/README.md",
        "ffe0592be4ce8b2d8520220b1f6fb20e858c55a2209b9ef56273263b0f49d39f",
        1201,
    ),
}
# Post-download shape assertion. A checksum catches a corrupted or swapped file;
# this catches the case where upstream legitimately republishes a DIFFERENT
# HYDICE under the same name and we would otherwise just see a hash mismatch
# with no idea why.
EXPECT_SHAPE = (80, 100, 175)
EXPECT_ANOM_PX = 21


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (url, want, size) in FILES.items():
        out = DEST / name
        if out.exists() and sha256(out) == want:
            print(f"  {name}: present, sha256 OK")
            continue
        print(f"  {name}: downloading {url}")
        urllib.request.urlretrieve(url, out)
        got = sha256(out)
        if got != want:
            print(f"SHA256 MISMATCH for {name}\n  expected {want}\n  got      {got}\n"
                  f"Do NOT use this file. It may be the Copperas Cove unmixing variant "
                  f"(D13.3) or a re-encoding. Investigate before overriding.",
                  file=sys.stderr)
            return 1
        if out.stat().st_size != size:
            print(f"size mismatch for {name}", file=sys.stderr)
            return 1
        print(f"  {name}: {out.stat().st_size} B, sha256 OK")

    import scipy.io as sio
    import numpy as np
    d = sio.loadmat(DEST / "HYDICE-urban.mat")
    cube, mask = d["data"], d["map"]
    if cube.shape != EXPECT_SHAPE:
        print(f"shape {cube.shape} != {EXPECT_SHAPE} -- this is not the Michigan "
              f"anomaly scene (D13.3)", file=sys.stderr)
        return 1
    if int((mask > 0).sum()) != EXPECT_ANOM_PX:
        print(f"anomaly pixels {(mask>0).sum()} != {EXPECT_ANOM_PX}", file=sys.stderr)
        return 1
    print(f"verified: {cube.shape} {cube.dtype}, {EXPECT_ANOM_PX} anomaly px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
