#!/usr/bin/env python3
"""PLAN.md §3A.9 -- fusion weight grid search, with the leakage split it requires.

D22.1 measured that §3A.9's stated default weights LOSE to the best single
component on 3 of the first 4 ABU scenes, so this sweep is load-bearing
rather than decorative: it is the only thing standing between the plan's
headline fusion claim and a table where fusion loses to its own best input.

THE LEAKAGE CONSTRAINT IS THE SHARP EDGE, and it is why this is a script
rather than three lines in the harness. §3A.9 says the weights are "tuned by
grid search on an ABU validation split; the tuning split is recorded and
NEVER reused for reporting." ABU is 13 scenes. Tuning over all 13 and then
reporting an AUC over the same 13 is straightforward train-on-test, and it
would produce a fusion result that beats every baseline for entirely
illegitimate reasons -- the most flattering possible number and the least
defensible.

So: a fixed, recorded, deterministic TUNE split is carved out here, the
weights are chosen on it alone, and the REPORT split is never touched during
selection. Both lists are written into the output JSON so the harness can
honour the separation without re-deriving it.

Components are computed ONCE per scene and cached; only the weighted
combination is swept. The expensive part (local_rx, ACE) does not depend on
the weights.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.fusion import fuse_scores                      # noqa: E402
from anomaly.local_rx import local_rx                       # noqa: E402
from anomaly.rx import global_rx                            # noqa: E402
from anomaly.scoring import (                               # noqa: E402
    ace_score,
    estimate_target_signature,
    spatial_context_score,
)

ABU = ROOT / "data" / "benchmark" / "abu"
OUT = ROOT / "experiments" / "rx_vs_ae" / "fusion_weights.json"

# Deterministic, recorded split. Chosen to put a spread of anomaly densities
# and both 100x100 and 150x150 shapes on each side, rather than by score --
# selecting the split by outcome would be a second, subtler leak.
TUNE_SCENES = ["abu-airport-1", "abu-airport-3", "abu-beach-1", "abu-urban-2", "abu-urban-5"]
REPORT_SCENES = ["abu-airport-2", "abu-airport-4", "abu-beach-2", "abu-beach-3",
                 "abu-beach-4", "abu-urban-1", "abu-urban-3", "abu-urban-4"]

# ABU ships no wavelengths (D13.4), so the `index` component cannot run here
# and fusion is 3-component (D20). Grid over the three that exist.
GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def components_for(name: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    m = sio.loadmat(ABU / f"{name}.mat")
    cube = m["data"].astype(np.float32)
    gt = m["map"].astype(bool)
    base = global_rx(cube)
    sig = estimate_target_signature(cube, base, top_frac=0.001)
    comps = {
        "rx": local_rx(cube, outer=15, inner=3, n_components=12),
        "ace": ace_score(cube, sig),
        "spatial": spatial_context_score(base, k=7),
    }
    return comps, gt


def auc(score: np.ndarray, gt: np.ndarray) -> float:
    v = ~np.isnan(score)
    return float(roc_auc_score(gt[v], score[v]))


def main() -> int:
    t0 = time.time()
    cache: dict[str, tuple[dict, np.ndarray]] = {}
    for name in TUNE_SCENES + REPORT_SCENES:
        cache[name] = components_for(name)
        print(f"  components: {name}", flush=True)
    print(f"components computed in {time.time() - t0:.1f}s\n", flush=True)

    singles = {n: {k: auc(v, cache[n][1]) for k, v in cache[n][0].items()} for n in cache}

    # --- grid search on the TUNE split only ---------------------------------
    results = []
    for w_rx, w_ace, w_sp in itertools.product(GRID, repeat=3):
        total = w_rx + w_ace + w_sp
        if total == 0:
            continue
        weights = {"rx": w_rx / total, "ace": w_ace / total, "spatial": w_sp / total}
        aucs = [auc(fuse_scores(cache[n][0], weights).score, cache[n][1]) for n in TUNE_SCENES]
        results.append((float(np.mean(aucs)), weights))

    results.sort(key=lambda r: -r[0])
    best_macro, best_weights = results[0]
    print(f"grid: {len(results)} weightings evaluated on {len(TUNE_SCENES)} tune scenes")
    print(f"best tune scene-macro AUC {best_macro:.4f} at "
          f"{ {k: round(v,4) for k,v in best_weights.items()} }\n")

    # --- apply to the untouched REPORT split --------------------------------
    default = {"rx": 0.40 / 0.85, "ace": 0.25 / 0.85, "spatial": 0.20 / 0.85}
    print(f"{'scene':16s} {'best_single':>11s} {'default_w':>10s} {'tuned_w':>9s} {'tuned>=best?':>13s}")
    wins = 0
    report_rows = []
    for n in REPORT_SCENES:
        comps, gt = cache[n]
        b = max(singles[n].values())
        d = auc(fuse_scores(comps, default).score, gt)
        f = auc(fuse_scores(comps, best_weights).score, gt)
        ok = f >= b
        wins += ok
        report_rows.append(dict(scene=n, best_single=b, default_fused=d, tuned_fused=f,
                                tuned_beats_best_single=bool(ok), singles=singles[n]))
        print(f"{n:16s} {b:11.4f} {d:10.4f} {f:9.4f} {str(ok):>13s}")

    macro_default = float(np.mean([r["default_fused"] for r in report_rows]))
    macro_tuned = float(np.mean([r["tuned_fused"] for r in report_rows]))
    macro_best_single = float(np.mean([r["best_single"] for r in report_rows]))
    print(f"\nREPORT split scene-macro: best_single {macro_best_single:.4f} | "
          f"default {macro_default:.4f} | tuned {macro_tuned:.4f}")
    print(f"tuned >= best single on {wins}/{len(REPORT_SCENES)} report scenes")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        note=("Weights selected on TUNE scenes ONLY. REPORT scenes were never used for "
              "selection and are the only ones any published fused number may come from "
              "(PLAN.md 3A.9, D22.1)."),
        components=["rx", "ace", "spatial"],
        index_component_absent_reason="ABU ships no wavelength array (D13.4/D20)",
        tune_scenes=TUNE_SCENES, report_scenes=REPORT_SCENES,
        grid_values=GRID, n_weightings=len(results),
        default_weights=default, tuned_weights=best_weights,
        tune_macro_auc=best_macro,
        report_macro=dict(best_single=macro_best_single, default_fused=macro_default,
                          tuned_fused=macro_tuned),
        report_rows=report_rows,
        top_10_on_tune=[dict(macro_auc=a, weights=w) for a, w in results[:10]],
    ), indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
