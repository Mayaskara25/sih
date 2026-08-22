"""PLAN.md 3E.6 -- the classical-vs-quantum comparison runner, and the
branch's actual deliverable (D27: "the comparison table is the deliverable,
not an advantage claim", Roadmap Section 1.10/Section 9.6). Loops eight rows
(seven arms; the quantum kernel runs twice, D28's rewrite) over the frozen
`quantum.data.build_split` and writes `experiments/quantum_results/
{results.csv, results_pooled.csv, run_manifest.json}`.

WHY EIGHT ROWS, NOT SIX (D27.3, D28 -- rewritten 2026-08-22 after this module
was started; read both before touching this file). The original brief named
five unsupervised arms plus one supervised one, ranked in a single column --
exactly Section 3E.2's own warning ("comparing on unequal ground measures the
ground, not the model") one level up: supervision. Fixed by a `supervision`
column and never emitting one pooled ranking across it (see `_pool` /
`main`). D28's rewrite then found a SECOND unequal-ground error, one level
down: the quantum kernel arm's angle-encoding scale is itself a
val-selected hyperparameter (sweeping it moves test AUC from 0.378 to
0.693 -- the single largest effect measured in this branch), and comparing a
val-tuned quantum arm against an untuned classical one measures tuning
budget, not quantumness. Fixed the same way twice over: every arm that is
CHEAP to sweep (classical_ae, classical_ocsvm, classical_svc, rx_8feat, and
the quantum kernel's angle_scale) is swept on VAL ONLY, never test, and a
`swept` column says which ones did. VQC and QAE are NOT swept -- a single
COBYLA fit is ~35 min (VQC) / ~15 min (QAE) at the frozen full-run defaults
(D27.0 finding 6, `quantum/quantum_autoencoder.py`'s own measurement), so a
grid of those is not affordable in this branch's compute budget, and
presenting a swept arm beside an unswept one as directly comparable would be
exactly the mistake this rewrite exists to prevent -- `swept=False` on those
rows says so explicitly rather than by omission.

THE KERNEL ANGLE-SCALE WRAPPER LIVES HERE, NOT IN quantum/quantum_kernel.py
OR quantum/data.py (explicit instruction, D28's rewrite). `QuantumKernelArm
.fit(split)` reads only `split.X_train`/`split.y_train` (verified from
source, quantum/quantum_kernel.py); `.score(X)` takes a bare array. So
`_ScaledQuantumKernelArm` below hands `fit` a `dataclasses.replace()`d
QuantumSplit with ONLY `X_train` rescaled (every other field -- X_val,
X_test, transformer, seed -- passed through unchanged, since `fit` never
reads them), and rescales its own `score(X)` argument directly before
delegating. This is sound because angle-encoding rotation gates are
2*pi-periodic (quantum/feature_map.py's own `QuantumFeatureTransformer`
docstring makes the same point about clipping vs. aliasing): a rescaled angle
is a different, well-defined point on the Bloch sphere, not an error. Two
labelled rows result: `quantum_kernel_spec` (angle_scale=1.0, `zz reps=2`
exactly as Section 3E.5 specifies) and `quantum_kernel_valscale` (the
val-selected scale from {1.0, 0.5, 0.25, 0.125} -- D28's rewrite measured val
selects 0.5, test 0.693 there against 0.378 at the spec point). Never
silently reporting only the better one -- D28's original framing (kept
below where it still applies): selecting the best configuration and
reporting one number would hide the finding, and the finding (how much a
"quantum kernel" number moves under an unequal-ground comparison) is more
valuable than either number alone.

D27.7 -- METRICS AT NATURAL PREVALENCE, NOT ON THE BALANCED FIT/CALIBRATION
SPLIT. Every reported ROC-AUC/PR-AUC comes from
`quantum.data.score_scene_natural`, looped over `quantum.data.test_scene_ids()`
(28 scenes in the real run), with `sample_weight` passed to
`roc_auc_score`/`average_precision_score`. `split.X_test`/`y_test` (the
capped, roughly-balanced population `build_split` also returns) is used
ONLY for the arms' own val-analogous bookkeeping never -- see
`quantum.data.build_split`'s own docstring warning about the "two different
test populations" trap; this module never calls `roc_auc_score` on
`split.X_test`/`y_test` at all, so that trap cannot reoccur here by a stray
line.

F1 THRESHOLD IS CALIBRATED ON VAL, PER ARM, ONCE -- `anomaly.scoring
.calibrate_threshold_for_recall(arm.score(split.X_val), split.y_val,
target_recall=...)`, never touching test. Every test-scene F1/precision/
recall in `results.csv` reuses that one threshold.

SCENE-MACRO IS PRIMARY (`scripts/run_benchmark.py`'s own convention, Section
3A.10). `_pool` below is a DELIBERATE REIMPLEMENTATION of
`scripts/run_benchmark.py::pool`'s macro/micro labelling, not an import of
it, for one reason: `pool()` silently drops a (dataset, detector) group with
zero `status=="ok"` rows from the pooled table entirely, and D27.7's
"kept, never dropped" rule (which `run_benchmark.py` already states for its
OWN per-scene rows) has to extend to the pooled level too here -- an arm that
failed on every test scene must still appear in `results_pooled.csv`, with
NaN metrics, not vanish from the comparison the reader is looking at.
`scripts/run_benchmark.py` itself is never imported or modified
(CONTRIBUTING.md: don't touch it).

D28 -- KERNEL-CONCENTRATION DIAGNOSTIC IN THE MANIFEST. For both quantum
kernel rows, `_kernel_concentration` recomputes the background Gram (via
`quantum.quantum_kernel.gram_statevector`, the fast exact path -- D27.0
finding 5) from the FITTED arm's own stored background angles and feature
map, and reports mean/p99 off-diagonal. This is what D28 used to diagnose
the exponential-concentration failure in the first place (mean off-diag
0.038 at the spec point, deepening with `reps`) and it is recomputed here,
not copied from the note, so a drift in feature-map defaults would show up
as a drifted diagnostic rather than a stale documented number.

'pauli' WITH DEFAULT PAULIS IS THE 'zz' MAP (D27.0's aside, restated because
it is exactly the kind of thing a runner sweeping `kind` might get wrong):
`pauli_feature_map`'s default `paulis=['Z','ZZ']` reproduces `zz_feature_map`
to the digit. This runner never sweeps `kind` for that reason -- there would
be only two independent circuits to sweep, not three, and reporting all
three as if independent would misrepresent the sweep.

Never a dependency of the operational pipeline (Roadmap Section 1.5, Section
9.10). PC + simulator only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.scoring import calibrate_threshold_for_recall             # noqa: E402
from quantum.classical_baselines import (                              # noqa: E402
    DenseAEArm,
    MahalanobisArm,
    OCSVMArm,
    SVCArm,
)
from quantum.data import QuantumSplit, build_split, score_scene_natural, test_scene_ids  # noqa: E402
from quantum.quantum_autoencoder import QuantumAutoencoderArm          # noqa: E402
from quantum.quantum_kernel import QuantumKernelArm, gram_statevector  # noqa: E402
from quantum.vqc_encoder import VQCArm                                 # noqa: E402

OUT_DIR = ROOT / "experiments" / "quantum_results"


# --------------------------------------------------------- kernel angle-scale wrap --

class _ScaledQuantumKernelArm:
    """Multiplies angle-encoded inputs by `angle_scale` before every
    fit/score call to an inner `QuantumKernelArm`. See module docstring for
    why this lives here and why it is sound (2*pi-periodicity of the
    rotation gates the feature map is built from).
    """

    supervision = "unsupervised"

    def __init__(self, *, kind: str = "zz", reps: int = 2, nu: float = 0.1,
                 angle_scale: float = 1.0, seed: int = 0) -> None:
        self.angle_scale = angle_scale
        self.inner = QuantumKernelArm(kind=kind, reps=reps, nu=nu, seed=seed)
        self.name = "quantum_kernel"
        self.fit_seconds: float | None = None

    def fit(self, split: QuantumSplit) -> None:
        """Hands the inner arm a QuantumSplit with ONLY X_train rescaled --
        `QuantumKernelArm.fit` reads nothing else (verified from source, see
        module docstring), so X_val/X_test/transformer/seed pass through
        untouched. `dataclasses.replace` on the frozen `QuantumSplit`
        produces a new object; the caller's own `split` is never mutated.
        """
        scaled = dataclasses.replace(
            split, X_train=np.asarray(split.X_train, dtype=np.float64) * self.angle_scale)
        self.inner.fit(scaled)
        self.fit_seconds = self.inner.fit_seconds

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.inner.score(np.asarray(X, dtype=np.float64) * self.angle_scale)

    def circuit_info(self) -> dict | None:
        return self.inner.circuit_info()


def _kernel_concentration(arm: _ScaledQuantumKernelArm) -> dict:
    """D28's diagnostic: mean/p99 off-diagonal of the FITTED background Gram,
    recomputed (not read back off the arm, which stores only the fitted
    OneClassSVM) via `gram_statevector` on `arm.inner.X_train` (the SCALED
    background angles the model was actually fit on) and `arm.inner
    .feature_map`. Requires `arm.fit` to have already run (`inner.X_train`
    is None otherwise -- raises the same RuntimeError `QuantumKernelArm
    .score` would).
    """
    if arm.inner.X_train is None:
        raise RuntimeError("_kernel_concentration: arm has not been fit yet")
    gram = gram_statevector(arm.inner.X_train, feature_map=arm.inner.feature_map)
    n = gram.shape[0]
    off = gram[~np.eye(n, dtype=bool)]
    return {
        "angle_scale": arm.angle_scale,
        "kind": arm.inner.kind,
        "reps": arm.inner.reps,
        "n_background": int(n),
        "mean_offdiag": float(off.mean()),
        "p99_offdiag": float(np.percentile(off, 99)),
    }


# ------------------------------------------------------------------------- rows --

@dataclass
class Row:
    """One (row_id, scene) observation. `status != "ok"` rows are KEPT, never
    dropped (D27.7 / `scripts/run_benchmark.py`'s own rule) -- an arm whose
    `fit` raised gets exactly one row with `scene="(all)"`; a scene whose
    `score_scene_natural` call raised gets its own row, the arm's other
    scenes are unaffected.
    """
    row_id: str
    arm: str
    supervision: str
    scene: str
    n_px_scored: int = 0
    n_anom: int = 0
    roc_auc: float | None = None
    pr_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    status: str = "ok"
    note: str = ""


@dataclass
class ArmInfo:
    """Arm-level (not per-scene) bookkeeping -- merged into
    `results_pooled.csv` and recorded in full in the manifest.

    `fit_wall_clock_s` is the SELECTED candidate's own `arm.fit_seconds`
    (D27.4's simulator-time column) -- NOT the time spent sweeping every
    candidate to find it. `sweep_wall_clock_s` is that total. Conflating the
    two would silently reintroduce the exact tuning-budget error the sweep
    itself exists to make visible (D28's rewrite, point 3): a 12-point
    OCSVM grid would report ~12x its winning fit's cost in the column a
    reader uses to compare per-arm cost against VQC's single unswept fit.
    """
    row_id: str
    arm: str
    supervision: str
    swept: bool
    params: dict
    sweep_table: list = field(default_factory=list)
    fit_wall_clock_s: float | None = None
    sweep_wall_clock_s: float | None = None
    wall_clock_kind: str = ""
    circuit_info: dict | None = None
    val_threshold: float | None = None
    val_threshold_fp_rate: float | None = None
    status: str = "ok"
    note: str = ""


# ---------------------------------------------------------------- val-only sweep --

def _select_on_val(build_fn, candidates: list[dict], split: QuantumSplit, seed: int,
                    *, extra_fn=None) -> tuple[object, dict, list[dict]]:
    """Fits `build_fn(seed=seed, **kwargs)` for every `kwargs` in
    `candidates`, scores `split.X_val`, and returns the arm with the highest
    val ROC-AUC -- NEVER touching `split.X_test`/`y_test` or any
    `score_scene_natural` output (D28's rewrite: selecting on test is the
    exact leak this repo's split discipline exists to prevent, restated one
    hyperparameter level down from the usual train/val/test split itself).

    A single-element `candidates` list degenerates to "fit once at these
    defaults, no selection" -- this is how VQC/QAE ('not swept', module
    docstring) and the quantum kernel's `quantum_kernel_spec` row reuse the
    exact same code path as the swept arms, with `swept = len(candidates) > 1`
    the only thing that differs downstream.

    `extra_fn`, when given, is called as `extra_fn(fitted_arm)` on EVERY
    candidate (not just the winner) and its return value merged into that
    candidate's `sweep_table` row under `"extra"`. This is how D28's kernel-
    concentration diagnostic gets recorded per swept angle_scale, not just
    for the selected one -- the whole point of the diagnostic is the
    scale -> concentration -> val AUC relationship, which is invisible if
    only the winning scale's Gram is ever measured.

    Returns (best_arm, best_params, sweep_table) -- `sweep_table` records
    EVERY candidate's params and val AUC (not just the winner), so the
    selection is auditable in the manifest, not just its result.
    """
    if not candidates:
        raise ValueError("_select_on_val: candidates must be non-empty")
    table: list[dict] = []
    best: tuple[object, float, dict] | None = None
    for kwargs in candidates:
        arm = build_fn(seed=seed, **kwargs)
        arm.fit(split)
        val_scores = np.asarray(arm.score(split.X_val))
        y_val = np.asarray(split.y_val)
        if val_scores.size and np.unique(y_val).size == 2:
            auc = float(roc_auc_score(y_val, val_scores))
        else:
            auc = float("nan")
        entry = {"params": kwargs, "val_auc": None if np.isnan(auc) else auc}
        if extra_fn is not None:
            entry["extra"] = extra_fn(arm)
        table.append(entry)
        if best is None:
            best = (arm, auc, kwargs)
        elif not np.isnan(auc) and (np.isnan(best[1]) or auc > best[1]):
            best = (arm, auc, kwargs)
    return best[0], best[2], table


# ------------------------------------------------------------------- slot builders --

def _build_ae(*, seed, **kw):
    return DenseAEArm(seed=seed, **kw)


def _build_ocsvm(*, seed, **kw):
    return OCSVMArm(seed=seed, **kw)


def _build_svc(*, seed, **kw):
    return SVCArm(seed=seed, **kw)


def _build_rx(*, seed, **kw):
    return MahalanobisArm(seed=seed, **kw)


def _build_vqc(*, seed, **kw):
    return VQCArm(seed=seed, **kw)


def _build_qae(*, seed, **kw):
    return QuantumAutoencoderArm(seed=seed, **kw)


def _build_kernel(*, seed, **kw):
    return _ScaledQuantumKernelArm(seed=seed, **kw)


def _slots(*, smoke: bool) -> list[dict]:
    """The eight rows (module docstring). Grids are tiny under `--smoke` --
    `--smoke` must finish well under a minute (brief), and a full grid times
    a COBYLA-trained arm's cost is exactly what makes the full run ~an hour,
    not a smoke run.
    """
    if smoke:
        ae_grid = [dict(n_latent=n, n_epochs=5) for n in (2, 4)]
        ocsvm_grid = [dict(gamma=g, nu=0.1) for g in ("scale", 1.0)]
        svc_grid = [dict(gamma="scale", C=1.0)]
        rx_grid = [dict(reg=r) for r in (1e-6, 1e-2)]
        kernel_scale_grid = [dict(kind="zz", reps=2, angle_scale=s) for s in (1.0, 0.5)]
        vqc_kwargs = dict(ansatz_reps=0, maxiter=2)
        qae_kwargs = dict(ansatz_reps=0, maxiter=2)
    else:
        ae_grid = [dict(n_latent=n, n_epochs=300) for n in (2, 4, 6)]
        ocsvm_grid = [dict(gamma=g, nu=nu) for g in ("scale", 0.1, 1.0, 10.0) for nu in (0.05, 0.1, 0.3)]
        svc_grid = [dict(gamma=g, C=c) for g in ("scale", 0.1, 1.0) for c in (0.1, 1.0, 10.0)]
        rx_grid = [dict(reg=r) for r in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
        kernel_scale_grid = [dict(kind="zz", reps=2, angle_scale=s) for s in (1.0, 0.5, 0.25, 0.125)]
        vqc_kwargs = dict(ansatz_reps=3, maxiter=200)
        qae_kwargs = dict(ansatz_reps=3, maxiter=150)

    # `supervision` is recorded STATICALLY per slot (known from each arm
    # class, module docstring's D27.3 table), not read off a fitted arm --
    # D27.3 requires "supervised"/"unsupervised" on EVERY row, including a
    # row whose `fit()` raised before any arm instance existed to ask.
    return [
        dict(row_id="classical_ae", build_fn=_build_ae, candidates=ae_grid,
             supervision="unsupervised"),
        dict(row_id="classical_ocsvm", build_fn=_build_ocsvm, candidates=ocsvm_grid,
             supervision="unsupervised"),
        dict(row_id="classical_svc", build_fn=_build_svc, candidates=svc_grid,
             supervision="supervised"),
        dict(row_id="rx_8feat", build_fn=_build_rx, candidates=rx_grid,
             supervision="unsupervised"),
        dict(row_id="vqc", build_fn=_build_vqc, candidates=[vqc_kwargs],
             supervision="supervised"),
        dict(row_id="qae", build_fn=_build_qae, candidates=[qae_kwargs],
             supervision="unsupervised"),
        dict(row_id="quantum_kernel_spec", build_fn=_build_kernel,
             candidates=[dict(kind="zz", reps=2, angle_scale=1.0)],
             supervision="unsupervised", extra_fn=_kernel_concentration),
        dict(row_id="quantum_kernel_valscale", build_fn=_build_kernel,
             candidates=kernel_scale_grid, supervision="unsupervised",
             extra_fn=_kernel_concentration),
    ]


_VALID_SUPERVISION = ("supervised", "unsupervised")


def _validate_supervision(supervision: str, row_id: str) -> None:
    """D27.3's `supervision` column is the fix for "ranking six arms in one
    column measures supervision, not quantumness" -- a missing/empty value
    would let a row slip past that fix silently (an empty string still
    groups and pools without error). Checked at the point each arm's
    `.supervision` is first read (right after a successful `_select_on_val`),
    not deferred to a downstream table where a blank label just looks like
    ordinary data.
    """
    if not supervision or supervision not in _VALID_SUPERVISION:
        raise ValueError(
            f"{row_id}: arm.supervision must be one of {_VALID_SUPERVISION}, "
            f"got {supervision!r} -- refusing to add an unlabelled row to a "
            "supervision-mixed comparison (D27.3)")


def _wall_clock_kind(row_id: str) -> str:
    """D27.4: the wall-clock column must be labelled as simulator time, and
    must carry no QPU implication. VQC/QAE run on shot-based/exact
    statevector simulation respectively; the quantum kernel rows run on
    `gram_statevector`'s exact statevector path (D27.0 finding 5); the
    classical arms are plain CPU wall-clock with no simulator involved at
    all -- conflating the two would misrepresent D27.4's own finding that an
    8-qubit AerSimulator/statevector run is a laptop-reproducible number, not
    evidence about a QPU.
    """
    if row_id in ("vqc",):
        return "aer_statevector_sampler_cpu"
    if row_id in ("qae", "quantum_kernel_spec", "quantum_kernel_valscale"):
        return "exact_statevector_cpu"
    return "classical_cpu"


# --------------------------------------------------------------- natural-prevalence --

def _score_test_scenes(arm, row_id: str, arm_name: str, supervision: str,
                        scenes: list[str], split: QuantumSplit, *, seed: int,
                        max_bg_per_scene: int, threshold: float | None) -> list[Row]:
    """One Row per (row_id, scene), via `quantum.data.score_scene_natural`
    (D27.7: natural-prevalence, `sample_weight`-corrected). `threshold` (val-
    calibrated, may be None if calibration failed) drives precision/recall/f1
    via WEIGHTED confusion counts -- `sample_weight` is not just for the two
    sklearn metrics; a plain unweighted TP/FP/FN count on the anomaly-
    enriched subsample would suffer the identical prevalence-distortion
    D27.7 measured for PR-AUC, so the same weights are reused here.
    """
    rows: list[Row] = []
    for scene_id in scenes:
        row = Row(row_id=row_id, arm=arm_name, supervision=supervision, scene=scene_id)
        try:
            scores, labels, weight = score_scene_natural(
                arm, scene_id, split, max_bg_per_scene=max_bg_per_scene, seed=seed)
        except Exception as exc:                                    # noqa: BLE001
            row.status = f"FAILED:{type(exc).__name__}"
            row.note = str(exc)[:200]
            rows.append(row)
            continue

        row.n_px_scored = int(labels.size)
        row.n_anom = int(labels.sum())
        y = labels.astype(bool)
        if y.sum() == 0 or y.all():
            row.status = "degenerate_labels"
            row.note = f"{int(y.sum())} positives of {y.size} scored px"
            rows.append(row)
            continue

        row.roc_auc = float(roc_auc_score(y, scores, sample_weight=weight))
        row.pr_auc = float(average_precision_score(y, scores, sample_weight=weight))

        if threshold is not None:
            pred = scores >= threshold
            tp = float(weight[pred & y].sum())
            fp = float(weight[pred & ~y].sum())
            fn = float(weight[~pred & y].sum())
            row.precision = tp / (tp + fp) if (tp + fp) else 0.0
            row.recall = tp / (tp + fn) if (tp + fn) else 0.0
            row.f1 = (2 * row.precision * row.recall / (row.precision + row.recall)
                      if (row.precision + row.recall) else 0.0)
        else:
            row.note = (row.note + "; " if row.note else "") + "no val threshold (F1 unavailable)"
        rows.append(row)
    return rows


# ----------------------------------------------------------------------- pooling --

def _pool(df: pd.DataFrame, all_row_ids: list[str]) -> pd.DataFrame:
    """Scene-macro (PRIMARY, each scene equal weight) and pixel-micro
    (SECONDARY, anomaly-pixel-count weighted), both explicitly labelled --
    `scripts/run_benchmark.py::pool`'s own convention, reimplemented here
    rather than imported (see module docstring for why: `pool()` drops a
    group with zero ok rows; this must not).
    """
    out = []
    for row_id in all_row_ids:
        g_all = df[df.row_id == row_id]
        g_ok = g_all[g_all.status == "ok"]
        rec = dict(row_id=row_id, n_scenes=int(len(g_ok)), n_failed=int((g_all.status != "ok").sum()))
        for m in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
            if len(g_ok) == 0:
                rec[f"{m}_MACRO"] = float("nan")
                rec[f"{m}_micro"] = float("nan")
                continue
            v = g_ok[m].to_numpy(dtype=float)
            w = g_ok.n_anom.to_numpy(dtype=float)
            rec[f"{m}_MACRO"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            rec[f"{m}_micro"] = (float(np.average(v[np.isfinite(v)], weights=w[np.isfinite(v)]))
                                  if np.isfinite(v).any() and w[np.isfinite(v)].sum() else float("nan"))
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------ manifest --

def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=ROOT)
        return out.stdout.strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md
    names = ["numpy", "scipy", "pandas", "scikit-learn", "torch",
              "qiskit", "qiskit-aer", "qiskit-machine-learning"]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            pass
    return out


# ----------------------------------------------------------------------------- main --

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-features", type=int, default=8)
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--smoke", action="store_true",
                     help="tiny split/grids/maxiter/scene-count -- finishes well under a minute")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Small on every axis, not just limit_scenes: build_split's OWN
        # per-scene/per-split caps (n_bg_per_scene/max_anom_per_scene/
        # max_train_total) default to values sized for the full run
        # (up to 600 train rows) regardless of limit_scenes, and VQC/QAE's
        # per-COBYLA-eval cost scales with N (D27.0 finding 6) -- a smoke
        # split built from limit_scenes alone still had ~500 train rows,
        # which is why an earlier version of this runner's --smoke took
        # minutes, not seconds. All four caps are shrunk together here.
        split = build_split(n_features=args.n_features, seed=args.seed, limit_scenes=2,
                             n_bg_per_scene=5, max_anom_per_scene=5,
                             max_train_total=30, max_test_total=30)
    else:
        split = build_split(n_features=args.n_features, seed=args.seed, limit_scenes=None)
    all_scenes = test_scene_ids()
    scenes = all_scenes[:2] if args.smoke else all_scenes
    max_bg_per_scene = 20 if args.smoke else 400

    print(f"split: train {split.X_train.shape} val {split.X_val.shape} "
          f"(scoring {len(scenes)} of {len(all_scenes)} test scenes, smoke={args.smoke})",
          flush=True)

    all_rows: list[Row] = []
    arm_infos: dict[str, ArmInfo] = {}
    kernel_concentration: dict[str, dict] = {}
    all_row_ids = [slot["row_id"] for slot in _slots(smoke=args.smoke)]

    for slot in _slots(smoke=args.smoke):
        row_id = slot["row_id"]
        t0 = time.perf_counter()
        try:
            arm, best_params, sweep_table = _select_on_val(
                slot["build_fn"], slot["candidates"], split, args.seed,
                extra_fn=slot.get("extra_fn"))
            _validate_supervision(arm.supervision, row_id)
        except Exception as exc:                                     # noqa: BLE001
            # `slot["supervision"]` (static, module docstring's D27.3 table)
            # is used here rather than an "unknown" placeholder: D27.3 wants
            # "supervised"/"unsupervised" on EVERY row, and a fit failure is
            # exactly the case where no fitted arm exists to read it off.
            note = f"{type(exc).__name__}: {str(exc)[:180]}"
            print(f"  {row_id:26s} FIT FAILED: {note}", flush=True)
            all_rows.append(Row(row_id=row_id, arm=row_id, supervision=slot["supervision"],
                                 scene="(all)", status=f"FAILED:{type(exc).__name__}", note=note))
            arm_infos[row_id] = ArmInfo(row_id=row_id, arm=row_id, supervision=slot["supervision"],
                                         swept=len(slot["candidates"]) > 1, params={},
                                         status=f"FAILED:{type(exc).__name__}", note=note)
            continue

        sweep_s = time.perf_counter() - t0
        # The SELECTED candidate's own fit time, not the sweep total above
        # -- see ArmInfo's docstring. Falls back to sweep_s only if an arm
        # class doesn't expose fit_seconds (none of this branch's arms
        # omit it, but a bare fallback beats an AttributeError here).
        fit_s = getattr(arm, "fit_seconds", None)
        if fit_s is None:
            fit_s = sweep_s
        swept = len(slot["candidates"]) > 1

        try:
            val_scores = np.asarray(arm.score(split.X_val))
            thr, fp_rate = calibrate_threshold_for_recall(
                val_scores, split.y_val, target_recall=args.target_recall)
            thr, fp_rate = float(thr), float(fp_rate)   # np.float64 -> float (JSON-clean manifest)
        except Exception as exc:                                     # noqa: BLE001
            thr, fp_rate = None, None
            thr_note = f"val threshold calibration failed: {type(exc).__name__}: {str(exc)[:150]}"
        else:
            thr_note = ""

        info = ArmInfo(row_id=row_id, arm=arm.name, supervision=arm.supervision, swept=swept,
                        params=best_params, sweep_table=sweep_table,
                        fit_wall_clock_s=float(fit_s), sweep_wall_clock_s=float(sweep_s),
                        wall_clock_kind=_wall_clock_kind(row_id),
                        circuit_info=arm.circuit_info(), val_threshold=thr,
                        val_threshold_fp_rate=fp_rate, note=thr_note)
        arm_infos[row_id] = info

        if row_id in ("quantum_kernel_spec", "quantum_kernel_valscale"):
            # Selected-candidate diagnostic, kept at the top level for quick
            # access; `info.sweep_table`'s own `"extra"` entries (populated
            # via _select_on_val's extra_fn=_kernel_concentration above) carry
            # the SAME diagnostic for every swept angle_scale, not just this
            # one -- D28's point is the scale -> concentration -> val AUC
            # relationship, which only the full sweep_table shows.
            kernel_concentration[row_id] = _kernel_concentration(arm)

        scene_rows = _score_test_scenes(arm, row_id, arm.name, arm.supervision, scenes, split,
                                         seed=args.seed, max_bg_per_scene=max_bg_per_scene,
                                         threshold=thr)
        all_rows.extend(scene_rows)

        n_ok = sum(1 for r in scene_rows if r.status == "ok")
        macro_auc = np.nanmean([r.roc_auc for r in scene_rows if r.roc_auc is not None]) \
            if n_ok else float("nan")
        print(f"  {row_id:26s} swept={swept!s:5s} fit {fit_s:6.2f}s (sweep total {sweep_s:6.2f}s)  "
              f"scenes ok {n_ok}/{len(scene_rows)}  scene-macro roc_auc {macro_auc:.4f}", flush=True)

    # D28's kernel-concentration diagnostic set, per the coordinator's
    # explicit follow-up: `zz reps=1` is recorded here as a DIAGNOSTIC-ONLY
    # Gram measurement (never scored on test scenes, never a `results.csv`
    # row) -- the original brief asked for it as a scored row, the rewrite's
    # "two labelled rows" instruction narrowed the SCORED table to
    # `quantum_kernel_spec`/`quantum_kernel_valscale`, and this preserves
    # the reps -> concentration relationship the original D28 note used to
    # diagnose the failure without reintroducing a third scored kernel row.
    try:
        reps1_arm = _build_kernel(seed=args.seed, kind="zz", reps=1, angle_scale=1.0)
        reps1_arm.fit(split)
        kernel_concentration["zz_reps1_scale1.0_diagnostic_only"] = _kernel_concentration(reps1_arm)
    except Exception as exc:                                          # noqa: BLE001
        kernel_concentration["zz_reps1_scale1.0_diagnostic_only"] = {
            "status": f"FAILED:{type(exc).__name__}", "note": str(exc)[:180]}

    df = pd.DataFrame([vars(r) for r in all_rows])
    df.to_csv(out_dir / "results.csv", index=False)

    pooled = _pool(df, all_row_ids)
    info_rows = []
    for row_id in all_row_ids:
        info = arm_infos[row_id]
        info_rows.append(dict(
            row_id=info.row_id, arm=info.arm, supervision=info.supervision, swept=info.swept,
            params=json.dumps(info.params), fit_wall_clock_s=info.fit_wall_clock_s,
            sweep_wall_clock_s=info.sweep_wall_clock_s,
            wall_clock_kind=info.wall_clock_kind,
            circuit_depth=(info.circuit_info or {}).get("depth"),
            n_qubits=(info.circuit_info or {}).get("n_qubits"),
            val_threshold=info.val_threshold, val_threshold_fp_rate=info.val_threshold_fp_rate,
            arm_status=info.status, arm_note=info.note,
        ))
    info_df = pd.DataFrame(info_rows)
    pooled = pooled.merge(info_df, on="row_id", how="left")
    pooled.to_csv(out_dir / "results_pooled.csv", index=False)

    manifest = dict(
        git_sha=_git_sha(), package_versions=_package_versions(),
        seed=args.seed, n_features=args.n_features, smoke=bool(args.smoke),
        target_recall=args.target_recall,
        n_test_scenes_total=len(all_scenes), n_test_scenes_scored=len(scenes),
        max_bg_per_scene=max_bg_per_scene,
        runtime_s=time.perf_counter() - t_start,
        arms={row_id: dict(
            arm=arm_infos[row_id].arm, supervision=arm_infos[row_id].supervision,
            swept=arm_infos[row_id].swept, params=arm_infos[row_id].params,
            sweep_table=arm_infos[row_id].sweep_table,
            fit_wall_clock_s=arm_infos[row_id].fit_wall_clock_s,
            sweep_wall_clock_s=arm_infos[row_id].sweep_wall_clock_s,
            wall_clock_kind=arm_infos[row_id].wall_clock_kind,
            circuit_info=arm_infos[row_id].circuit_info,
            val_threshold=arm_infos[row_id].val_threshold,
            val_threshold_fp_rate=arm_infos[row_id].val_threshold_fp_rate,
            status=arm_infos[row_id].status, note=arm_infos[row_id].note,
        ) for row_id in all_row_ids},
        kernel_concentration=kernel_concentration,
        outputs=dict(results_csv=str(out_dir / "results.csv"),
                     results_pooled_csv=str(out_dir / "results_pooled.csv")),
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\n=== SCENE-MACRO (PRIMARY) vs pixel-micro (secondary), natural prevalence ===")
    cols = ["row_id", "supervision", "swept", "n_scenes", "n_failed",
            "roc_auc_MACRO", "roc_auc_micro", "pr_auc_MACRO", "pr_auc_micro"]
    print(pooled[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out_dir/'results.csv'}, results_pooled.csv, run_manifest.json "
          f"in {time.perf_counter() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
