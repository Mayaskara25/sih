"""S3A.9 -- multi-signal score fusion (D3 rank normalization, D20 component-adaptive).

D20's collision, restated: `spectral_index_score` needs `meta.wavelengths`,
which ABU, HYDICE and Indian Pines do not ship (D13.4) -- HAD100 is the only
benchmark that does. So `fuse_scores` cannot assume all four §3A.8 components
(`rx`, `ace`, `index`, `spatial`) are always available.

**Decision (D20): fusion is component-adaptive.** `fuse_scores` takes
whatever components it is given and renormalizes the weights of the
components actually present to sum to 1.0. It never substitutes a zero
raster for a missing component -- after rank normalization a zero-filled
channel is not "no information", it is a CONSTANT, and averaging a constant
into a weighted sum drags every fused score toward that same value,
quietly damaging the very ranking the AUC is computed from. The active
component set is recorded in the result so a number is never reported as
bare "fusion" when it was actually `fusion(rx+ace+spatial)` (the ABU case).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anomaly.scoring import rank_normalize

# S3A.9 defaults. Only consulted for components actually present in a given
# call -- an unused key (e.g. "index" when scoring ABU) is simply ignored,
# not required to be dropped by the caller first.
DEFAULT_WEIGHTS: dict[str, float] = {"rx": 0.40, "ace": 0.25, "index": 0.15, "spatial": 0.20}


@dataclass(frozen=True)
class FusionResult:
    """Bundles the fused score with the component set that produced it.

    Why not just return `np.ndarray`, matching the repo's usual detector
    convention? Because D20's failure mode is specifically a *labeling*
    failure: a 3-component ABU result (`rx+ace+spatial`) silently reported
    as if it were the 4-component HAD100 one, because the array alone
    carries no record of what went into it. A bare-array return would push
    that bookkeeping onto every caller -- thread the active set through a
    separate variable, remember not to lose it across a function boundary --
    which is exactly the kind of manual discipline that fails silently
    (§3A.9's accept criterion literally requires the method column to say
    `fusion(rx+ace+spatial)`, never bare "fusion"). Binding `score`,
    `components` and `weights` in one immutable object makes the mislabeling
    structurally harder rather than merely documented against, at the cost
    of one extra attribute access (`result.score`) versus a bare array.
    """
    score: np.ndarray            # [H, W] float32, in [0, 1], NaN preserved
    components: tuple[str, ...]  # active component set, alphabetically sorted
    weights: dict[str, float]    # renormalized weights actually used; sums to 1.0


def fuse_scores(components: dict[str, np.ndarray], weights: dict[str, float] | None = None,
                 *, valid: np.ndarray | None = None) -> FusionResult:
    """Fuse heterogeneous detector outputs into one [0,1] anomaly score.

    Every component is rank-normalized (D3's `rank_normalize`, reused from
    `anomaly.scoring` -- not reimplemented here) BEFORE weighting: an RX
    Mahalanobis distance, an ACE cosine-squared ratio and a spectral index
    are on incomparable native scales, and rank normalization is the only
    transform in this codebase that guarantees a common one. The
    rank-normalized components are already each in [0, 1]; since the active
    weights are renormalized to sum to 1.0 (below), their weighted sum is a
    convex combination and therefore already in [0, 1] -- `clip` only
    defends against float round-off at the boundary, it never needs to
    rescale.

    Component-adaptive weighting (D20): `active = sorted(components)`. Only
    the weights for components actually present are consulted, and they are
    divided by their own sum so the active set's weights sum to exactly 1.0.
    A component simply absent from `components` (e.g. `index` on ABU,
    because `spectral_index_score` raised on a wavelength-less scene) is
    dropped from both the numerator and the weight-normalization denominator
    -- it is never treated as a zero-valued component with its original
    weight still applied, which would silently bias every fused score toward
    zero along that missing dimension.

    Parameters
    ----------
    components : e.g. `{"rx": rx_map, "ace": ace_map, "spatial": spatial_map}`
        (`index` omitted on ABU/HYDICE/Indian Pines, D13.4/D20). Every
        array must share the same [H, W] shape.
    weights : keyed the same as `components`; defaults to S3A.9's
        `{rx: 0.40, ace: 0.25, index: 0.15, spatial: 0.20}`. Keys for
        components not present in `components` are ignored; every key
        present in `components` MUST have a weight.
    valid : optional [H, W] bool mask. If omitted, computed as the
        intersection of "not NaN" across every active component -- a pixel
        NaN in any one component cannot be meaningfully ranked against the
        others, so it is excluded from all of them (NaN locality, D15-style:
        NaN in any active component -> NaN out, positionally).

    Returns
    -------
    FusionResult
        `score`: [H, W] float32 in [0, 1], NaN wherever any active
        component (or the caller-supplied `valid` mask) was invalid.
        `components`: the active component names, sorted.
        `weights`: the renormalized weights actually used, summing to 1.0.
    """
    if not components:
        raise ValueError("fuse_scores: no components given")
    if weights is None:
        weights = DEFAULT_WEIGHTS

    active = tuple(sorted(components))

    missing_weight = [name for name in active if name not in weights]
    if missing_weight:
        raise ValueError(
            f"fuse_scores: no weight configured for component(s) {missing_weight}; "
            f"weights given for {sorted(weights)}")

    raw_total = sum(weights[name] for name in active)
    if raw_total <= 0:
        raise ValueError(f"fuse_scores: active components' weights sum to {raw_total} <= 0")
    active_weights = {name: weights[name] / raw_total for name in active}

    shapes = {components[name].shape for name in active}
    if len(shapes) != 1:
        raise ValueError(
            f"fuse_scores: component shapes disagree: "
            f"{ {name: components[name].shape for name in active} }")
    shape = shapes.pop()

    if valid is None:
        valid = np.ones(shape, dtype=bool)
        for name in active:
            valid = valid & ~np.isnan(components[name])

    fused = np.zeros(shape, dtype=np.float64)
    for name in active:
        ranked = rank_normalize(components[name], valid=valid)
        # ranked is NaN outside `valid` by construction (rank_normalize);
        # zero-fill only for the weighted sum's arithmetic, then mask the
        # whole result back to NaN at those positions below -- this is NOT
        # the D20-forbidden "zero-fill a missing component": every name
        # summed here is an active, present component, just NaN at a small
        # set of positionally-invalid pixels within it.
        fused += active_weights[name] * np.nan_to_num(ranked, nan=0.0)

    fused = np.clip(fused, 0.0, 1.0)
    fused = np.where(valid, fused, np.nan).astype(np.float32)

    return FusionResult(score=fused, components=active, weights=active_weights)
