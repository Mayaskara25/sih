"""PLAN.md §3A.3 -- RBF-kernel RX (Kwon & Nasrabadi-style kernelized RX) via a
Woodbury-identity dual solve, matching anomaly/rx.py::global_rx's
cho_factor/cho_solve convention.

Derivation. Linear RX in a (possibly infinite-dimensional) feature space is
    r(x) = (phi(x) - mu)^T (Sigma_phi + reg*I)^-1 (phi(x) - mu)
    Sigma_phi = (1/N) * Phi_c^T Phi_c        (Phi_c: background, feature-centered)
Applying the Woodbury identity to (Sigma_phi + reg*I)^-1 swaps the B x B (or
infinite-dimensional) inverse for an N x N one built entirely from kernel
evaluations on the N background points:
    r(x) = (1/reg) * [ ||phi(x) - mu||^2  -  g_x^T (N*reg*I_N + G)^-1 g_x ]
where G is the doubly-centered background Gram matrix (G_ij = <phi(x_i)-mu,
phi(x_j)-mu>) and g_x is the centered kernel vector between x and each
background point (same double-centering as sklearn.preprocessing
.KernelCenterer / kernel-PCA out-of-sample projection). Setting k(a,b) = a.b
(linear kernel) reduces this exactly to global_rx's Mahalanobis form -- the
standard correctness check for a "kernelized" detector. The N x N system is
solved with cho_factor/cho_solve, never an explicit inverse.

Background is a FIXED random subsample of n_background valid pixels (the
kernel Gram matrix is O(N^2), so scoring against every valid pixel does not
scale) -- seeded, because an unseeded subsample makes the reported AUC
irreproducible run to run.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist, pdist


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma * ||a_i - b_j||^2) for all pairs. a: [M,B], b: [N,B] -> [M,N]."""
    return np.exp(-gamma * cdist(a, b, metric="sqeuclidean"))


def _median_heuristic_gamma(bg: np.ndarray) -> float:
    """1 / (2 * median(pdist(bg)^2)) -- the standard data-driven RBF
    bandwidth when no prior on the target scale exists. Exposed as its own
    function (rather than inlined) so the "gamma is computed, not None"
    behaviour is directly testable.
    """
    if bg.shape[0] < 2:
        return 1.0
    sq = pdist(bg, metric="sqeuclidean")
    med = np.median(sq)
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def kernel_rx(cube: np.ndarray, *, gamma: float | None = None, n_background: int = 2000,
              reg: float = 1e-2, seed: int = 0) -> np.ndarray:
    """RBF-kernel RX in feature space.

    cube: [H, W, B] float32, NaN = nodata.
    returns: [H, W] float32, NaN wherever the input pixel was NaN.

    O(N^2) in background samples, so the background is a fixed random
    subsample of n_background pixels (seeded -- an unseeded subsample makes
    the reported AUC irreproducible). gamma=None -> median-heuristic:
    1 / (2 * median(pdist(bg)^2)).

    `reg` DEFAULTS TO 1e-2, NOT to global_rx's 1e-6 (D22.2). The plan's
    signature carried 1e-6 over from linear RX, but the regularized operator
    here is a different object: the centered RBF Gram has unit-scale entries
    (k(x,x) == 1), so `N*reg = 2000*1e-6 = 2e-3` leaves the N x N system
    effectively UNREGULARIZED, and it therefore interpolates -- fitting the
    background subsample exactly.
    The score then measures "were you in the random subsample?" rather than
    "are you anomalous", and the failure is SILENT: a +50 sigma implanted
    outlier ranked 481st of 900, identically to a +2 sigma one, because rank
    had stopped depending on the data at all. On abu-beach-2, reg=1e-6 scores
    AUC 0.845 against 0.921 for anything in [1e-3, 1] -- a flat plateau, so
    1e-2 is a mid-plateau choice rather than a tuned one.
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b).astype(np.float64)
    valid = ~np.any(np.isnan(flat), axis=-1)
    valid_idx = np.flatnonzero(valid)

    scores = np.full(flat.shape[0], np.nan, dtype=np.float64)
    if valid_idx.size == 0:
        return scores.reshape(h, w).astype(np.float32)

    rng = np.random.default_rng(seed)
    n_bg = min(n_background, valid_idx.size)
    bg_idx = rng.choice(valid_idx, size=n_bg, replace=False)
    bg = flat[bg_idx]

    if gamma is None:
        gamma = _median_heuristic_gamma(bg)

    k_bg = _rbf(bg, bg, gamma)                       # [N, N], diag == 1 for RBF
    bg_row_mean = k_bg.mean(axis=1)                   # [N]  (== col mean, K symmetric)
    total_mean = float(bg_row_mean.mean())
    # Double-centering H K H (kernel-PCA convention): this is G = Phi_c Phi_c^T.
    g = k_bg - bg_row_mean[:, None] - bg_row_mean[None, :] + total_mean

    n = n_bg
    system = n * reg * np.eye(n) + g
    c, lower = cho_factor(system)

    z = flat[valid_idx]                                # [M, B]
    k_test = _rbf(z, bg, gamma)                        # [M, N]; k(z,z) == 1 for RBF
    z_row_mean = k_test.mean(axis=1)                   # [M]
    d2 = 1.0 - 2.0 * z_row_mean + total_mean            # ||phi(z) - mu||^2

    # Centered kernel vector g_z, same double-centering applied out-of-sample
    # (sklearn.preprocessing.KernelCenterer.transform convention).
    g_z = k_test - z_row_mean[:, None] - bg_row_mean[None, :] + total_mean  # [M, N]
    solved = cho_solve((c, lower), g_z.T)               # [N, M]
    quad = np.einsum("mi,im->m", g_z, solved)           # [M]

    scores[valid_idx] = (d2 - quad) / reg
    return scores.reshape(h, w).astype(np.float32)
