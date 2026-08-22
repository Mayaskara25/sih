"""PLAN.md 3E.6 / D27.2, D27.3, D27.5, D28 (rewritten 2026-08-22) -- the
classical partners in the classical-vs-quantum comparison, on the IDENTICAL
8 angle-encoded features `quantum.data.build_split` produces for every
quantum arm (VQC, QAE, quantum kernel). Comparing a quantum model on one
feature basis against a classical model on another measures the basis, not
the model (Section 3E.2's own warning, restated by D27 seven separate ways) --
so nothing here re-derives features; every arm consumes `split.X_train` /
`split.X_val` / `split.X_test` exactly as the quantum arms do.

FOUR ARMS, NOT THREE. The original brief specified three (`DenseAEArm`,
`OCSVMArm`, `SVCArm`); D28's rewrite added a fourth, `MahalanobisArm`, because
D28's own numbers made it the load-bearing one: plain Mahalanobis/RX on these
8 features already beats every measured quantum-kernel configuration on held-
out flightlines (train 0.8255 / val 0.7396 / test 0.6635), and a comparison
table omitting the cheapest, strongest baseline would flatter the branch by
omission. `MahalanobisArm` duplicates `anomaly/rx.py::global_rx`'s ridge
formula (`reg * trace(Sigma)/b * I` -- D22's scale-relative ridge, not an
absolute one) rather than importing it, because `global_rx` fits mu/Sigma
from and scores the SAME cube in one call; this branch needs fit-on-
background-train, score-on-arbitrary-X, which global_rx's signature does not
support. The formula is copied deliberately, not reinvented -- see D22's own
note on why an absolute ridge silently breaks on un-standardized scale.

SUPERVISION, PER ARM (D27.3 -- ranking a supervised and four unsupervised
arms in one column measures supervision, not quantumness):
    DenseAEArm, OCSVMArm, MahalanobisArm  -- unsupervised, background-only
        (`split.X_train[split.y_train == 0]`), same reason QAE trains
        background-only (Section 3A.6's original point, restated for this
        branch by D27.2): an autoencoder or one-class model trained on data
        containing its own targets learns to reconstruct/accept them.
    SVCArm                                -- supervised, full labelled
        `split.X_train` / `split.y_train` (the D27.3-required partner to
        VQC, the branch's other supervised arm).

VAL-ONLY HYPERPARAMETER SWEEPS LIVE IN THE RUNNER, NOT HERE. D28's rewrite
found the quantum kernel arm carrying a val-tuned hyperparameter (angle
scale) against an untuned RX baseline was itself an unequal-ground comparison
(the same D27/Section 3E.2 error, one level down: comparing tuning budgets,
not models) -- so `classical_vs_quantum.py` sweeps a small grid per arm,
selecting on VAL ROC-AUC only, never test. This module stays a plain
constructor-parameterized interface (gamma/nu/C/n_latent/reg are all __init__
kwargs) so the runner can instantiate one candidate per grid point; it does
not sweep itself.

SIGN ORIENTATION -- every arm's score() is HIGHER = MORE ANOMALOUS, matching
every other detector in this repo (`anomaly/kernel_rx.py`'s house style, the
frozen interface's own warning: a flipped score gives a mirrored AUC that
looks like a plausible bad result, not an obvious bug):
    - OCSVMArm negates `OneClassSVM.decision_function` (positive for
      inliers, sklearn's own convention -- `quantum/quantum_kernel.py`'s
      identical trap, same fix).
    - SVCArm does NOT negate `SVC.decision_function`. Verified empirically
      (not assumed): for a binary fit with labels {0, 1}, sklearn sorts
      `classes_` ascending, and `decision_function` is POSITIVE on the
      `classes_[1]` side -- confirmed on a synthetic two-blob split
      (mean decision_function 1.255 for class-1 points, -1.292 for class-0).
      Since this branch's label convention is 1 == anomaly (QuantumSplit),
      `classes_ == [0, 1]` makes positive decision_function already mean
      "more anomalous" with no sign flip needed. `fit()` asserts
      `classes_ == [0, 1]` structurally (not just once, empirically) so a
      future relabelling cannot silently reintroduce the mirror.
    - DenseAEArm and MahalanobisArm are error/distance scores by
      construction (reconstruction MSE, Mahalanobis distance): larger is
      already "further from the background model", i.e. more anomalous, no
      sign correction needed -- but this module's own test suite still
      pins it, per the "a check that passes for the wrong reason is worse
      than one that fails" rule D27/D28 both restate.

None of these four arms builds a quantum circuit -- `circuit_info()` returns
`None` for all of them, always. A runner recording `None` (not `0`) for these
rows is required, per D27.4: a missing circuit is a different fact than a
zero-depth one.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.svm import SVC, OneClassSVM
from torch import nn

from quantum.data import QuantumSplit


def _background_rows(split: QuantumSplit) -> np.ndarray:
    """split.X_train[split.y_train == 0] -- the one background-only slicing
    convention shared by DenseAEArm/OCSVMArm/MahalanobisArm, factored out so
    all three raise the SAME error on the SAME empty-background edge case
    (a --smoke run with a starved split could otherwise hit this three
    different ways with three different messages).
    """
    X = np.asarray(split.X_train, dtype=np.float64)
    y = np.asarray(split.y_train)
    return X[y == 0]


# --------------------------------------------------------------------- dense AE --

class _DenseAE(nn.Module):
    """8 -> n_hidden -> n_latent -> n_hidden -> 8 MLP, ReLU hidden activations,
    linear output (reconstruction is a real-valued angle-scaled feature, not a
    probability -- no output nonlinearity). Small on purpose: this must train
    in seconds on CPU (the brief's own words), not because the branch's other
    arms are slow but because a runner sweeping `n_latent` in {2, 4, 6}
    (D28's rewrite) pays this fit cost once per grid point per arm.
    """

    def __init__(self, n_features: int, n_latent: int, n_hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DenseAEArm:
    """Dense (non-convolutional) autoencoder on the 8 angle-encoded features
    -- the classical partner to `QuantumAutoencoderArm`
    (`quantum/quantum_autoencoder.py`), matched on INPUT REPRESENTATION
    (D27.2: a 15x15x30 spectral-spatial conv AE against an 8-dim vector QAE
    would compare representations, not architectures; this compares
    architectures on the SAME representation instead). `n_latent=4` by
    default to match `QuantumAutoencoderArm`'s own default.

    Trained on BACKGROUND ROWS ONLY (`split.y_train == 0`), same reasoning
    as the QAE (module docstring). Score = per-row reconstruction MSE,
    HIGHER = MORE ANOMALOUS (no sign correction needed -- an error score is
    already oriented this way; see the sign-orientation test).
    """

    name = "classical_ae"
    supervision = "unsupervised"

    def __init__(self, *, n_latent: int = 4, n_hidden: int = 6, n_epochs: int = 300,
                 lr: float = 1e-2, seed: int = 0) -> None:
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed

        self._model: _DenseAE | None = None
        self.fit_seconds: float | None = None

    def fit(self, split: QuantumSplit) -> None:
        """Full-batch Adam over `n_epochs` on the background rows. Full-batch
        (not mini-batched) because the background pool here is <= a few
        hundred rows -- see `quantum/data.py::build_split`'s per-split
        caps -- and full-batch keeps the fit deterministic given `seed`
        without a second RNG (a shuffled DataLoader) to seed.

        `torch.manual_seed(seed)` is called BEFORE constructing the model,
        so it also seeds the model's weight initialization, not just the
        optimizer step -- otherwise `seed` would reproduce training dynamics
        from a random starting point, not a reproducible fit end to end.
        """
        X_bg = _background_rows(split)
        if X_bg.shape[0] == 0:
            raise ValueError(
                "DenseAEArm.fit: split.X_train has zero background (y_train==0) "
                "rows -- nothing to train on")
        n_features = X_bg.shape[1]

        torch.manual_seed(self.seed)
        model = _DenseAE(n_features, self.n_latent, self.n_hidden).double()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        X = torch.from_numpy(X_bg)

        t0 = time.perf_counter()
        model.train()
        for _ in range(self.n_epochs):
            opt.zero_grad()
            recon = model(X)
            loss = torch.mean((recon - X) ** 2)
            loss.backward()
            opt.step()
        self.fit_seconds = time.perf_counter() - t0

        model.eval()
        self._model = model

    def score(self, X: np.ndarray) -> np.ndarray:
        """[N, n_features] -> [N] float64 per-row reconstruction MSE, HIGHER
        = MORE ANOMALOUS. `torch.no_grad()` -- scoring never needs gradients
        and computing them anyway would be pure waste on `score_scene_natural`'s
        per-scene calls (up to 28 per arm).
        """
        if self._model is None:
            raise RuntimeError("DenseAEArm.score called before fit()")
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float64))
        with torch.no_grad():
            recon = self._model(Xt)
            err = torch.mean((recon - Xt) ** 2, dim=1)
        return err.numpy()

    def circuit_info(self) -> dict | None:
        return None


# ------------------------------------------------------------------------ OC-SVM --

class OCSVMArm:
    """`sklearn.svm.OneClassSVM(kernel="rbf")` on background-only rows.

    Reference numbers (measured on the real split, brief-supplied, this
    class's job is to REPRODUCE them, not to be adjusted toward a different
    target if it doesn't): `gamma="scale", nu=0.1` -> train 0.6053 / val
    0.4508 / test 0.5568; `gamma=1.0` -> train 0.7573 / val 0.5613 / test
    0.5797. See `tests/test_quantum_comparison.py` for the reproduction
    check.
    """

    name = "classical_ocsvm"
    supervision = "unsupervised"

    def __init__(self, *, gamma: str | float = "scale", nu: float = 0.1, seed: int = 0) -> None:
        self.gamma = gamma
        self.nu = nu
        self.seed = seed  # accepted for interface symmetry; OneClassSVM's QP
        # solve is deterministic given data, so there is no RNG here to seed
        # (same convention QuantumKernelArm's own docstring documents).

        self._model: OneClassSVM | None = None
        self.fit_seconds: float | None = None

    def fit(self, split: QuantumSplit) -> None:
        X_bg = _background_rows(split)
        if X_bg.shape[0] == 0:
            raise ValueError(
                "OCSVMArm.fit: split.X_train has zero background (y_train==0) "
                "rows -- nothing to train on")
        t0 = time.perf_counter()
        model = OneClassSVM(kernel="rbf", gamma=self.gamma, nu=self.nu)
        model.fit(X_bg)
        self.fit_seconds = time.perf_counter() - t0
        self._model = model

    def score(self, X: np.ndarray) -> np.ndarray:
        """NEGATES `decision_function` -- POSITIVE for inliers in sklearn's
        convention, and this repo's contract is HIGHER = MORE ANOMALOUS
        (same trap, same fix, as `quantum/quantum_kernel.py::score_ocsvm`).
        """
        if self._model is None:
            raise RuntimeError("OCSVMArm.score called before fit()")
        return -np.asarray(self._model.decision_function(np.asarray(X, dtype=np.float64)),
                            dtype=np.float64)

    def circuit_info(self) -> dict | None:
        return None


# --------------------------------------------------------------------------- SVC --

class SVCArm:
    """`sklearn.svm.SVC(kernel="rbf", probability=False)` -- the branch's
    supervised classical arm, trained on the FULL labelled train split
    (both classes), the D27.3-required partner to `VQCArm`.
    """

    name = "classical_svc"
    supervision = "supervised"

    def __init__(self, *, gamma: str | float = "scale", C: float = 1.0, seed: int = 0) -> None:
        self.gamma = gamma
        self.C = C
        self.seed = seed  # interface symmetry only, see OCSVMArm's note; SVC's
        # QP solve is likewise deterministic given data.

        self._model: SVC | None = None
        self.fit_seconds: float | None = None

    def fit(self, split: QuantumSplit) -> None:
        X = np.asarray(split.X_train, dtype=np.float64)
        y = np.asarray(split.y_train)
        if np.unique(y).size < 2:
            raise ValueError(
                f"SVCArm.fit: split.X_train has {np.unique(y).size} distinct label(s) "
                "-- SVC needs both classes present in the (supervised) train split")
        t0 = time.perf_counter()
        model = SVC(kernel="rbf", gamma=self.gamma, C=self.C, probability=False)
        model.fit(X, y)
        self.fit_seconds = time.perf_counter() - t0
        # Structural sign-orientation guard (module docstring's SIGN
        # ORIENTATION note): decision_function is positive on the
        # classes_[1] side. This branch's labels are 1 == anomaly, so
        # classes_ MUST be exactly [0, 1] for score() below to already be
        # oriented correctly with no negation -- fail loudly rather than
        # silently mirror every score if that ever stops being true.
        if list(model.classes_) != [0, 1]:
            raise RuntimeError(
                f"SVCArm.fit: expected classes_ == [0, 1] (QuantumSplit's own "
                f"label convention), got {model.classes_!r} -- refusing to guess "
                "decision_function's sign and risk a silent mirror")
        self._model = model

    def score(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the separating hyperplane, HIGHER = MORE
        ANOMALOUS -- NOT negated, see module docstring's SIGN ORIENTATION
        note and `fit()`'s structural `classes_` guard.
        """
        if self._model is None:
            raise RuntimeError("SVCArm.score called before fit()")
        return np.asarray(self._model.decision_function(np.asarray(X, dtype=np.float64)),
                           dtype=np.float64)

    def circuit_info(self) -> dict | None:
        return None


# ------------------------------------------------------------------- Mahalanobis --

class MahalanobisArm:
    """Plain Mahalanobis distance on the 8 angle-encoded features,
    background-only -- `anomaly/rx.py::global_rx`'s formula, fit/score split
    apart (global_rx fits mu/Sigma from and scores the SAME cube in one
    call; this arm needs fit-on-train-background, score-on-arbitrary-X, so
    the formula is reimplemented here rather than imported -- see module
    docstring).

    D28's rewrite made this arm load-bearing, not decorative: measured on
    the real split it is train 0.8255 / val 0.7396 / test 0.6635, beating
    every measured quantum-kernel configuration on held-out flightlines --
    "four lines of numpy" per D28's own words. `name="rx_8feat"` marks it as
    RX on the 8-feature quantum basis, distinct from the classical-detector
    benchmark's own `global_rx` (which runs on full-band cubes, a different
    feature basis entirely -- Section 3E.2's warning about comparing across
    bases applies here too, which is why this arm exists rather than citing
    that other number directly).

    `reg` uses `anomaly/rx.py::global_rx`'s SCALE-RELATIVE ridge convention
    (D22): `reg * trace(Sigma)/b * I`, not an absolute `reg * I` -- an
    absolute ridge is meaningless against data whose scale is not controlled
    (D22's own ABU covariance-diagonal example), and these 8 features are
    angle-encoded to `[0, pi]`, a different scale than raw radiance, so the
    same argument applies even though the failure mode (`cho_factor`
    `LinAlgError`) has not been observed here.
    """

    name = "rx_8feat"
    supervision = "unsupervised"

    def __init__(self, *, reg: float = 1e-6, seed: int = 0) -> None:
        self.reg = reg
        self.seed = seed  # interface symmetry only; the Cholesky solve below
        # has no randomness to seed.

        self._mu: np.ndarray | None = None
        self._chol: tuple | None = None
        self.fit_seconds: float | None = None

    def fit(self, split: QuantumSplit) -> None:
        X_bg = _background_rows(split)
        if X_bg.shape[0] == 0:
            raise ValueError(
                "MahalanobisArm.fit: split.X_train has zero background (y_train==0) "
                "rows -- nothing to train on")
        t0 = time.perf_counter()
        mu = X_bg.mean(axis=0)
        centered = X_bg - mu
        b = X_bg.shape[1]
        sigma = (centered.T @ centered) / X_bg.shape[0]
        sigma = sigma + self.reg * (np.trace(sigma) / b) * np.eye(b, dtype=sigma.dtype)
        self._chol = cho_factor(sigma)
        self._mu = mu
        self.fit_seconds = time.perf_counter() - t0

    def score(self, X: np.ndarray) -> np.ndarray:
        """(x - mu)^T @ inv(Sigma + reg*I) @ (x - mu), HIGHER = MORE
        ANOMALOUS -- a squared Mahalanobis distance is already oriented this
        way, no sign correction needed.
        """
        if self._mu is None or self._chol is None:
            raise RuntimeError("MahalanobisArm.score called before fit()")
        X = np.asarray(X, dtype=np.float64)
        dv = X - self._mu
        solved = cho_solve(self._chol, dv.T)
        return np.einsum("ij,ji->i", dv, solved)

    def circuit_info(self) -> dict | None:
        return None
