"""Gated multimodal linear models for fMRI delta prediction.

This module provides :class:`GatedMultimodalLinearDeltaModel` — a fully
linear, closed-form model that mixes a video-only ridge predictor with an
audio-only ridge predictor through a per-voxel scalar gate ``α``.

Design rationale
----------------
The companion banded-ridge model in :mod:`models` typically reduces video
dimensionality with PCA before concatenation.  Applying PCA to a feature
matrix that is itself a (forecasting) time series mixes temporally adjacent
samples into each principal component — i.e. it acts as a non-causal
spectral filter — which is awkward when the downstream task is to predict
the *future* BOLD signal.

This module avoids PCA entirely: the full-dimensional video and audio
streams are fed into two independent per-voxel ridge regressions, and a
closed-form per-voxel scalar ``α_j ∈ [0, 1]`` chooses how much each voxel
listens to video vs audio.  This gives a directly visualisable 3-D map of
modality preference without any neural network, sigmoid layer, or
iterative optimiser.
"""

import numpy as np

import utils
from models import Preprocessor


class GatedMultimodalLinearDeltaModel(Preprocessor):
    """Predict frame-to-frame fMRI delta with a per-voxel video / audio gate.

    Three stages, all in closed form:

    1. **Video-only ridge** ``W_v``:  predicts the BOLD delta from
       ``X_video`` alone with penalty ``alpha_video``.
    2. **Audio-only ridge** ``W_a``:  predicts the BOLD delta from
       ``X_audio`` alone with penalty ``alpha_audio``.
    3. **Per-voxel gate** ``α``: for every voxel ``j``, the optimal mixture
       coefficient is

       .. math::

           \\alpha_j = \\mathrm{clip}\\!\\left(
                \\frac{\\langle D_j,\\, R_j \\rangle}
                     {\\langle D_j,\\, D_j \\rangle + \\varepsilon},
                0,\\, 1
            \\right)

       where ``D_j = V_j − A_j`` (video − audio prediction) and
       ``R_j = Y_j − A_j`` (audio-only residual), both on the training set.

    The final per-voxel prediction is

    .. math::

        \\hat{y}_j(t) = \\alpha_j\\, V_j(t) + (1 - \\alpha_j)\\, A_j(t).

    Interpretation of ``α``
    -----------------------
    ``α_j`` is a single scalar per voxel:

    * ``α_j → 1``   ⇒  voxel ``j`` is best explained by **video**;
    * ``α_j → 0``   ⇒  voxel ``j`` is best explained by **audio**;
    * ``α_j ≈ 0.5`` ⇒  both modalities contribute similarly.

    ``alpha_volume()`` reshapes ``α`` to the downsampled BOLD grid for
    direct visualisation as a 3-D map.

    Why no PCA on video?
    --------------------
    Applying PCA to a forecasting-style time series mixes temporally
    adjacent samples into each principal component (spectral filtering),
    which is questionable when the target is to predict the *future* signal.
    Here the video stream is fed at full dimensionality and ridge
    regularisation handles the high dimensionality.

    Parameters
    ----------
    X_video : np.ndarray
        Video feature matrix of shape ``(n_frames, d_video)``.  Should be
        already resampled and standardised.
    X_audio : np.ndarray
        Audio feature matrix of shape ``(n_frames, d_audio)`` aligned to the
        same time grid as ``X_video``.
    sub : data_loading.Sub
        Subject providing the BOLD volume.
    dt : float
        Hemodynamic delay in seconds.
    coef : int
        AvgPool factor for the BOLD volume (``coef == 1`` means no spatial
        downsampling).
    alpha_video : float
        Ridge penalty for the video-only fit.
    alpha_audio : float
        Ridge penalty for the audio-only fit.
    train_size : float, default 0.7
        Fraction of TRs used for training.

    Attributes
    ----------
    W_video : np.ndarray, shape (n_voxels, d_video)
    W_audio : np.ndarray, shape (n_voxels, d_audio)
    alpha   : np.ndarray, shape (n_voxels,)
        Per-voxel gating coefficients in ``[0, 1]``.
    """

    def __init__(
        self,
        X_video: np.ndarray,
        X_audio: np.ndarray,
        sub,
        dt: float,
        coef: int,
        alpha_video: float,
        alpha_audio: float,
        train_size: float = 0.7,
    ) -> None:
        if X_video.shape[0] != X_audio.shape[0]:
            raise ValueError(
                f"X_video and X_audio must have matching n_frames; "
                f"got {X_video.shape[0]} vs {X_audio.shape[0]}."
            )

        self.d_video = X_video.shape[1]
        self.d_audio = X_audio.shape[1]
        self.alpha_video = alpha_video
        self.alpha_audio = alpha_audio

        # Concatenate so the parent Preprocessor can build paired (X, Y) data
        # using the same train/test split logic as the other models.
        vector_list = np.concatenate([X_video, X_audio], axis=1)
        super().__init__(vector_list, sub, dt, coef, train_size)
        self.delta = True

        (
            self.Xv_train, self.Xa_train,
            self.Y_train, self.deltaY_train,
            self.Xv_test, self.Xa_test,
            self.Y_test, self.deltaY_test,
        ) = self._build_split_delta_data()

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #

    def _build_split_delta_data(self):
        """Build delta train/test pairs and split X back into video/audio blocks."""
        delta_train = [
            (self.train[n][0], self.train[n][1] - self.train[n - 1][1])
            for n in range(1, len(self.train))
        ]
        delta_test = [
            (self.test[n][0], self.test[n][1] - self.test[n - 1][1])
            for n in range(1, len(self.test))
        ]

        X_train = np.array([p[0] for p in delta_train])
        X_test = np.array([p[0] for p in delta_test])

        Xv_train = X_train[:, : self.d_video]
        Xa_train = X_train[:, self.d_video :]
        Xv_test = X_test[:, : self.d_video]
        Xa_test = X_test[:, self.d_video :]

        Y_train = np.array([p[1] for p in self.train]).T
        Y_test = np.array([p[1] for p in self.test]).T
        deltaY_train = np.array([p[1] for p in delta_train]).T
        deltaY_test = np.array([p[1] for p in delta_test]).T

        return (
            Xv_train, Xa_train, Y_train, deltaY_train,
            Xv_test, Xa_test, Y_test, deltaY_test,
        )

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ridge_solve(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
        """Closed-form ridge regression.

        Returns the weight matrix ``W`` of shape ``(n_outputs, d)`` so that
        ``Y_pred = W @ X.T``.
        """
        if alpha > 0:
            A = (
                np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T
            )
        else:
            A = np.linalg.pinv(X)
        return (A @ Y.T).T

    def fit(self) -> None:
        """Fit video-only ridge, audio-only ridge, and per-voxel gate ``α``."""
        # Stages 1 & 2: independent per-modality ridge fits on BOLD deltas.
        self.W_video = self._ridge_solve(
            self.Xv_train, self.deltaY_train, self.alpha_video
        )
        self.W_audio = self._ridge_solve(
            self.Xa_train, self.deltaY_train, self.alpha_audio
        )

        # Stage 3: per-voxel optimal mixing weight α_j (closed form).
        delta_v_pred = self.W_video @ self.Xv_train.T  # (n_voxels, n_t)
        delta_a_pred = self.W_audio @ self.Xa_train.T

        D = delta_v_pred - delta_a_pred       # video - audio
        R = self.deltaY_train - delta_a_pred  # residual after audio
        num = np.sum(D * R, axis=1)
        den = np.sum(D * D, axis=1) + 1e-12
        self.alpha = np.clip(num / den, 0.0, 1.0)  # (n_voxels,)

    # ------------------------------------------------------------------ #
    # Prediction & evaluation
    # ------------------------------------------------------------------ #

    def predict(self) -> None:
        """Predict gated deltas and reconstruct absolute BOLD volumes."""
        delta_v_train = self.W_video @ self.Xv_train.T
        delta_a_train = self.W_audio @ self.Xa_train.T
        delta_v_test = self.W_video @ self.Xv_test.T
        delta_a_test = self.W_audio @ self.Xa_test.T

        a = self.alpha[:, None]  # (n_voxels, 1) for broadcasting
        self.deltaY_train_predicted = (
            a * delta_v_train + (1 - a) * delta_a_train
        )
        self.deltaY_test_predicted = (
            a * delta_v_test + (1 - a) * delta_a_test
        )

        self.Y_train_predicted = (
            np.delete(self.Y_train, -1, 1) + self.deltaY_train_predicted
        )
        self.Y_test_predicted = (
            np.delete(self.Y_test, -1, 1) + self.deltaY_test_predicted
        )

        # Cache modality-only delta predictions for diagnostics.
        self._delta_v_test = delta_v_test
        self._delta_a_test = delta_a_test

    def evaluate(self, mask: np.ndarray | None = None) -> None:
        """Compute and store train / test MSE.

        Parameters
        ----------
        mask : np.ndarray of dtype bool, optional
            1-D voxel mask passed through to :func:`utils.MSE`.  When set,
            MSE is averaged only over the selected voxels (e.g. a brain
            mask), so background air does not dominate the score.
        """
        self.MSE_train = utils.MSE(
            self.Y_train_predicted - np.delete(self.Y_train, 0, 1), mask=mask
        )
        self.MSE_test = utils.MSE(
            self.Y_test_predicted - np.delete(self.Y_test, 0, 1), mask=mask
        )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def modality_mse(self, mask: np.ndarray | None = None) -> dict:
        """Return test-set MSE for video-only, audio-only, and gated models.

        Useful sanity check: the gated MSE should not exceed
        ``min(video, audio)``; if it does, ``alpha_video`` / ``alpha_audio``
        are likely mis-tuned.

        Parameters
        ----------
        mask : np.ndarray of dtype bool, optional
            1-D voxel mask.  When set, all three MSEs are averaged only
            over the selected voxels.  ``self.MSE_test`` is recomputed
            under the same mask for a fair comparison.
        """
        if not hasattr(self, "_delta_v_test"):
            raise RuntimeError("Call predict() before modality_mse().")

        true_test = np.delete(self.Y_test, 0, 1)
        Y_v = np.delete(self.Y_test, -1, 1) + self._delta_v_test
        Y_a = np.delete(self.Y_test, -1, 1) + self._delta_a_test

        gated_mse = utils.MSE(self.Y_test_predicted - true_test, mask=mask)
        return {
            "video": utils.MSE(Y_v - true_test, mask=mask),
            "audio": utils.MSE(Y_a - true_test, mask=mask),
            "gated": gated_mse,
        }

    def alpha_volume(self) -> np.ndarray:
        """Return ``α`` as a 3-D voxel volume of shape ``(_d1, _d2, _d3)``."""
        if not hasattr(self, "alpha"):
            raise RuntimeError("Call fit() before alpha_volume().")
        return self.alpha.reshape(self._d1, self._d2, self._d3)

    def mean_bold_volume(self) -> np.ndarray:
        """Return the temporal-mean BOLD on the downsampled grid for use as an underlay."""
        return self.sub._tensor.numpy().mean(axis=-1)

    def confidence_mask(
        self,
        predictability_quantile: float = 0.5,
        separability_quantile: float = 0.3,
    ) -> np.ndarray:
        """Identify voxels where ``α`` is meaningful.

        Two flat-array (per-voxel) criteria, both relative quantiles so they
        adapt to each subject:

        1. **Predictability**: per-voxel test-set R² above the
           ``predictability_quantile``-th quantile.  Voxels whose BOLD time
           series neither modality can fit are not informative about
           modality preference.
        2. **Separability**: per-voxel ``‖V − A‖²`` above the
           ``separability_quantile``-th quantile.  Where the two modalities
           predict almost the same delta, ``α`` is numerically unstable.

        Parameters
        ----------
        predictability_quantile : float, default 0.5
            Lower-tail fraction of voxels (by R²) to drop.
        separability_quantile : float, default 0.3
            Lower-tail fraction of voxels (by ``‖V − A‖``) to drop.

        Returns
        -------
        np.ndarray of dtype bool, shape ``(n_voxels,)``.
        """
        if not hasattr(self, "_delta_v_test"):
            raise RuntimeError("Call predict() before confidence_mask().")

        true_test = np.delete(self.Y_test, 0, 1)
        err = np.mean(
            (self.Y_test_predicted - true_test) ** 2, axis=1
        )
        var_y = np.var(true_test, axis=1) + 1e-12
        r2 = 1.0 - err / var_y

        diff = self._delta_v_test - self._delta_a_test
        sep = np.mean(diff ** 2, axis=1)

        pred_thr = np.quantile(r2, predictability_quantile)
        sep_thr = np.quantile(sep, separability_quantile)
        return (r2 > pred_thr) & (sep > sep_thr)

    def confidence_volume(self, **mask_kwargs) -> np.ndarray:
        """Confidence mask reshaped to the 3-D voxel grid."""
        return self.confidence_mask(**mask_kwargs).reshape(
            self._d1, self._d2, self._d3
        )
