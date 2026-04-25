import numpy as np
import torch
from sklearn.decomposition import PCA
import utils

class Preprocessor:
    def __init__(self, vector_list, sub, dt, coef, train_size=0.7):
        self.vector_list = vector_list
        self.sub = sub
        self.dt = dt
        self.coef = coef
        self.train_size = train_size

        self.nu = 25
        self.mu = 641. / 390.
        self.d1 = self.sub.tensor.shape[0]
        self.d2 = self.sub.tensor.shape[1]
        self.d3 = self.sub.tensor.shape[2]
        self.d4 = self.sub.tensor.shape[3]
        self.N = 641 - int(self.mu * self.dt)

        self.train, self.test = self.get_train_test()

    def get_train_test(self):
        pairs = [
            (int(i * self.nu / self.mu), int(self.mu * self.dt + i))
            for i in range(self.N)
        ]

        if self.coef > 1:
            maxpool = torch.nn.AvgPool3d(kernel_size=self.coef, stride=self.coef)
            self.sub._tensor = maxpool(
                self.sub.tensor.permute(3, 0, 1, 2)
            ).permute(1, 2, 3, 0)
        else:
            self.sub._tensor = self.sub.tensor

        self._d1 = self.sub._tensor.shape[0]
        self._d2 = self.sub._tensor.shape[1]
        self._d3 = self.sub._tensor.shape[2]
        self._d4 = self.sub._tensor.shape[3]

        voxels = [
            self.sub._tensor[:, :, :, i].reshape(self._d1 * self._d2 * self._d3).numpy()
            for i in range(self._d4)
        ]

        data = [(self.vector_list[n], voxels[k]) for n, k in pairs]

        l = int(self.train_size * self._d4)
        train, test = data[:l], data[l:]

        train_voxels = np.array([p[1] for p in train])
        test_voxels  = np.array([p[1] for p in test])
        mn, mx = np.min(train_voxels), np.max(train_voxels)
        train_voxels = (train_voxels - mn) / (mx - mn)
        test_voxels  = (test_voxels  - mn) / (mx - mn)

        train = [(p[0], tv) for p, tv in zip(train, train_voxels)]
        test  = [(p[0], tv) for p, tv in zip(test,  test_voxels)]
        return train, test


class LinearModel(Preprocessor):
    def __init__(self, vector_list, sub, dt, coef, alpha, train_size=0.7):
        super().__init__(vector_list, sub, dt, coef, train_size)
        self.delta = False
        self.alpha = alpha
        self.X_train, self.Y_train, self.X_test, self.Y_test = self._get_XY()

    def _get_XY(self):
        return (
            np.array([p[0] for p in self.train]),
            np.array([p[1] for p in self.train]).T,
            np.array([p[0] for p in self.test]),
            np.array([p[1] for p in self.test]).T,
        )

    def fit(self):
        A = (np.linalg.inv(self.X_train.T @ self.X_train
             + self.alpha * np.eye(self.X_train.shape[1]))
             @ self.X_train.T) if self.alpha > 0 else np.linalg.pinv(self.X_train)
        self.W = (A @ self.Y_train.T).T

    def predict(self):
        self.Y_train_predicted = self.W @ self.X_train.T
        self.Y_test_predicted  = self.W @ self.X_test.T

    def evaluate(self):
        self.MSE_train = utils.MSE(self.Y_train_predicted - self.Y_train)
        self.MSE_test  = utils.MSE(self.Y_test_predicted  - self.Y_test)


class LinearDeltaModel(Preprocessor):
    def __init__(self, vector_list, sub, dt, coef, alpha, train_size=0.7):
        super().__init__(vector_list, sub, dt, coef, train_size)
        self.delta = True
        self.alpha = alpha
        (self.X_train, self.Y_train, self.deltaY_train,
         self.X_test,  self.Y_test,  self.deltaY_test) = self._get_XY_delta()

    def _get_XY_delta(self):
        delta_train = [(self.train[n][0], self.train[n][1] - self.train[n-1][1])
                       for n in range(1, len(self.train))]
        delta_test  = [(self.test[n][0],  self.test[n][1]  - self.test[n-1][1])
                       for n in range(1, len(self.test))]
        return (
            np.array([p[0] for p in delta_train]),
            np.array([p[1] for p in self.train]).T,
            np.array([p[1] for p in delta_train]).T,
            np.array([p[0] for p in delta_test]),
            np.array([p[1] for p in self.test]).T,
            np.array([p[1] for p in delta_test]).T,
        )

    def fit(self):
        A = (np.linalg.inv(self.X_train.T @ self.X_train
             + self.alpha * np.eye(self.X_train.shape[1]))
             @ self.X_train.T) if self.alpha > 0 else np.linalg.pinv(self.X_train)
        self.W = (A @ self.deltaY_train.T).T

    def predict(self):
        self.deltaY_train_predicted = self.W @ self.X_train.T
        self.deltaY_test_predicted  = self.W @ self.X_test.T
        self.Y_train_predicted = np.delete(self.Y_train, -1, 1) + self.deltaY_train_predicted
        self.Y_test_predicted  = np.delete(self.Y_test,  -1, 1) + self.deltaY_test_predicted

    def evaluate(self, Y_test_predicted=None):
        if Y_test_predicted is None:
            self.MSE_train = utils.MSE(self.Y_train_predicted - np.delete(self.Y_train, 0, 1))
            self.MSE_test  = utils.MSE(self.Y_test_predicted  - np.delete(self.Y_test,  0, 1))
        else:
            return utils.MSE(Y_test_predicted - np.delete(self.Y_test, 0, 1))

    def repredict(self, W):
        dY = W @ self.X_test.T
        return np.delete(self.Y_test, -1, 1) + dY



class MultimodalPreprocessor(Preprocessor):
    """
    Extends Preprocessor with optional PCA applied to the full feature matrix.
    The feature matrix is assumed to be already normalised per modality
    (see dataloader.get_multimodal_encoding with normalize=True).
    """

    def __init__(self, vector_list, sub, dt, coef,
                 pca_components: int | None, train_size=0.7):
        self.pca_components = pca_components
        self.pca = None

        if pca_components is not None and pca_components < vector_list.shape[1]:
            print(f"[PCA] {vector_list.shape[1]} → {pca_components} components")
            self.pca = PCA(n_components=pca_components, random_state=42)
            vector_list = self.pca.fit_transform(vector_list)
            print(f"[PCA] Explained variance: {self.pca.explained_variance_ratio_.sum()*100:.1f}%")

        super().__init__(vector_list, sub, dt, coef, train_size)


class MultimodalLinearDeltaModel(MultimodalPreprocessor):
    """
    Predicts frame-to-frame fMRI delta using concatenated audio + video features.

    Banded ridge regression
    -----------------------
    A separate regularisation coefficient is applied to each modality block:

        min  ‖δ - X θ‖²  +  α_v ‖θ_video‖²  +  α_a ‖θ_audio‖²
         θ

    This is equivalent to a block-diagonal penalty matrix:

        Λ = diag( α_v · I_{d_v},  α_a · I_{d_a} )

    and the closed-form solution is:

        θ = (XᵀX + Λ)⁻¹ Xᵀ δ

    Why banded ridge?
    -----------------
    ViT features (768-d) and MFCC features (15-d) live in very different
    spaces and carry different amounts of information per dimension.  A single
    global alpha either over-regularises the audio block (making it invisible)
    or under-regularises the video block (causing overfitting on video noise).
    Separate alphas let each modality contribute optimally.

    Note: when PCA is enabled the modality blocks are mixed, so banded ridge
    is not applicable — only global alpha_video is used in that case.

    Parameters
    ----------
    vector_list : np.ndarray, shape (n_frames, d_video + d_audio)
        Pre-built, per-modality normalised multimodal features.
    X_video, X_audio : np.ndarray
        Individual modality arrays returned by get_multimodal_encoding().
        Only needed when pca_components is None (for banded ridge).
    alpha_video : float
        Ridge penalty for the video block (or global penalty when PCA is used).
    alpha_audio : float
        Ridge penalty for the audio block (ignored when PCA is used).
    """

    def __init__(
        self,
        vector_list: np.ndarray,
        sub,
        dt: float,
        coef: int,
        alpha_video: float,
        alpha_audio: float,
        X_video: np.ndarray | None = None,
        X_audio: np.ndarray | None = None,
        pca_components: int | None = None,
        train_size: float = 0.7,
    ):
        self.alpha_video = alpha_video
        self.alpha_audio = alpha_audio
        self._X_video_full = X_video
        self._X_audio_full = X_audio

        super().__init__(vector_list, sub, dt, coef, pca_components, train_size)
        self.delta = True

        (self.X_train, self.Y_train, self.deltaY_train,
         self.X_test,  self.Y_test,  self.deltaY_test) = self._get_XY_delta()


    def _get_XY_delta(self):
        delta_train = [(self.train[n][0], self.train[n][1] - self.train[n-1][1])
                       for n in range(1, len(self.train))]
        delta_test  = [(self.test[n][0],  self.test[n][1]  - self.test[n-1][1])
                       for n in range(1, len(self.test))]
        return (
            np.array([p[0] for p in delta_train]),
            np.array([p[1] for p in self.train]).T,
            np.array([p[1] for p in delta_train]).T,
            np.array([p[0] for p in delta_test]),
            np.array([p[1] for p in self.test]).T,
            np.array([p[1] for p in delta_test]).T,
        )

    def _build_penalty(self) -> np.ndarray:
        """
        Build the block-diagonal penalty vector Λ (diagonal of the penalty matrix).

        When PCA is active the modality blocks are mixed, so we fall back to a
        uniform penalty using alpha_video as the global value.
        """
        d = self.X_train.shape[1]

        if self.pca is not None:
            print("[BandedRidge] PCA active → uniform alpha =", self.alpha_video)
            return np.full(d, self.alpha_video)

        if self._X_video_full is None or self._X_audio_full is None:
            raise ValueError(
                "Pass X_video and X_audio to MultimodalLinearDeltaModel "
                "to use banded ridge without PCA."
            )

        d_video = self._X_video_full.shape[1]
        d_audio = self._X_audio_full.shape[1]

        if d_video + d_audio != d:
            raise ValueError(
                f"X_video ({d_video}) + X_audio ({d_audio}) ≠ "
                f"vector_list dim ({d})."
            )

        return np.concatenate([
            np.full(d_video, self.alpha_video),
            np.full(d_audio, self.alpha_audio),
        ])

    def fit(self):
        """Fit per-voxel banded ridge regression on training deltas."""
        Lambda = self._build_penalty()                       
        XtX = self.X_train.T @ self.X_train                 
        A = np.linalg.inv(XtX + np.diag(Lambda)) @ self.X_train.T  
        self.W = (A @ self.deltaY_train.T).T               

    def predict(self):
        self.deltaY_train_predicted = self.W @ self.X_train.T
        self.deltaY_test_predicted  = self.W @ self.X_test.T
        self.Y_train_predicted = np.delete(self.Y_train, -1, 1) + self.deltaY_train_predicted
        self.Y_test_predicted  = np.delete(self.Y_test,  -1, 1) + self.deltaY_test_predicted

    def evaluate(self, Y_test_predicted=None):
        if Y_test_predicted is None:
            self.MSE_train = utils.MSE(self.Y_train_predicted - np.delete(self.Y_train, 0, 1))
            self.MSE_test  = utils.MSE(self.Y_test_predicted  - np.delete(self.Y_test,  0, 1))
        else:
            return utils.MSE(Y_test_predicted - np.delete(self.Y_test, 0, 1))

    def repredict(self, W: np.ndarray) -> np.ndarray:
        return np.delete(self.Y_test, -1, 1) + W @ self.X_test.T


    def modality_mse(self) -> dict:
        """
        MSE when using only video or only audio weights.
        Only available without PCA (modalities must remain separable).
        """
        if not hasattr(self, "W"):
            raise RuntimeError("Call fit() first.")
        if self.pca is not None:
            print("[modality_mse] Not available with PCA (modalities are mixed).")
            return {"audio": None, "video": None, "multimodal": self.MSE_test}

        d_video = self._X_video_full.shape[1]

        W_video = self.W.copy(); W_video[:, d_video:] = 0.0
        W_audio = self.W.copy(); W_audio[:, :d_video] = 0.0

        return {
            "video":      self.evaluate(self.repredict(W_video)),
            "audio":      self.evaluate(self.repredict(W_audio)),
            "multimodal": self.MSE_test,
        }


    @staticmethod
    def grid_search_alpha(
        vector_list, X_video, X_audio, sub, dt, coef,
        alpha_video_grid, alpha_audio_grid,
        train_size=0.7,
    ) -> tuple[float, float, float]:
        """
        Exhaustive grid search over (alpha_video, alpha_audio).

        Returns
        -------
        best_alpha_video, best_alpha_audio, best_mse
        """
        best_mse = np.inf
        best_av, best_aa = None, None

        total = len(alpha_video_grid) * len(alpha_audio_grid)
        done = 0
        for av in alpha_video_grid:
            for aa in alpha_audio_grid:
                m = MultimodalLinearDeltaModel(
                    vector_list=vector_list,
                    sub=sub, dt=dt, coef=coef,
                    alpha_video=av, alpha_audio=aa,
                    X_video=X_video, X_audio=X_audio,
                    pca_components=None,
                    train_size=train_size,
                )
                m.fit(); m.predict(); m.evaluate()
                done += 1
                print(f"  [{done}/{total}] α_v={av:.0e}  α_a={aa:.0e}  "
                      f"MSE={m.MSE_test:.3e}")
                if m.MSE_test < best_mse:
                    best_mse = m.MSE_test
                    best_av, best_aa = av, aa

        print(f"\n→ Best: α_video={best_av:.0e}  α_audio={best_aa:.0e}  MSE={best_mse:.3e}")
        return best_av, best_aa, best_mse