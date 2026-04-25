"""Feature extraction and subject loading for the multimodal fMRI project.

This module provides:

* :class:`Sub` — wrapper around a 4-D BOLD volume from the Berezutskaya et al.
  (2022) film-stimulus dataset.
* :class:`VideoEncoder` and :func:`get_video_encoding` — per-frame ViT-B/16
  embeddings of the film stimulus.
* :func:`get_lowlevel_video_features` — low-level visual descriptors
  (brightness, contrast, motion energy) for early visual cortex baselines.
* :func:`get_audio_encoding` — standardised MFCC features of the soundtrack.
* :func:`get_multimodal_encoding` — resampled, normalised and concatenated
  audio + video feature matrix.
"""

import os

import cv2
import librosa
import nibabel as nib
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale
from tqdm import tqdm


class VideoEncoder:
    """Extract per-frame ViT-B/16 embeddings (768-d) from a video file."""

    def __init__(self) -> None:
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.preprocess = weights.transforms()
        self.model = torchvision.models.vit_b_16(weights=weights)
        self.model.heads = torch.nn.Identity()
        self.model.eval()

    def encode_video(self, video_path: str, batch_size: int = 32) -> np.ndarray:
        """Return ``(n_frames, 768)`` ViT embeddings for every frame.

        Parameters
        ----------
        video_path : str
            Path to a video file readable by OpenCV.
        batch_size : int, default 32
            Number of frames forwarded through ViT in one batch.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_frames, 768)``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames = []
        success, frame = cap.read()
        while success:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            success, frame = cap.read()
        cap.release()

        vectors = []
        with torch.no_grad():
            for i in tqdm(range(0, len(frames), batch_size), desc="Encoding video"):
                batch = torch.stack([self.preprocess(f) for f in frames[i:i + batch_size]])
                vectors.append(self.model(batch).numpy())

        return np.concatenate(vectors, axis=0)


def get_video_encoding(
    video_path: str | None = None,
    cache_path: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Return ``(n_frames, 768)`` ViT-B/16 embeddings, using ``cache_path`` if available.

    Parameters
    ----------
    video_path : str, optional
        Defaults to ``<cwd>/src/Film stimulus.mp4``.
    cache_path : str, optional
        Path to a ``.npy`` file used to cache the embeddings.
    batch_size : int, default 32
        Forward batch size.
    """
    if video_path is None:
        video_path = os.path.join(os.getcwd(), "src", "Film stimulus.mp4")

    if cache_path is not None and os.path.exists(cache_path):
        print(f"[VideoEncoder] Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print("[VideoEncoder] Encoding video with ViT-B/16 …")
    vectors = VideoEncoder().encode_video(video_path, batch_size=batch_size)

    if cache_path is not None:
        np.save(cache_path, vectors)
        print(f"[VideoEncoder] Saved to {cache_path}")

    return vectors


def get_lowlevel_video_features(
    video_path: str | None = None,
    cache_path: str | None = None,
) -> np.ndarray:
    """Extract low-level per-frame visual descriptors.

    Useful as a baseline for early visual cortex.  The returned features are
    mean brightness, std brightness (contrast), mean R / G / B channels, and
    frame-to-frame absolute pixel difference (motion energy).

    Parameters
    ----------
    video_path : str, optional
        Defaults to ``<cwd>/src/Film stimulus.mp4``.
    cache_path : str, optional
        Path to a ``.npy`` cache file.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_frames, 6)``.
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[LowLevel] Loading cached features from {cache_path}")
        return np.load(cache_path)

    if video_path is None:
        video_path = os.path.join(os.getcwd(), "src", "Film stimulus.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    features = []
    prev_gray = None
    success, frame = cap.read()
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        rgb = frame.astype(np.float32) / 255.0

        mean_bright = gray.mean()
        std_bright = gray.std()
        mean_r = rgb[:, :, 2].mean()
        mean_g = rgb[:, :, 1].mean()
        mean_b = rgb[:, :, 0].mean()

        motion = np.abs(gray - prev_gray).mean() if prev_gray is not None else 0.0

        features.append([mean_bright, std_bright, mean_r, mean_g, mean_b, motion])
        prev_gray = gray
        success, frame = cap.read()

    cap.release()
    arr = np.array(features)

    if cache_path is not None:
        np.save(cache_path, arr)
        print(f"[LowLevel] Saved to {cache_path}")

    print(f"[LowLevel] Extracted {arr.shape} features")
    return arr


def get_audio_encoding(sr: int = 44100, n_mfcc: int = 15) -> np.ndarray:
    """Return ``(T, n_mfcc)`` standardised MFCC features for the film audio.

    Parameters
    ----------
    sr : int, default 44100
        Target sampling rate.  The source ``Film stimulus.mp3`` is loaded at
        44100 Hz and (optionally) resampled.
    n_mfcc : int, default 15
        Number of MFCC coefficients per frame.
    """
    audio_path = os.path.join(os.getcwd(), "src", "Film stimulus.mp3")
    x, _ = librosa.load(audio_path, sr=44100)
    x = librosa.resample(y=x, orig_sr=44100, target_sr=sr)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfcc = scale(mfcc, axis=1)
    return mfcc.T


def _resample_to_grid(X: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Linearly interpolate every column of ``X`` onto the time grid ``t_target``."""
    t_src = np.linspace(0.0, 1.0, X.shape[0])
    out = np.empty((len(t_target), X.shape[1]))
    for d in range(X.shape[1]):
        out[:, d] = interp1d(
            t_src, X[:, d],
            kind="linear", bounds_error=False, fill_value="extrapolate",
        )(t_target)
    return out


def get_multimodal_encoding(
    video_features: np.ndarray | None = None,
    n_mfcc: int = 15,
    sr: int = 44100,
    target_fps: float = 25.0,
    duration_s: float = 390.0,
    video_path: str | None = None,
    video_cache_path: str | None = None,
    normalize: bool = True,
    video_pca: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build multimodal features by concatenating video and MFCC audio.

    Both modalities are resampled to ``target_fps``.  When ``video_pca`` is
    set, PCA is applied to the video features **before** concatenation with
    audio — this preserves the audio signal intact.

    Parameters
    ----------
    video_features : np.ndarray, optional
        Pre-computed ``(n_frames_src, d_video)`` video embeddings.  If
        ``None``, embeddings are computed via :func:`get_video_encoding`.
    n_mfcc : int, default 15
        Number of MFCC coefficients.
    sr : int, default 44100
        Audio sampling rate.
    target_fps : float, default 25.0
        Target frame rate for resampling.
    duration_s : float, default 390.0
        Duration of the stimulus in seconds.
    video_path, video_cache_path : str, optional
        Forwarded to :func:`get_video_encoding` when ``video_features`` is
        ``None``.
    normalize : bool, default True
        If ``True``, apply per-modality :class:`StandardScaler` after
        resampling.
    video_pca : int, optional
        If set, reduce video dimensionality to ``video_pca`` components
        before concatenation.

    Returns
    -------
    X_multi : np.ndarray
        Shape ``(n_target, d_video + n_mfcc)`` — concatenated features.
    X_video : np.ndarray
        Shape ``(n_target, d_video)`` — video features alone.
    X_audio : np.ndarray
        Shape ``(n_target, n_mfcc)`` — audio features alone.
    """
    if video_features is None:
        video_features = get_video_encoding(
            video_path=video_path, cache_path=video_cache_path,
        )

    n_target = int(target_fps * duration_s)
    t_target = np.linspace(0.0, 1.0, n_target)

    video_res = _resample_to_grid(video_features, t_target)
    audio_res = _resample_to_grid(get_audio_encoding(sr=sr, n_mfcc=n_mfcc), t_target)

    if video_pca is not None and video_pca < video_res.shape[1]:
        pca = PCA(n_components=video_pca, random_state=42)
        video_res = pca.fit_transform(video_res)
        explained = pca.explained_variance_ratio_.sum() * 100
        print(
            f"[Multimodal] Video PCA: {video_features.shape[1]} → {video_pca} components "
            f"({explained:.1f}% variance explained)"
        )

    if normalize:
        video_res = StandardScaler().fit_transform(video_res)
        audio_res = StandardScaler().fit_transform(audio_res)
        print("[Multimodal] Per-modality StandardScaler applied.")

    X_multi = np.concatenate([video_res, audio_res], axis=1)
    print(
        f"[Multimodal] video {video_res.shape} + audio {audio_res.shape} "
        f"→ multi {X_multi.shape}"
    )
    return X_multi, video_res, audio_res


class Sub:
    """A single subject with a 4-D BOLD volume from the film-stimulus run.

    The expected on-disk layout (BIDS, relative to ``cwd``) is::

        sub-<id>/ses-mri3t/func/sub-<id>_ses-mri3t_task-film_run-1_bold.nii.gz

    Attributes
    ----------
    number : str
        Two-digit subject identifier.
    path : str
        Resolved path to the subject's BOLD ``.nii.gz`` file.
    scan : nibabel.Nifti1Image
        The loaded NIfTI image.
    data : np.ndarray
        Raw BOLD data, shape ``(d1, d2, d3, n_TR)``.
    tensor : torch.Tensor
        ``data`` wrapped as a torch tensor.
    tensor_np : np.ndarray
        Numpy view of ``tensor`` (kept for backwards compatibility).
    """

    subs_with_fmri = [
        "04", "07", "08", "09", "11", "13", "14", "15", "16", "18",
        "22", "24", "27", "28", "29", "31", "35", "41", "43", "44",
        "45", "46", "47", "51", "52", "53", "55", "56", "60", "62",
    ]

    def __init__(self, number: str) -> None:
        """Load the BOLD volume for subject ``number``.

        Parameters
        ----------
        number : str
            Two-digit subject identifier (must be in :attr:`subs_with_fmri`).

        Raises
        ------
        ValueError
            If ``number`` is not in the list of subjects with available fMRI data.
        """
        if number not in Sub.subs_with_fmri:
            raise ValueError(f"Subject {number} has no fMRI scans available.")
        self.number = number
        self.path = os.path.join(
            os.getcwd(), f"sub-{self.number}",
            "ses-mri3t", "func",
            f"sub-{self.number}_ses-mri3t_task-film_run-1_bold.nii.gz",
        )
        self.scan = nib.load(self.path)
        self.data = self.scan.get_fdata()
        self.tensor = torch.tensor(self.data)
        self.tensor_np = self.tensor.numpy()
