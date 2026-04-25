import numpy as np
import os
import torch
import torchvision
import cv2
from PIL import Image
import nibabel as nib
from tqdm import tqdm

import librosa
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d



class VideoEncoder:
    """
    Extracts per-frame ViT-B/16 embeddings (768-d) from a video file.
    Frames are processed in batches for speed.
    """

    def __init__(self):
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.preprocess = weights.transforms()
        self.model = torchvision.models.vit_b_16(weights=weights)
        self.model.heads = torch.nn.Identity()   
        self.model.eval()

    def encode_video(self, video_path: str, batch_size: int = 32) -> np.ndarray:
        """Return (n_frames, 768) ViT embeddings for every frame of *video_path*."""
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
                batch = torch.stack([self.preprocess(f) for f in frames[i : i + batch_size]])
                vectors.append(self.model(batch).numpy())

        return np.concatenate(vectors, axis=0)   


def get_video_encoding(
    video_path: str | None = None,
    cache_path: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Return (n_frames, 768) ViT-B/16 embeddings, loading from cache if available."""
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
    """
    Extract low-level per-frame features that correlate with early visual cortex:
      - mean brightness
      - std brightness (contrast)
      - mean R, G, B channels
      - frame-to-frame absolute pixel difference (motion energy)

    Returns (n_frames, 6) array.
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

        if prev_gray is not None:
            motion = np.abs(gray - prev_gray).mean()
        else:
            motion = 0.0

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
    """Return (T, n_mfcc) standardised MFCC features for the film audio."""
    audio_path = os.path.join(os.getcwd(), "src", "Film stimulus.mp3")
    x, _ = librosa.load(audio_path, sr=44100)
    x = librosa.resample(y=x, orig_sr=44100, target_sr=sr)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    return mfcc.T   


def _resample_to_grid(X: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Linearly interpolate each column of X onto *t_target*."""
    t_src = np.linspace(0.0, 1.0, X.shape[0])
    out = np.empty((len(t_target), X.shape[1]))
    for d in range(X.shape[1]):
        out[:, d] = interp1d(t_src, X[:, d],
                             kind="linear", bounds_error=False,
                             fill_value="extrapolate")(t_target)
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
    """
    Build multimodal features by concatenating video and MFCC audio,
    both resampled to *target_fps*.

    Key fix: PCA within video only
    ------------------------------
    When *video_pca* is set, PCA is applied to video features **before**
    concatenation with audio.  This preserves the audio signal intact.

    Returns
    -------
    X_multi : (n_target, d_video + n_mfcc)  — concatenated features
    X_video : (n_target, d_video)            — video features alone
    X_audio : (n_target, n_mfcc)             — audio features alone
    """
    if video_features is None:
        video_features = get_video_encoding(
            video_path=video_path, cache_path=video_cache_path
        )

    n_target = int(target_fps * duration_s)
    t_target = np.linspace(0.0, 1.0, n_target)

    video_res = _resample_to_grid(video_features, t_target)
    audio_res = _resample_to_grid(get_audio_encoding(sr=sr, n_mfcc=n_mfcc), t_target)

    if video_pca is not None and video_pca < video_res.shape[1]:
        pca = PCA(n_components=video_pca, random_state=42)
        video_res = pca.fit_transform(video_res)
        expl = pca.explained_variance_ratio_.sum() * 100
        print(f"[Multimodal] Video PCA: 768 → {video_pca} components "
              f"({expl:.1f}% variance explained)")

    if normalize:
        video_res = StandardScaler().fit_transform(video_res)
        audio_res = StandardScaler().fit_transform(audio_res)
        print("[Multimodal] Per-modality StandardScaler applied.")

    X_multi = np.concatenate([video_res, audio_res], axis=1)
    print(f"[Multimodal] video {video_res.shape} + audio {audio_res.shape} "
          f"→ multi {X_multi.shape}")
    return X_multi, video_res, audio_res


class Sub:
    """Responsible for the subject and contains information about his data."""

    subs_with_fmri = [
        '04', '07', '08', '09', '11', '13', '14', '15', '16', '18',
        '22', '24', '27', '28', '29', '31', '35', '41', '43', '44',
        '45', '46', '47', '51', '52', '53', '55', '56', '60', '62',
    ]

    def __init__(self, number: str):
        if number not in Sub.subs_with_fmri:
            raise ValueError(f"У {number} испытуемого отсутствуют снимки фМРТ")
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