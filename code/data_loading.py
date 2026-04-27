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
* :func:`brain_mask` — Otsu + temporal-variance brain mask used by overlay
  visualisations to drop background voxels.
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

    def encode_video(
        self,
        video_path: str,
        batch_size: int = 32,
        chunk_dir: str | None = None,
        chunk_size: int = 512,
    ) -> np.ndarray:
        """Return ``(n_frames, 768)`` ViT embeddings, streaming frames with bounded RAM.

        Frames are decoded one at a time and forwarded through ViT in
        micro-batches of ``batch_size``, so peak memory is bounded by a single
        batch of decoded frames rather than the whole video.

        When ``chunk_dir`` is supplied, partial embeddings are flushed to
        ``chunk_dir/chunk_NNNN.npy`` every ``chunk_size`` frames.  Re-running
        with the same ``chunk_dir`` automatically resumes from the next frame,
        so a crash mid-encoding does not lose progress.

        Parameters
        ----------
        video_path : str
            Path to a video file readable by OpenCV.
        batch_size : int, default 32
            Number of frames forwarded through ViT in one tensor batch.
        chunk_dir : str, optional
            If set, intermediate embeddings are saved as
            ``chunk_dir/chunk_NNNN.npy`` and re-runs resume from the last
            saved chunk.  If ``None``, embeddings are kept in memory only.
        chunk_size : int, default 512
            Number of frames per on-disk chunk file.  Ignored when
            ``chunk_dir`` is ``None``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_frames, 768)``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        # ----- Resume from any existing chunks -----
        chunk_files: list[str] = []
        skip = 0
        next_chunk_idx = 0
        if chunk_dir is not None:
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_files = sorted(
                os.path.join(chunk_dir, f)
                for f in os.listdir(chunk_dir)
                if f.startswith("chunk_") and f.endswith(".npy")
            )
            for f in chunk_files:
                skip += np.load(f, mmap_mode="r").shape[0]
            next_chunk_idx = len(chunk_files)
            for _ in range(skip):
                if not cap.read()[0]:
                    break
            if skip:
                print(
                    f"[VideoEncoder] Resuming after {skip} encoded frames "
                    f"({next_chunk_idx} chunk(s) on disk)."
                )

        # ----- Streaming encode -----
        pil_buf: list[Image.Image] = []
        emb_buf: list[np.ndarray] = []

        def _forward() -> None:
            """Forward the current PIL buffer through ViT and clear it."""
            if not pil_buf:
                return
            with torch.no_grad():
                batch = torch.stack([self.preprocess(f) for f in pil_buf])
                emb_buf.append(self.model(batch).numpy())
            pil_buf.clear()

        def _flush_chunk() -> None:
            """Persist the current embedding buffer as a single chunk file."""
            nonlocal next_chunk_idx
            if chunk_dir is None or not emb_buf:
                return
            merged = np.concatenate(emb_buf, axis=0)
            path = os.path.join(chunk_dir, f"chunk_{next_chunk_idx:04d}.npy")
            np.save(path, merged)
            chunk_files.append(path)
            emb_buf.clear()
            next_chunk_idx += 1

        with tqdm(total=total, initial=skip, desc="Encoding video") as pbar:
            ok, frame = cap.read()
            while ok:
                pil_buf.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
                if len(pil_buf) >= batch_size:
                    _forward()
                    pbar.update(batch_size)
                    if (
                        chunk_dir is not None
                        and sum(e.shape[0] for e in emb_buf) >= chunk_size
                    ):
                        _flush_chunk()
                ok, frame = cap.read()

            if pil_buf:
                n = len(pil_buf)
                _forward()
                pbar.update(n)

        _flush_chunk()  # Last partial chunk, if any.
        cap.release()

        # ----- Re-assemble final array -----
        if chunk_dir is not None:
            return np.concatenate(
                [np.load(f) for f in sorted(chunk_files)], axis=0
            )
        return np.concatenate(emb_buf, axis=0)


def get_video_encoding(
    video_path: str | None = None,
    cache_path: str | None = None,
    batch_size: int = 32,
    chunk_dir: str | None = None,
    chunk_size: int = 512,
) -> np.ndarray:
    """Return ``(n_frames, 768)`` ViT-B/16 embeddings, using ``cache_path`` if available.

    For videos that do not fit in RAM, pass ``chunk_dir``: intermediate
    embeddings are flushed there as ``chunk_NNNN.npy`` files and re-runs
    automatically resume from the last saved chunk.

    Parameters
    ----------
    video_path : str, optional
        Defaults to ``<cwd>/src/Film stimulus.mp4``.
    cache_path : str, optional
        Path to a ``.npy`` file used to cache the final embeddings.
    batch_size : int, default 32
        Forward batch size.
    chunk_dir : str, optional
        If set, embeddings are flushed to disk every ``chunk_size`` frames
        during encoding.  Re-runs resume from the last saved chunk.
    chunk_size : int, default 512
        Frames per on-disk chunk.  Ignored when ``chunk_dir`` is ``None``.
    """
    if video_path is None:
        video_path = os.path.join(os.getcwd(), "src", "Film stimulus.mp4")

    if cache_path is not None and os.path.exists(cache_path):
        print(f"[VideoEncoder] Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print("[VideoEncoder] Encoding video with ViT-B/16 …")
    vectors = VideoEncoder().encode_video(
        video_path,
        batch_size=batch_size,
        chunk_dir=chunk_dir,
        chunk_size=chunk_size,
    )

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


def get_audio_encoding(
    sr: int = 44100,
    n_mfcc: int = 15,
    audio_path: str | None = None,
) -> np.ndarray:
    """Return ``(T, n_mfcc)`` standardised MFCC features for the film audio.

    Lightweight legacy extractor, kept for backwards compatibility with the
    audio-only baseline notebook.  For multimodal experiments prefer
    :func:`get_rich_audio_encoding`, which produces a much richer
    representation closer in capacity to the 768-d ViT video stream.

    Parameters
    ----------
    sr : int, default 44100
        Target sampling rate.  The audio is loaded directly at this rate.
    n_mfcc : int, default 15
        Number of MFCC coefficients per frame.
    audio_path : str, optional
        Defaults to ``<cwd>/src/Film stimulus.mp3``.
    """
    if audio_path is None:
        audio_path = os.path.join(os.getcwd(), "src", "Film stimulus.mp3")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = scale(mfcc, axis=1)
    return mfcc.T


def get_rich_audio_encoding(
    sr: int = 22050,
    n_mfcc: int = 40,
    *,
    audio_path: str | None = None,
    hop_length: int = 512,
    n_fft: int = 2048,
    include_deltas: bool = True,
    include_chroma: bool = True,
    include_spectral_contrast: bool = True,
    include_spectral_summary: bool = True,
    include_rms: bool = True,
    include_zcr: bool = True,
    standardise: bool = True,
) -> np.ndarray:
    """Multi-family per-frame audio features for the film soundtrack.

    Produces a much richer per-frame vector than :func:`get_audio_encoding`
    by concatenating several librosa feature families.  Each enabled flag
    appends a block:

    +--------------------------+----------+------------------------------------+
    | Block                    | Dim      | Description                        |
    +==========================+==========+====================================+
    | MFCC                     | n_mfcc   | Cepstral envelope                  |
    +--------------------------+----------+------------------------------------+
    | + Δ MFCC, ΔΔ MFCC        | 2·n_mfcc | First / second time derivatives    |
    +--------------------------+----------+------------------------------------+
    | Chroma (STFT)            | 12       | Tonal / pitch-class energy         |
    +--------------------------+----------+------------------------------------+
    | Spectral contrast        | 7        | Per-octave peak − valley           |
    +--------------------------+----------+------------------------------------+
    | Centroid + bandwidth + rolloff | 3 | "Brightness" / spectral shape      |
    +--------------------------+----------+------------------------------------+
    | RMS                      | 1        | Frame energy (loudness proxy)      |
    +--------------------------+----------+------------------------------------+
    | ZCR                      | 1        | Zero-crossing rate (voiced cue)    |
    +--------------------------+----------+------------------------------------+

    With every flag enabled and ``n_mfcc=40`` the output has **144 features
    per frame** (vs 15 from the legacy MFCC extractor).

    Parameters
    ----------
    sr : int, default 22050
        Target sampling rate.  ``22050`` Hz (librosa default) is enough for
        all features below; bump to 44100 if you want to retain
        higher-frequency content at 2× the storage cost.
    n_mfcc : int, default 40
        Number of MFCC coefficients (and, if ``include_deltas``, of each
        Δ and ΔΔ block).
    audio_path : str, optional
        Defaults to ``<cwd>/src/Film stimulus.mp3``.
    hop_length, n_fft : int
        STFT parameters.  ``hop_length=512`` at ``sr=22050`` gives a ~43 Hz
        frame rate, ample for the 25 fps target after ``_resample_to_grid``.
    include_deltas : bool, default True
        If ``True`` append Δ MFCC and ΔΔ MFCC.  This is the single biggest
        gain for forecasting BOLD *deltas*, since the BOLD signal reflects
        *changes* in neural activity.
    include_chroma, include_spectral_contrast, include_spectral_summary,
        include_rms, include_zcr : bool
        Toggle the corresponding feature family.
    standardise : bool, default True
        If ``True`` each feature dimension is z-scored over time, matching
        the convention of :func:`get_audio_encoding`.

    Returns
    -------
    np.ndarray
        Per-frame feature matrix of shape ``(T, D)``.  ``D`` is the sum of
        the enabled blocks; ``T`` is set by ``hop_length`` and the audio
        duration (~43 Hz with the defaults).
    """
    if audio_path is None:
        audio_path = os.path.join(os.getcwd(), "src", "Film stimulus.mp3")

    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    blocks: list[np.ndarray] = []
    block_names: list[str] = []

    # --- MFCC (+ Δ, ΔΔ) ---
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft,
    )
    blocks.append(mfcc)
    block_names.append(f"mfcc({n_mfcc})")
    if include_deltas:
        blocks.append(librosa.feature.delta(mfcc, order=1))
        blocks.append(librosa.feature.delta(mfcc, order=2))
        block_names.append(f"Δmfcc({n_mfcc})")
        block_names.append(f"ΔΔmfcc({n_mfcc})")

    # --- Chroma (tonal / pitch class) ---
    if include_chroma:
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=hop_length, n_fft=n_fft,
        )
        blocks.append(chroma)
        block_names.append("chroma(12)")

    # --- Spectral contrast (speech vs music cue) ---
    if include_spectral_contrast:
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=hop_length, n_fft=n_fft,
        )
        blocks.append(contrast)
        block_names.append(f"contrast({contrast.shape[0]})")

    # --- Spectral summary: centroid + bandwidth + rolloff ---
    if include_spectral_summary:
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length, n_fft=n_fft,
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop_length, n_fft=n_fft,
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=hop_length, n_fft=n_fft,
        )
        blocks.extend([centroid, bandwidth, rolloff])
        block_names.append("centroid+bandwidth+rolloff(3)")

    # --- RMS energy (loudness proxy) ---
    if include_rms:
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        blocks.append(rms)
        block_names.append("rms(1)")

    # --- Zero-crossing rate ---
    if include_zcr:
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
        blocks.append(zcr)
        block_names.append("zcr(1)")

    # Some librosa features can differ by 1 frame depending on padding;
    # clip to the shortest block so concatenation is safe.
    n_frames = min(b.shape[1] for b in blocks)
    blocks = [b[:, :n_frames] for b in blocks]
    feats = np.concatenate(blocks, axis=0)  # (D, T)

    if standardise:
        feats = scale(feats, axis=1)

    print(
        f"[Audio] {feats.shape[0]}-d features at {feats.shape[1]} frames "
        f"(~{feats.shape[1] / max(librosa.get_duration(y=y, sr=sr), 1e-9):.1f} Hz)\n"
        f"        composition: {', '.join(block_names)}"
    )
    return feats.T


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


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's threshold: maximise between-class variance over a histogram.

    Pure-numpy implementation so the project does not need ``scikit-image``.

    Parameters
    ----------
    values : np.ndarray
        Any-shape numeric array.  NaN / inf entries are ignored.
    n_bins : int, default 256
        Histogram resolution.

    Returns
    -------
    float
        Threshold ``t`` such that ``values > t`` separates foreground from
        background under the Otsu criterion.
    """
    flat = np.asarray(values).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0

    hist, bin_edges = np.histogram(flat, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    cum_hist = np.cumsum(hist).astype(np.float64)
    cum_mean = np.cumsum(hist * bin_centers)
    total = cum_hist[-1]
    grand_mean = cum_mean[-1]

    w1 = cum_hist
    w2 = total - cum_hist
    valid = (w1 > 0) & (w2 > 0)
    if not valid.any():
        return float(bin_centers[0])

    mean1 = np.where(w1 > 0, cum_mean / np.maximum(w1, 1.0), 0.0)
    mean2 = np.where(w2 > 0, (grand_mean - cum_mean) / np.maximum(w2, 1.0), 0.0)
    sigma_b_squared = w1 * w2 * (mean1 - mean2) ** 2
    sigma_b_squared[~valid] = -np.inf
    return float(bin_centers[int(np.argmax(sigma_b_squared))])


def brain_mask(bold: np.ndarray, var_quantile: float = 0.10) -> np.ndarray:
    """Build a 3-D brain mask from a 4-D BOLD volume.

    Combines two simple criteria:

    1. **Otsu** on the temporal-mean intensity to drop air / background.
    2. **Variance threshold**: within the Otsu-passed voxels, drop the
       lowest ``var_quantile`` fraction by temporal variance — these tend
       to be skull / static-noise voxels with no real BOLD signal.

    Parameters
    ----------
    bold : np.ndarray
        4-D array of shape ``(d1, d2, d3, n_TR)``.
    var_quantile : float, default 0.10
        Fraction of low-variance voxels to drop.  Set to ``0`` to disable
        the variance step.

    Returns
    -------
    np.ndarray of dtype bool, shape ``(d1, d2, d3)``.
    """
    if bold.ndim != 4:
        raise ValueError(
            f"brain_mask expects a 4-D BOLD volume; got shape {bold.shape}"
        )

    mean_bold = bold.mean(axis=-1)
    var_bold = bold.var(axis=-1)

    thr_mean = _otsu_threshold(mean_bold)
    mask = mean_bold > thr_mean

    if var_quantile > 0 and mask.any():
        var_thr = np.quantile(var_bold[mask], var_quantile)
        mask &= var_bold > var_thr

    return mask


class Sub:
    """A single subject with a 4-D BOLD volume from the film-stimulus run.

    The expected on-disk layout (BIDS) under ``data_root`` is::

        <data_root>/sub-<id>/ses-mri3t/func/sub-<id>_ses-mri3t_task-film_run-1_bold.nii.gz

    Path resolution order for ``data_root``:

    1. The ``data_root`` argument passed to :meth:`__init__`, if not ``None``.
    2. The class-level :attr:`Sub.DATA_ROOT`, if set.
    3. ``os.getcwd()`` as a last-resort fallback.

    Examples
    --------
    Set the dataset root once at the top of a notebook::

        Sub.DATA_ROOT = "/data/ds003688"
        sub_a = Sub("22")
        sub_b = Sub("31")

    Or pass it explicitly per-instance::

        sub = Sub("22", data_root="/data/ds003688")

    Attributes
    ----------
    number : str
        Two-digit subject identifier.
    data_root : str
        Resolved root used to build :attr:`path`.
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

    #: Optional class-level default for ``data_root``.  Set once per session
    #: (e.g. at the top of a notebook) to avoid passing the path to every
    #: ``Sub(...)`` call.
    DATA_ROOT: str | None = None

    subs_with_fmri = [
        "04", "07", "08", "09", "11", "13", "14", "15", "16", "18",
        "22", "24", "27", "28", "29", "31", "35", "41", "43", "44",
        "45", "46", "47", "51", "52", "53", "55", "56", "60", "62",
    ]

    def __init__(
        self,
        number: str,
        data_root: str | os.PathLike | None = None,
    ) -> None:
        """Load the BOLD volume for subject ``number``.

        Parameters
        ----------
        number : str
            Two-digit subject identifier (must be in :attr:`subs_with_fmri`).
        data_root : str or os.PathLike, optional
            Root directory of the BIDS dataset.  Falls back to
            :attr:`Sub.DATA_ROOT` and then to ``os.getcwd()``.

        Raises
        ------
        ValueError
            If ``number`` is not in the list of subjects with available fMRI
            data.
        FileNotFoundError
            If the resolved BOLD file does not exist (raised by ``nibabel``).
        """
        if number not in Sub.subs_with_fmri:
            raise ValueError(f"Subject {number} has no fMRI scans available.")
        self.number = number

        if data_root is None:
            data_root = (
                Sub.DATA_ROOT if Sub.DATA_ROOT is not None else os.getcwd()
            )
        self.data_root = str(data_root)

        self.path = os.path.join(
            self.data_root,
            f"sub-{self.number}",
            "ses-mri3t",
            "func",
            f"sub-{self.number}_ses-mri3t_task-film_run-1_bold.nii.gz",
        )
        self.scan = nib.load(self.path)
        self.data = self.scan.get_fdata()
        self.tensor = torch.tensor(self.data)
        self.tensor_np = self.tensor.numpy()
