"""Brain-anatomy-aware overlays for fMRI prediction visualisations.

These helpers plot per-voxel scalar maps (e.g. the gating coefficient ``α``,
or per-voxel MSE improvements) on top of a grayscale BOLD underlay, masked
to brain voxels and optionally to "confident" voxels only.  This avoids the
noisy "everything is red" appearance you get when ``imshow`` is applied to
the whole volume bounding box without masking.

All functions are stateless plotting utilities — they take pre-computed
arrays / masks and just render them.  The masks themselves come from
:func:`data_loading.brain_mask` (anatomy) and
:meth:`gated_models.GatedMultimodalLinearDeltaModel.confidence_mask`
(prediction quality).
"""

import matplotlib.pyplot as plt
import numpy as np


def _slice_3d(volume: np.ndarray, dim: int, idx: int) -> np.ndarray:
    """Return a 2-D slice of a 3-D volume oriented for ``imshow``."""
    if dim == 0:
        return volume[idx, :, :].T
    if dim == 1:
        return volume[:, idx, :].T
    if dim == 2:
        return volume[:, :, idx].T
    raise ValueError(f"dim must be 0, 1 or 2; got {dim}")


def _default_slice_specs(volume_shape: tuple) -> list:
    """Mid-volume sagittal / coronal / axial slice specs."""
    d1, d2, d3 = volume_shape
    return [
        (0, d1 // 2, "Sagittal (x={})"),
        (1, d2 // 2, "Coronal (y={})"),
        (2, d3 // 2, "Axial (z={})"),
    ]


def _make_transparent_cmap(name: str):
    """Return a copy of cmap ``name`` with NaN cells fully transparent."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(alpha=0.0)
    return cmap


def show_alpha_overlay(
    alpha_3d: np.ndarray,
    bold_underlay: np.ndarray,
    mask_3d: np.ndarray,
    slice_specs: list | None = None,
    title: str | None = None,
    cmap_overlay: str = "RdBu_r",
    overlay_alpha: float = 0.85,
    figsize: tuple = (15, 4),
    cbar_label: str = r"$\alpha$  (red = video, blue = audio)",
    save_path: str | None = None,
) -> None:
    """Overlay a per-voxel ``α`` map on a grayscale BOLD underlay.

    Voxels outside ``mask_3d`` show only the underlay (so the background
    "red haze" disappears and you actually see the anatomy).

    Parameters
    ----------
    alpha_3d : np.ndarray, shape (d1, d2, d3)
        Per-voxel quantity to plot in colour (assumed in ``[0, 1]``).
    bold_underlay : np.ndarray, shape (d1, d2, d3)
        Grayscale background, typically the temporal-mean BOLD.
    mask_3d : np.ndarray of dtype bool, shape (d1, d2, d3)
        Where ``True``, the overlay is drawn; elsewhere only the underlay
        shows through.  Combine ``brain_mask`` AND ``confidence_mask`` here.
    slice_specs : list of (dim, idx, title_template), optional
        Defaults to mid-volume sagittal / coronal / axial.
    title : str, optional
        Figure-level title.
    cmap_overlay : str, default 'RdBu_r'
    overlay_alpha : float, default 0.85
        Transparency of the coloured overlay (0 = invisible, 1 = opaque).
    figsize : tuple
    cbar_label : str
    save_path : str, optional
        If set, ``savefig`` to this path before ``show``.
    """
    if slice_specs is None:
        slice_specs = _default_slice_specs(alpha_3d.shape)

    cmap = _make_transparent_cmap(cmap_overlay)

    fig, axes = plt.subplots(1, len(slice_specs), figsize=figsize)
    if len(slice_specs) == 1:
        axes = [axes]

    last_im = None
    for ax, (dim, idx, title_template) in zip(axes, slice_specs):
        bg = _slice_3d(bold_underlay, dim, idx)
        fg = _slice_3d(alpha_3d, dim, idx)
        m = _slice_3d(mask_3d, dim, idx)

        ax.imshow(bg, cmap="gray", origin="lower")
        masked_fg = np.where(m, fg, np.nan)
        last_im = ax.imshow(
            masked_fg, cmap=cmap, origin="lower",
            vmin=0.0, vmax=1.0, alpha=overlay_alpha,
        )
        ax.set_title(title_template.format(idx))
        ax.axis("off")

    fig.colorbar(last_im, ax=axes, label=cbar_label, shrink=0.8)
    if title:
        fig.suptitle(title, fontsize=13)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def show_signed_overlay(
    values_3d: np.ndarray,
    bold_underlay: np.ndarray,
    mask_3d: np.ndarray,
    slice_specs: list | None = None,
    title: str | None = None,
    cbar_label: str = "value",
    cmap: str = "RdBu_r",
    overlay_alpha: float = 0.85,
    vmax: float | None = None,
    vmax_quantile: float = 0.95,
    figsize: tuple = (15, 4),
    save_path: str | None = None,
) -> None:
    """Overlay a signed per-voxel map (e.g. MSE improvement) on a BOLD underlay.

    The colour scale is centred at zero with symmetric limits ``±vmax``.
    If ``vmax`` is ``None`` it is taken as the ``vmax_quantile`` quantile
    of ``|values|`` inside the mask, so a few extreme outliers do not
    saturate the colour scale.
    """
    if slice_specs is None:
        slice_specs = _default_slice_specs(values_3d.shape)

    if vmax is None:
        if mask_3d.any():
            vmax = float(
                np.nanpercentile(np.abs(values_3d[mask_3d]), vmax_quantile * 100)
            )
        else:
            vmax = float(np.nanpercentile(np.abs(values_3d), vmax_quantile * 100))
        if vmax <= 0:
            vmax = 1.0

    cmap_t = _make_transparent_cmap(cmap)

    fig, axes = plt.subplots(1, len(slice_specs), figsize=figsize)
    if len(slice_specs) == 1:
        axes = [axes]

    last_im = None
    for ax, (dim, idx, title_template) in zip(axes, slice_specs):
        bg = _slice_3d(bold_underlay, dim, idx)
        fg = _slice_3d(values_3d, dim, idx)
        m = _slice_3d(mask_3d, dim, idx)

        ax.imshow(bg, cmap="gray", origin="lower")
        masked_fg = np.where(m, fg, np.nan)
        last_im = ax.imshow(
            masked_fg, cmap=cmap_t, origin="lower",
            vmin=-vmax, vmax=vmax, alpha=overlay_alpha,
        )
        ax.set_title(title_template.format(idx))
        ax.axis("off")

    fig.colorbar(last_im, ax=axes, label=cbar_label, shrink=0.8)
    if title:
        fig.suptitle(title, fontsize=13)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
