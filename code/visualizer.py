"""Visualisation utilities for fMRI prediction models.

Renders 2-D slices of test / predicted / difference / delta volumes, animates
slices into GIFs, and plots distributions of the learnt weight matrix.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from PIL import Image


# Filename suffixes that contain signed data and should not be clipped to [0, 1].
_SIGNED_SUFFIXES = (
    "-difference.png",
    "-recovered-difference.png",
    "-delta.png",
    "-recovered-delta.png",
)


class Visualizer:
    """Render slices, GIFs and weight distributions for a fitted model.

    Parameters
    ----------
    model : LinearModel | LinearDeltaModel | MultimodalLinearDeltaModel
        A fitted model exposing ``Y_test``, ``Y_test_predicted`` and
        ``_d1`` / ``_d2`` / ``_d3`` (the spatial dimensions of the
        downsampled BOLD volume).
    """

    def __init__(self, model) -> None:
        self.model = model
        self.figures = "figures_delta" if self.model.delta else "figures"
        self.filename = (
            f"sub-{self.model.sub.number}-{self.model.dt}-{self.model.coef}"
        )

        if hasattr(self.model, "alpha") and self.model.alpha > 0:
            self.filename += f"-{self.model.alpha}"
        elif hasattr(self.model, "alpha_video"):
            self.filename += (
                f"-av{self.model.alpha_video}-aa{self.model.alpha_audio}"
            )

    def show_scan_slice(
        self,
        scan,
        scan_number: int,
        dim: int,
        slice: int,
        title: str,
        filename_end: str,
        mask=None,
    ) -> None:
        """Render and save a 2-D slice of a 3-D scan along ``dim`` at index ``slice``.

        Parameters
        ----------
        scan : np.ndarray
            Volume of shape ``(_d1, _d2, _d3)``.
        scan_number : int
            Index appended to the filename (typically the test-set TR index).
        dim : int
            Slicing axis: 0 (sagittal-like), 1 (coronal-like), 2 (axial-like).
        slice : int
            Index along ``dim`` to render.
        title : str
            Title printed before showing the figure.
        filename_end : str
            Suffix appended to the saved image filename.
        mask : np.ndarray, optional
            Boolean volume of the same shape as ``scan``; non-zero voxels are
            overlaid in red.
        """
        if dim == 0:
            scan_slice = scan[slice, :, :].T
            slices = f"-{slice}-_-_"
            scan_slice_masked = mask[slice, :, :].T if mask is not None else None
        elif dim == 1:
            scan_slice = scan[:, slice, :].T
            slices = f"-_-{slice}-_"
            scan_slice_masked = mask[:, slice, :].T if mask is not None else None
        elif dim == 2:
            scan_slice = scan[:, :, slice].T
            slices = f"-_-_-{slice}"
            scan_slice_masked = mask[:, :, slice].T if mask is not None else None
        else:
            raise ValueError(f"dim must be 0, 1 or 2, got {dim}.")

        slice_filename = (
            self.filename + f"-{scan_number}" + slices + filename_end
        )
        self.last_slices = slices
        self.last_slice_filename = slice_filename

        folder_path = os.path.join(
            os.path.dirname(os.getcwd()), self.figures, self.filename
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(title)

        if filename_end in _SIGNED_SUFFIXES:
            plt.imshow(scan_slice, cmap="gray", origin="lower")
        else:
            plt.imshow(scan_slice, cmap="gray", origin="lower", vmin=0, vmax=1)

        plt.colorbar()

        if scan_slice_masked is not None:
            cmap = colors.ListedColormap(["black", "red"])
            plt.imshow(scan_slice_masked, cmap=cmap, origin="lower", alpha=0.3)

        plt.savefig(
            os.path.join(folder_path, slice_filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _show_scan_test_slice(self, scan: int, dim: int, slice: int, mask=None) -> None:
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3)
        )
        self.show_scan_slice(scan_test, scan, dim, slice, "TEST", "-test.png", mask)

    def _show_scan_predicted_slice(self, scan: int, dim: int, slice: int, mask=None) -> None:
        scan_predicted = self.model.Y_test_predicted.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3)
        )
        self.show_scan_slice(
            scan_predicted, scan, dim, slice, "PREDICTED", "-predicted.png", mask
        )

    def _show_scan_difference_slice(self, scan: int, dim: int, slice: int, mask=None) -> None:
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3)
        )
        scan_predicted = self.model.Y_test_predicted.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3)
        )
        scan_difference = abs(scan_test - scan_predicted)
        self.show_scan_slice(
            scan_difference, scan, dim, slice, "DIFFERENCE", "-difference.png", mask
        )

    def _show_recovered_scan_test_slice(self, scan: int, dim: int, slice: int) -> None:
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3)
        )
        self.show_scan_slice(
            scan_test, scan, dim, slice, "TEST", "-recovered-test.png"
        )

    def _show_recovered_scan_predicted_slice(self, scan: int, dim: int, slice: int) -> None:
        scan_predicted = (
            self.model.Y_test.T[0]
            + np.sum(self.model.deltaY_test_predicted.T[:scan], axis=0)
        ).reshape((self.model._d1, self.model._d2, self.model._d3))
        self.show_scan_slice(
            scan_predicted, scan, dim, slice, "PREDICTED", "-recovered-predicted.png"
        )

    def _show_recovered_scan_delta_slice(self, scan: int, dim: int, slice: int) -> None:
        scan_delta = np.sum(
            self.model.deltaY_test_predicted.T[:scan], axis=0
        ).reshape((self.model._d1, self.model._d2, self.model._d3))
        self.show_scan_slice(
            scan_delta, scan, dim, slice, "DELTA", "-recovered-delta.png"
        )

    def _show_recovered_scan_difference_slice(self, scan: int, dim: int, slice: int) -> None:
        scan_test = self.model.Y_test.T[scan].reshape(
            (self.model._d1, self.model._d2, self.model._d3)
        )
        scan_predicted = (
            self.model.Y_test.T[0]
            + np.sum(self.model.deltaY_test_predicted.T[:scan], axis=0)
        ).reshape((self.model._d1, self.model._d2, self.model._d3))
        scan_difference = abs(scan_test - scan_predicted)
        self.show_scan_slice(
            scan_difference, scan, dim, slice,
            "DIFFERENCE", "-recovered-difference.png",
        )

    def show_scan_slices(self, scan: int, dim: int, slice: int, mask=None) -> None:
        """Render test / predicted / difference slices side by side."""
        self._show_scan_test_slice(scan, dim, slice, mask)
        self._show_scan_predicted_slice(scan, dim, slice, mask)
        self._show_scan_difference_slice(scan, dim, slice, mask)

    def show_recovered_scan_slices(self, scan: int, dim: int, slice: int) -> None:
        """Render reconstructed slices for delta models (cumulative deltas)."""
        if not self.model.delta:
            raise ValueError(
                "show_recovered_scan_slices is only available for delta models "
                "(model.delta == True)."
            )
        self._show_recovered_scan_test_slice(scan, dim, slice)
        self._show_recovered_scan_predicted_slice(scan, dim, slice)
        self._show_recovered_scan_delta_slice(scan, dim, slice)
        self._show_recovered_scan_difference_slice(scan, dim, slice)

    def get_slice_gif(self, dim: int, slice: int, filename_end: str) -> None:
        """Animate the chosen slice across all test TRs into a GIF."""
        frames = []
        length = (
            self.model.Y_test.shape[1]
            if not self.model.delta
            else self.model.deltaY_test.shape[1]
        )
        for scan in range(length):
            if filename_end == "-test.gif":
                self._show_scan_test_slice(scan, dim, slice)
            elif filename_end == "-predicted.gif":
                self._show_scan_predicted_slice(scan, dim, slice)
            elif filename_end == "-recovered-test.gif":
                self._show_recovered_scan_test_slice(scan, dim, slice)
            elif filename_end == "-recovered-predicted.gif":
                self._show_recovered_scan_predicted_slice(scan, dim, slice)

            frame = Image.open(
                os.path.join(
                    os.path.dirname(os.getcwd()),
                    self.figures,
                    self.filename,
                    self.last_slice_filename,
                )
            )
            frames.append(frame)

        frames[0].save(
            os.path.join(
                os.path.dirname(os.getcwd()),
                self.figures,
                self.filename,
                "GIF-" + self.filename + self.last_slices + filename_end,
            ),
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0,
        )

    def get_test_slice_gif(self, dim: int, slice: int) -> None:
        """Animate the test-volume slice across all test TRs."""
        self.get_slice_gif(dim, slice, "-test.gif")

    def get_predicted_slice_gif(self, dim: int, slice: int) -> None:
        """Animate the predicted-volume slice across all test TRs."""
        self.get_slice_gif(dim, slice, "-predicted.gif")

    def get_recovered_test_slice_gif(self, dim: int, slice: int) -> None:
        """Animate the recovered (cumulative) test slice across all test TRs."""
        self.get_slice_gif(dim, slice, "-recovered-test.gif")

    def get_recovered_predicted_slice_gif(self, dim: int, slice: int) -> None:
        """Animate the recovered (cumulative) predicted slice across all test TRs."""
        self.get_slice_gif(dim, slice, "-recovered-predicted.gif")

    def show_voxel_weight_distribution(self, voxel: int) -> None:
        """Histogram of the weight-vector components for a single voxel."""
        ax = sns.histplot(
            self.model.W[voxel, :], element="poly", linewidth=0, kde=True, bins=20
        )
        ax.set(xlabel="Weights vector component value")
        ax.set(ylabel="Number of components")

    def show_mean_weight_distribution(self) -> None:
        """Histogram of the mean weight-vector components across all voxels."""
        W_mean_rows = np.mean(self.model.W, axis=0)
        ax = sns.histplot(
            W_mean_rows, element="poly", linewidth=0, kde=True, bins=30
        )
        ax.set(xlabel="Weights vector component value")
        ax.set(ylabel="Number of components")
        plt.grid(alpha=0.1)
