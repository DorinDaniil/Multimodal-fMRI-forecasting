"""Project-wide utility functions."""

import numpy as np


def MSE(residual: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Return the mean squared error of a residual array.

    The input is expected to already be the difference between predicted
    and true values (i.e. ``y_pred - y_true``).

    Parameters
    ----------
    residual : np.ndarray
        Pre-computed residuals.  When ``mask`` is provided, the leading
        dimension is interpreted as voxels and the rest as time / extras.
    mask : np.ndarray of dtype bool, optional
        1-D boolean array selecting voxels to include in the average.  Its
        length must equal ``residual.shape[0]``.  When ``None`` (default),
        every element of ``residual`` is averaged — the original behaviour.

    Returns
    -------
    float
        Mean of ``residual ** 2`` over the selected entries.
    """
    if mask is None:
        return np.mean(residual ** 2)
    return np.mean(residual[mask] ** 2)
