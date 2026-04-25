"""Project-wide utility functions."""

import numpy as np


def MSE(residual: np.ndarray) -> float:
    """Return the mean squared error of a residual array.

    The input is expected to already be the difference between predicted
    and true values (i.e. ``y_pred - y_true``); this function only takes
    the mean of the elementwise square.

    Parameters
    ----------
    residual : np.ndarray
        Pre-computed residuals.  Any shape; all elements are squared and
        averaged.

    Returns
    -------
    float
        Mean of ``residual ** 2``.
    """
    return np.mean(residual ** 2)
