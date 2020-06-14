import numpy as np
from numba import njit


@njit
def median_mask(image: np.ndarray, factor: float = 2) -> np.ndarray:
    """Return median-based binary mask.

    Parameters
    -----------------------
    image: np.ndarray,
        Image to be thresholded.
    factor: float,
        Divider factor for median threshold.

    Returns
    -----------------------
    Median-based binary mask.
    """
    return image > np.median(image)/factor
