import numpy as np
from numba import njit


@njit
def median_mask(image: np.ndarray) -> np.ndarray:
    """Return median-based binary mask.

    Parameters
    -----------------------
    image: np.ndarray,
        Image to be thresholded.

    Returns
    -----------------------
    Median-based binary mask.
    """
    return image > np.median(image)/2
