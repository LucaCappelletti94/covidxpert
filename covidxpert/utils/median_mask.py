import numpy as np


def median_mask(image: np.ndarray, median: float = None, factor: float = 2) -> np.ndarray:
    """Return median-based binary mask.

    Parameters
    -----------------------
    image: np.ndarray,
        Image to be thresholded.
    median: float = None,
        The median to use. If not provided (default), it is compute using
        the provided image.
    factor: float,
        Divider factor for median threshold.

    Returns
    -----------------------
    Median-based binary mask.
    """
    if median is None:
        median = np.median(image)
    return image > median/factor
