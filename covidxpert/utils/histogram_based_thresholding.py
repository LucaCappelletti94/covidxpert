import numpy as np
from ..blur_bbox.blur_bbox import build_slice, count_from_right_side


def histogram_based_vertical_thresholding(image: np.ndarray, mask: np.ndarray, percentage: float = 0.6) -> np.ndarray:
    """Return the filtered image according to the histograms of the given mask.

    Parameters
    ---------------------
    image:np.ndarray,
        The image to be filtered.
    mask:np.ndarray,
        The mask obtained from image to use for filtering.
    percentage:float,
        Controls the sensitivity of the threshold

    Returns
    ---------------------
    Filtered image.
    """
    mask = mask.max() - mask
    y = mask.mean(axis=1)
    median = np.median(y[mask.any(axis=1)]) / 10
    vertical_slice = build_slice(
        left=0,
        right=int(-count_from_right_side(y < median) * percentage),
        maximum=y.size
    )

    return image[vertical_slice]
