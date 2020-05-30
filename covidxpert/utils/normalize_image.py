import numpy as np
from numba import njit


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image between minimum and maximum value in a range from 0 to 255.

    Parameters
    ----------------------
    image: np.ndarray,
        The image to be normalize.

    Returns
    ----------------------
    Return the normalized image.
    """
    image = image.astype(np.float64)
    return (((image - image.min()) / (image.max() - image.min()))*255).astype(np.uint8)
