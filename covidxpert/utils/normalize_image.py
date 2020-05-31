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
    delta = image.max() - image.min()
    # If the image has only zeros, we don't renormalize it.
    if np.isclose(delta, 0):
        return image
    return (((image - image.min()) / (delta))*255).astype(np.uint8)
