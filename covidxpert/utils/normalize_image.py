import numpy as np


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
    return np.uint8(((image - image.min()) / (image.max() - image.min()))*255)
