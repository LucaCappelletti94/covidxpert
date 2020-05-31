import cv2
import numpy as np
from numba import njit


def fill_lower_max(image: np.ndarray, lower_padding: int = 50) -> np.ndarray:
    """Return image with lower part filled in as the local maximum.

    Parameters
    -----------------------
    image: np.ndarray,
        The image to fill in.
    lower_padding: int = 50,
        The lowe part of the image to skip, tipically since there are line
        artefacts on the lower part.

    Returns
    -----------------------
    The image with lower part filled in.
    """
    half_image = np.zeros_like(image)
    half = half_image.shape[0]//2
    half_image[half:-lower_padding] = image[half:-lower_padding]
    argmax = np.argmax(half_image.mean(axis=1))
    half_image[argmax:] = half_image[argmax]
    image = image.copy()
    image[half_image > 0] = image.max()
    return image