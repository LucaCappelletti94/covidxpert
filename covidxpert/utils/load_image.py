import cv2
import numpy as np
from .normalize_image import normalize_image


def load_image(path: str) -> np.ndarray:
    """Return normalized image at given path.

    Parameters
    ----------------
    path: str,
        Path to image to be loaded.

    Returns
    ----------------
    Return numpy array containing loaded image.
    """
    image = cv2.imread(path, 0) # pylint: disable=no-member
    return normalize_image(image)
