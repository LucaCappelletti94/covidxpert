import cv2
import numpy as np
import os
from scipy import ndimage


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
    image = cv2.imread(path, 0)
    image = ndimage.median_filter(image, 3)
    image = ndimage.gaussian_filter(image, sigma=5)
    image = (image - image.min()) / (image.max() - image.min())
    return image
