from typing import Tuple

import cv2
import numpy as np


def darken(image: np.ndarray, clip: float = 2.0, kernel: Tuple = (3, 3)) -> np.ndarray:
    """Return the image with both global and local normalization.

    Parameters
    --------------------
    image:np.ndarray,
        The image to be darkened.
    clip:float=15,
        Maximum local sum after which we clip.
    kernel:Tuple=(15,15),
        Kernel size for local histogram normalization.

    Returns
    --------------------
    Return the darkened image.
    """
    image = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=kernel)
    image = clahe.apply(image)
    image = cv2.equalizeHist(image)
    return image
