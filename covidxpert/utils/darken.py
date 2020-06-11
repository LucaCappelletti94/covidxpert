from typing import Tuple

import cv2
import numpy as np
from .retinex import automated_msrcr

def darken(image: np.ndarray) -> np.ndarray:
    """Return the image with both global and local normalization.

    Parameters
    --------------------
    image:np.ndarray,
        The image to be darkened.
    sigma_darken:list=[10, 20, 30],
        Recomemended sigma values [10, 20, 30]

    Returns
    --------------------
    Return the darkened image.
    """
    min_side = min(image.shape)
    sigma_darken = min_side/5, min_side/4, min_side/3
    darken = automated_msrcr(image, sigma_darken)
    return darken


