from typing import Tuple

import cv2
import numpy as np
from covidxpert.utils import automated_msrcr


def darken(image: np.ndarray, clip: float = 2.0, kernel: Tuple = (9, 9)) -> np.ndarray:
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
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=kernel)
    image = clahe.apply(image)
    image = cv2.equalizeHist(image)
    return image


def darken_msrcr(image: np.ndarray, sigma_darken: list=[10, 20, 30]) -> np.ndarray:
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
    darken = automated_msrcr(image, sigma_darken)
    return darken


