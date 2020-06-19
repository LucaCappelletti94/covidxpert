import numpy as np
from .retinex import automated_msrcr


def darken(image: np.ndarray) -> np.ndarray:
    """Return the image with both global and local normalization.

    Parameters
    --------------------
    image:np.ndarray,
        The image to be darkened.

    Returns
    --------------------
    Return the darkened image.
    """
    min_side = min(image.shape)
    sigma_darken = min_side/5, min_side/4, min_side/3
    image_darken = automated_msrcr(image, sigma_darken)
    return image_darken
