import numpy as np
import cv2
from typing import Tuple
from ..utils import (darken, median_mask, fill_lower_max, trim_flip, normalize_image,
                     get_simmetry_axis, fill_small_black_blobs, fill_small_white_blobs)
from ..borders import get_refined_borders
from numba import njit


def get_default_mask(image: np.ndarray, borders: np.ndarray, padding: float = 0.45) -> np.ndarray:
    """Apply default mask over given image.

    Parameters
    ----------------
    image:np.ndarray,
        The image over which to apply the mask.
    borders:np.ndarray
        The borders to use.

    Returns
    ----------------
    The mask obtained over the image.
    """
    mask = normalize_image(median_mask(image))
    mask[borders > 0] = 255
    width = mask.shape[1]
    mask[:, int(width*padding):int(width*(1-padding))] = 255
    return mask


def get_simmetry_mask(image: np.ndarray, x: float) -> np.ndarray:
    """Return simmetry-based mask for given image.

    IMPORTANT: the simmetry mask DOES not have the same size of the input image
    but it is based on its transformation to simmetric image.

    Parameters
    -----------------------
    image: np.ndarray,
        Image for which to compute the simmetry mask.
    x: float,
        The simmetry axis to use.

    Returns
    -----------------------
    Boolean mask based on simmetry.
    """
    # Getting the refined borders.
    borders = get_refined_borders(image, x)

    darkened_image = darken(image)
    darkened_image[darkened_image < np.median(darkened_image)] = 0
    darkened_image = fill_small_black_blobs(darkened_image, 200)
    cut, flipped = trim_flip(darkened_image, x)

    product = normalize_image(cut.astype(float) * flipped.astype(float))
    mask = get_default_mask(product, borders)
    mask = fill_small_white_blobs(mask, 20)
    mask = cv2.dilate(mask, kernel=np.ones((5, 5)))
    mask = fill_small_white_blobs(mask, 20)
    mask = fill_small_black_blobs(mask, 20)
    mask = fill_lower_max(mask)
    mask = fill_small_black_blobs(mask, 20)

    return mask
