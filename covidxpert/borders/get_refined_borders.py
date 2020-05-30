import numpy as np
import cv2
from numba import njit
from ..utils import (normalize_image, trim_flip, darken, median_mask,
                     fill_small_black_blobs, add_padding, trim_padding)


def get_border_regions(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """Return the regions nearby the outer border of the image.

    Parameters
    ---------------------
    image: np.ndarray,
        Image from which to extract the border.
    radius: int = 3,
        Radius from the border of the image to take in consideration.

    Returns
    ---------------------
    Binary mask containing only the outer borders of the image.
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(
        image.max() - image,
        connectivity=8
    )
    sizes = stats[1:, -1]

    border_regions = np.zeros_like(image, dtype=np.bool)

    for i in range(len(sizes)):
        region = output == i+1
        x = region.any(axis=0)
        y = region.any(axis=1)
        if any(k[:radius].any() or k[-radius:].any() for k in (x, y)):
            border_regions |= region

    return border_regions


def get_borders(image: np.ndarray, x: int) -> np.ndarray:
    """Return the borders on given image using given simmetry axis.

    Parameters
    -------------------
    image: np.ndarray,
        Image from which to extract the borders.
    x: int,
        Simmetry axis to use.

    Returns
    -------------------
    Borders boolean mask.
    """
    cut, flipped = trim_flip(image, x)

    cut_mask = normalize_image(median_mask(cut))
    flipped_mask = normalize_image(median_mask(flipped))
    sum_mask = normalize_image(median_mask(
        cut.astype(float) + flipped.astype(float)
    ))

    cut_borders = get_border_regions(cut_mask)
    flipped_borders = get_border_regions(flipped_mask)
    sum_borders = get_border_regions(sum_mask)

    borders = normalize_image((cut_borders | flipped_borders) & sum_borders)
    borders = cv2.dilate(borders, np.ones((9, 9)))
    borders = cv2.medianBlur(borders, 15)
    borders = fill_small_black_blobs(borders, 20)

    return borders


def get_refined_borders(image: np.ndarray, x: int, padding: int = 10) -> np.ndarray:
    """Return the borders on given image using given simmetry axis.

    Parameters
    -------------------
    image: np.ndarray,
        Image from which to extract the borders.
    x: int,
        Simmetry axis to use.
    padding: int = 25,
        Padding of white space to add on every border.

    Returns
    -------------------
    Borders boolean mask.
    """
    borders = normalize_image(
        get_borders(darken(image), x) + get_borders(image, x)
    )
    borders = add_padding(trim_padding(borders, padding), padding, 255)
    borders = cv2.dilate(borders, np.ones((9, 9)))
    borders = cv2.medianBlur(borders, 15)
    borders = fill_small_black_blobs(borders, 20)

    return borders
