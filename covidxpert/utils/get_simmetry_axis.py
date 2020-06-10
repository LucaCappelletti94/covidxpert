from typing import Tuple
import numpy as np
import cv2
from numba import njit


@njit
def trim_flip(image: np.ndarray, x: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return tuple with the trimmed image and the trimmed flipped image.

    Parameters
    --------------------
    image: np.ndarray,
        Image to be trimmed and flipped
    x: int,
        Simmetry axis around which to trim.

    Returns
    --------------------
    Return tuple with the trimmed image and the trimmed flipped image.
    """
    min_side = min(image.shape[1] - x, x)
    cut_image = image[:, x-min_side: x+min_side]
    flipped = cut_image[:, ::-1]
    return cut_image, flipped


def simmetry_loss(image: np.ndarray, x: int) -> float:
    """Return score for the simmetry at given simmetry axis.

    Parameters
    --------------------
    image: np.ndarray,
        Image for which to compute the loss.
    x: int,
        Abscissa position to use for compute simmetry loss.

    Returns
    --------------------
    Return score for the simmetry at given simmetry axis.
    """
    cut_image, flipped = trim_flip(image, x)
    differences = (cut_image-flipped)**2
    return np.mean(differences[cut_image == 0 | flipped == 0])


def numba_simmetry_axis(image: np.ndarray, width: int, padding: float) -> int:
    """Return optimal simmetry axis.

    Parameters
    --------------------
    image: np.ndarray,
        Image for which to compute the optimal simmetry axis.
    width: int,
        Width to which to resize before processing the image.
    padding: float,
        Percentage of image to skip from left and ride.

    Returns
    --------------------
    Return optimal simmetry axis.
    """
    best_axis = width//2
    min_loss = simmetry_loss(image, best_axis)
    candidates = np.arange(int(width*padding), int(width*(1-padding)))

    for candidate in candidates:
        loss = simmetry_loss(image, candidate)
        if loss < min_loss:
            best_axis = candidate
            min_loss = loss

    return best_axis


def get_simmetry_axis(image: np.ndarray, width: int = 256, padding: float = 0.45) -> int:
    """Return optimal simmetry axis.

    Parameters
    --------------------
    image: np.ndarray,
        Image for which to compute the optimal simmetry axis.
    width: int = 256,
        Width to which to resize before processing the image.
    padding: float = 0.45,
        Percentage of image to skip from left and ride.

    Returns
    --------------------
    Return optimal simmetry axis.
    """
    height = int(image.shape[0] * width/image.shape[1])
    resized_image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    return int(
        numba_simmetry_axis(resized_image, width, padding) /
        width * image.shape[1]
    )
