import numpy as np
import cv2
from .normalize_image import normalize_image


def get_thumbnail(image: np.ndarray, width: int) -> np.ndarray:
    """Return image resized to given size if image is bigger.

    Parameters
    ------------------------
    image: np.ndarray,
        The image to be resized.
    width: int,
        The width to resized the image to.

    Returns
    ------------------------
    Resized image.
    """
    if image.shape[1] <= width:
        return image
    height = int(image.shape[0] * width/image.shape[1])
    return cv2.resize(  # pylint: disable=no-member
        image,
        (width, height),
        cv2.INTER_AREA  # pylint: disable=no-member
    )
