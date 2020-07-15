import numpy as np
import cv2
from .normalize_image import normalize_image


def inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return inpainted image over given mask.

    Parameters
    ---------------------
    image: np.ndarray,
        Grayscale image to fill in.
    mask: np.ndarray,
        Mask with the regions to fill in.

    Returns
    ---------------------
    The image with masked areas filled in.
    """
    backtorgb = cv2.cvtColor(  # pylint: disable=no-member
        image, cv2.COLOR_GRAY2RGB)  # pylint: disable=no-member
    cleared_image = cv2.inpaint(  # pylint: disable=no-member
        backtorgb, normalize_image(mask), 11, cv2.INPAINT_TELEA)  # pylint: disable=no-member
    return cv2.cvtColor(cleared_image, cv2.COLOR_RGB2GRAY)  # pylint: disable=no-member
