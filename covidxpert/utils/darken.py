import cv2
import numpy as np


def darken(image: np.ndarray, clip: float = 2.0, kernel: tuple = (3, 3)) -> np.ndarray:
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
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    image = clahe.apply(image)
    image = cv2.equalizeHist(image)
    return image
