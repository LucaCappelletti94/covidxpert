import numpy as np
import cv2
from .normalize_image import normalize_image


def compute_artefacts(image: np.ndarray, sigma: float = 15) -> np.ndarray:
    """Return the artefacts identified in given image.

    Parameters
    -----------------------------
    image: np.ndarray,
        The image from where to extract the artefacts.
    sigma: float = 15,
        The sigma to use for the Gaussian blur.

    Returns
    -----------------------------
    Boolean mask with the artefacts identified.
    """
    blur = cv2.GaussianBlur(image, (5, 5), sigma, sigma).astype(float)
    result = image - blur
    result = normalize_image(result)
    result = cv2.threshold(
        image,
        thresh=np.median(result),
        maxval=255,
        type=cv2.THRESH_BINARY
    )[1]
    _, output, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    sizes = stats[1:, -1]
    area = np.prod(result.shape)
    artefacts = np.zeros(result.shape, dtype=bool)

    for i, size in enumerate(sizes):
        if size < area/10:
            artefacts |= output == i+1

    kernel = np.ones((3, 3), np.uint8)

    artefacts = cv2.dilate(artefacts.astype(np.uint8), kernel).astype(bool)

    return artefacts


def remove_artefacts(image: np.ndarray, sigma: float = 15) -> np.ndarray:
    """Return image without identified artefacts.

    Parameters
    -----------------------------
    image: np.ndarray,
        The image from where to extract the artefacts.
    sigma: float = 15,
        The sigma to use for the Gaussian blur.

    Returns
    -----------------------------
    Image without the identified artefacts.
    """
    artefacts = compute_artefacts(image, sigma)
    cleared_image = image.copy()
    cleared_image[artefacts] = 0

    return cleared_image
