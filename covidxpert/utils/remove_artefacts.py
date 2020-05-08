import numpy as np
import cv2
from .normalize_image import normalize_image


def compute_artefacts(image: np.ndarray) -> np.ndarray:
    """Return the artefacts identified in given image.

    Parameters
    -----------------------------
    image: np.ndarray,
        The image from where to extract the artefacts.

    Returns
    -----------------------------
    Boolean mask with the artefacts identified.
    """
    result = cv2.threshold(
        image,
        thresh=np.median(image),
        maxval=255,
        type=cv2.THRESH_BINARY
    )[1]
    _, output, stats, _ = cv2.connectedComponentsWithStats(
        result, connectivity=8)
    sizes = stats[1:, -1]
    area = np.prod(result.shape)
    artefacts = np.zeros(result.shape, dtype=bool)

    for i, size in enumerate(sizes):
        if size < area/400:
            artefacts |= output == i+1

    kernel = np.ones((3, 3), np.uint8)

    artefacts = cv2.dilate(artefacts.astype(np.uint8), kernel).astype(bool)

    return artefacts


def remove_artefacts(image: np.ndarray) -> np.ndarray:
    """Return image without identified artefacts.

    Parameters
    -----------------------------
    image: np.ndarray,
        The image from where to extract the artefacts.

    Returns
    -----------------------------
    Image without the identified artefacts.
    """
    artefacts = compute_artefacts(image)
    cleared_image = image.copy()
    cleared_image[artefacts] = 0

    return cleared_image
