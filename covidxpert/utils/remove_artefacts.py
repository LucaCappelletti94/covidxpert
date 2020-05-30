import numpy as np
import cv2
from .inpaint import inpaint
print(cv2.useOptimized())


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
        if size < area/400 and size > 10:
            artefacts |= output == i+1

    kernel = np.ones((10, 10), np.uint8)

    artefacts = cv2.dilate(artefacts.astype(np.uint8), kernel)
    artefacts = cv2.morphologyEx(
        artefacts, cv2.MORPH_CLOSE, kernel=np.ones((3, 3)), iterations=10)

    return artefacts.astype(bool)


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
    return inpaint(image, compute_artefacts(image))


def fill_small_white_blobs(mask: np.ndarray, fact: float):
    """Return mask without white blobs smaller than area divided by factor.

    Parameters
    -----------------------------
    mask: np.ndarray,
        Input mask.
    fact: float,
        Mask smoothing factor.

    Returns
    -----------------------------
    Mask without white blobs smaller than area divided by factor.
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    sizes = stats[1:, -1]
    area = np.prod(mask.shape)

    for i, size in enumerate(sizes):
        if size < area/fact:
            mask[output == i+1] = 0

    return mask


def fill_small_black_blobs(mask, factor: int):
    """Return mask without black blobs smaller than area divided by factor.
    Parameters
    -----------------------------
    mask: np.ndarray,
        Input mask.
    fact: float,
        Mask smoothing factor.

    Returns
    -----------------------------
    Mask without black blobs smaller than area divided by factor.
    """
    inverted = mask.max() - mask
    _, output, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8)
    sizes = stats[1:, -1]
    area = np.prod(inverted.shape)

    for i, size in enumerate(sizes):
        if size < area/factor:
            mask[output == i+1] = 255

    return mask
