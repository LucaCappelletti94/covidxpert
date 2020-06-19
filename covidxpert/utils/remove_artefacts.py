import numpy as np
import cv2
from .inpaint import inpaint


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
    result = cv2.threshold(  # pylint: disable=no-member
        image,
        thresh=np.median(image),
        maxval=255,
        type=cv2.THRESH_BINARY  # pylint: disable=no-member
    )[1]
    _, output, stats, _ = cv2.connectedComponentsWithStats(  # pylint: disable=no-member
        result, connectivity=8)
    sizes = stats[1:, -1]
    area = np.prod(result.shape)
    artefacts = np.zeros(result.shape, dtype=bool)

    for i, size in enumerate(sizes):
        if area/400 > size > 10:
            artefacts |= output == i+1

    kernel = np.ones((10, 10), np.uint8)

    artefacts = cv2.dilate(artefacts.astype(np.uint8), kernel)  # pylint: disable=no-member
    artefacts = cv2.morphologyEx(  # pylint: disable=no-member
        artefacts, cv2.MORPH_CLOSE, kernel=np.ones((3, 3)), iterations=10)  # pylint: disable=no-member

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


def fill_small_white_blobs(mask: np.ndarray, factor: float):
    """Return mask without white blobs smaller than area divided by factor.

    Parameters
    -----------------------------
    mask: np.ndarray,
        Input mask.
    factor: float,
        Mask smoothing factor.

    Returns
    -----------------------------
    Mask without white blobs smaller than area divided by factor.
    """
    if factor == 0:
        raise ValueError('Factor must be different from 0')
    _, output, stats, _ = cv2.connectedComponentsWithStats(  # pylint: disable=no-member
        mask, connectivity=8
    )
    sizes = stats[1:, -1]
    area = np.prod(mask.shape)

    for i, size in enumerate(sizes):
        if size < area/factor:
            mask[output == i+1] = 0

    return mask


def fill_small_black_blobs(mask, factor: float):
    """Return mask without black blobs smaller than area divided by factor.
    Parameters
    -----------------------------
    mask: np.ndarray,
        Input mask.
    factor: float,
        Mask smoothing factor.

    Returns
    -----------------------------
    Mask without black blobs smaller than area divided by factor.
    """
    if factor == 0:
        raise ValueError('Factor must be different from 0')

    inverted = mask.max() - mask
    _, output, stats, _ = cv2.connectedComponentsWithStats( # pylint: disable=no-member
        inverted, connectivity=8)
    sizes = stats[1:, -1]
    area = np.prod(inverted.shape)

    for i, size in enumerate(sizes):
        if size < area/factor:
            mask[output == i+1] = 255

    return mask
