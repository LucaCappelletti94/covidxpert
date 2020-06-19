import numpy as np
import cv2
from ..utils import add_padding
from .get_corners import get_corners
from .cut_bounding_box import cut_bounding_box
from typing import List, Union, Tuple


def get_new_cardinals(image: np.ndarray) -> np.ndarray:
    return np.float32([
        [0.0, 0.0],  # top_left,
        [image.shape[1], 0],  # top_right
        [image.shape[1], image.shape[0]],  # bottom_right,
        [0, image.shape[0]],  # bottom_left,
    ])


def perspective_correction(image: np.ndarray, others: List[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """Return image with perspective correction.

    Parameters
    -----------------------
    image: np.ndarray, 
        The image whose perspective is to be corrected.
    others: List[np.ndarray] = None,
        Optional parameter to specify images to be transformed alongside the
        given original image.

    Returns
    ------------------------
    Either the corrected image or a tuple with both the corrected image and
    the list of other images associates, if provided.
    """
    padded = add_padding(image)
    corners, requires_correction, _ = get_corners(padded)
    if not requires_correction:
        if others is not None:
            return image, others
        return image
    padded = cut_bounding_box(padded, corners)
    corners -= corners.min(axis=0)
    new_corners = get_new_cardinals(padded)
    M, _ = cv2.findHomography(  # pylint: disable=no-member
        corners,
        new_corners
    )
    result = cv2.warpPerspective(  # pylint: disable=no-member
        padded,
        M,
        (padded.shape[1], padded.shape[0])
    )

    if others is None:
        return result

    return result, [
        cv2.warpPerspective(  # pylint: disable=no-member
            add_padding(other),
            M,
            (padded.shape[1], padded.shape[0])
        )
        for other in others
    ]
