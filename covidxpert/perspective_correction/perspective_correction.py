import numpy as np
import cv2
from .add_padding import add_padding
from .get_corners import get_corners
from .cut_bounding_box import cut_bounding_box

def get_new_cardinals(image: np.ndarray) -> np.ndarray:
    return np.float32([
        [0.0, 0.0],  # top_left,
        [image.shape[1], 0],  # top_right
        [image.shape[1], image.shape[0]],  # bottom_right,
        [0, image.shape[0]],  # bottom_left,
    ])


def perspective_correction(image: np.ndarray) -> np.ndarray:
    padded = add_padding(image)
    corners, requires_correction, _ = get_corners(padded)
    if not requires_correction:
        return image
    padded = cut_bounding_box(padded, corners)
    corners -= corners.min(axis=0)
    new_corners = get_new_cardinals(padded)
    M, _ = cv2.findHomography(
        corners,
        new_corners
    )
    return cv2.warpPerspective(padded, M, (padded.shape[1], padded.shape[0]))
