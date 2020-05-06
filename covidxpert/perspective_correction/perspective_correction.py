import numpy as np
import cv2
from .add_padding import add_padding
from .get_corners import get_corners
from .cut_bounding_box import cut_bounding_box


def perspective_correction(image: np.ndarray) -> np.ndarray:
    padded = add_padding(image)
    corners, requires_correction = get_corners(padded)
    if not requires_correction:
        return image
    padded = cut_bounding_box(padded, corners)
    corners -= corners.min(axis=0)
    new_corners = np.float32([
        [0.0, 0.0],  # top_left,
        [padded.shape[1], 0],  # top_right
        [padded.shape[1], padded.shape[0]],  # bottom_right,
        [0, padded.shape[0]],  # bottom_left,
    ])
    M, _ = cv2.findHomography(
        corners,
        new_corners
    )
    return cv2.warpPerspective(padded, M, (padded.shape[1], padded.shape[0]))
