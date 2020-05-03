import numpy as np
import cv2
from .add_padding import add_padding
from .convex_mask import convex_mask
from .get_corners import get_corners
from .cut_bounding_box import cut_bounding_box
from .transpose_corners import transpose_corners
from .sort_corners import sort_corners


def perspective_correction(image: np.ndarray) -> np.ndarray:
    padded = add_padding(image)
    chull = convex_mask(padded)
    corners = get_corners(padded, chull)

    if len(corners) < 4:
        return image

    padded = cut_bounding_box(padded, chull)
    corners = transpose_corners(corners, chull)
    corners = np.float32(sort_corners(corners))
    new_corners = np.float32([
        [0.0, 0.0],  # top_left,
        [0.0, padded.shape[1]],  # top_right
        [padded.shape[0], padded.shape[1]],  # bottom_right,
        [padded.shape[0], 0],  # bottom_left,
    ])
    new_corners = np.float32([
        (x, y) for y, x in new_corners.tolist()
    ])
    corners = np.float32([
        (x, y) for y, x in corners.tolist()
    ])
    M, _ = cv2.findHomography(
        corners,
        new_corners
    )
    return cv2.warpPerspective(padded, M, (padded.shape[1], padded.shape[0]))
