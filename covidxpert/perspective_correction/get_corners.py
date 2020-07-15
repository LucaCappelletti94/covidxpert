from typing import Tuple
import numpy as np
import cv2
from .get_masked_image import get_masked_image


def get_top_left_corner(corners: np.ndarray) -> int:
    return np.argmin(np.sum(corners, axis=1))


def get_bottom_right_corner(corners: np.ndarray) -> int:
    return np.argmax(np.sum(corners, axis=1))


def flip_corners(corners: np.ndarray) -> np.ndarray:
    max_x, _ = corners.max(axis=0)
    return np.abs((max_x, 0) - corners)


def get_cardinal_corner_points(corners: np.ndarray) -> np.ndarray:
    flipped = flip_corners(corners)
    return corners[[
        get_top_left_corner(corners),
        get_top_left_corner(flipped),
        get_bottom_right_corner(corners),
        get_bottom_right_corner(flipped)
    ]].astype(int)


def get_polygon_area(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def get_image_white_area(image):
    return (image == 255).sum() / image.size


def get_corners(image: np.ndarray, corners_number: int = 1000, area_threshold: float = 0.03) -> Tuple[np.ndarray, bool]:
    """Return image convex mask."""
    image_mask = get_masked_image(image)

    # Get up to corners_number corners
    # and filter up only the 4 cardinal corners
    corners = get_cardinal_corner_points(cv2.goodFeaturesToTrack(  # pylint: disable=no-member
        image=image_mask,
        maxCorners=corners_number,
        qualityLevel=0.01,
        minDistance=1
    ).reshape(-1, 2))

    polygon_rate = get_polygon_area(*corners.T) / np.prod(image_mask.shape)
    image_white_area = get_image_white_area(image_mask)

    score = np.abs(1 - polygon_rate / image_white_area)

    return corners, score < area_threshold, score
