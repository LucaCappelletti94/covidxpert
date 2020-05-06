import numpy as np
import cv2
from typing import Tuple


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


def get_corners(image: np.ndarray, corners_number: int = 1000, area_threshold: float = 0.01) -> Tuple[np.ndarray, bool]:
    """Return image convex mask."""
    _, thresholded_mask = cv2.threshold(image, 0, 255, cv2.THRESH_TRIANGLE)

    _, output, stats, _ = cv2.connectedComponentsWithStats(
        thresholded_mask,
        connectivity=8
    )

    max_sizeR = np.argmax(stats[1:, -1])
    image_mask = np.zeros((output.shape), dtype=np.uint8)
    image_mask[output == max_sizeR + 1] = 255

    # We determine the contours of the mask
    contours, _ = cv2.findContours(
        image=image_mask,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE
    )

    # And fill up the mask within thr contours
    # as they might remain holes within.
    image_mask = cv2.fillPoly(
        image_mask,
        pts=[contours[0]],
        color=(255, 255, 255)
    )

    # Get up to corners_number corners
    # and filter up only the 4 cardinal corners
    corners = get_cardinal_corner_points(cv2.goodFeaturesToTrack(
        image=image_mask,
        maxCorners=corners_number,
        qualityLevel=0.01,
        minDistance=1
    ).reshape(-1, 2))

    polygon_rate = get_polygon_area(*corners.T) / np.prod(image_mask.shape)
    image_white_area = get_image_white_area(image_mask)

    return corners, np.abs(1 - polygon_rate / image_white_area) < area_threshold
