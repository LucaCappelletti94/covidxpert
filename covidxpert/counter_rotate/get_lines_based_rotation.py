from itertools import chain
import numpy as np
from ..utils import polar2cartesian, compute_linear_coefficients, rotate_image, normalize_image, get_simmetry_axis
from .get_dominant_lines import get_dominant_lines
from typing import Tuple
import cv2


def normalize_angle(angle: float) -> float:
    return np.sign(angle)*(90 - abs(angle)) if abs(angle) < 90 else abs(angle) - 90


def get_lines_based_rotation(image: np.ndarray) -> float:
    """Return best rotation angle.

    If no rotation is detected, zero is returned.

    Parameters
    ------------------
    image: np.ndarray,
        The image to apply the Hough transform to.
    n_lines: int,
        The number of lines to be considered from the HoughLines functions

    Returns
    ------------------
    Best rotation angle.
    """

    # Compute almost vertical lines with Hough from given image
    lines = cv2.HoughLines(image, 1, np.pi / 180, 100, None, 0, 0)
    lines = () if lines is None else lines
    lines = get_dominant_lines(polar2cartesian(lines), *image.shape)

    # Compute almost vertical lines with Hough from given image
    prob_lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=image.shape[1]//10
    )
    prob_lines = () if prob_lines is None else prob_lines
    prob_lines = get_dominant_lines(prob_lines, *image.shape)

    all_lines = list(chain(lines, prob_lines))

    if not all_lines:
        return 0

    points = np.median(all_lines, axis=0)

    x0, _, x1, _ = points

    m, _ = compute_linear_coefficients(*points)
    angle = np.degrees(np.arctan(m))
    return normalize_angle(angle)