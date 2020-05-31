from itertools import chain
import numpy as np
from ..utils import polar2cartesian, compute_linear_coefficients, rotate_image, normalize_image
from .get_dominant_lines import get_dominant_lines
import cv2


def get_rotation(image: np.ndarray) -> float:
    """Return angle for rotation.

    If no rotation is detected, zero is returned.

    Parameters
    ------------------
    image: np.ndarray,
        The image to apply the Hough transform to.
    n_lines: int,
        The number of lines to be considered from the HoughLines functions

    Returns
    ------------------
    Angle of inclination of the given image.
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

    m, _ = compute_linear_coefficients(*points)
    return np.degrees(np.arctan(m))
