from itertools import chain
import numpy as np
from ..utils import polar2cartesian, compute_linear_coefficients, rotate_image, normalize_image
from .get_dominant_lines import get_dominant_lines
import cv2


def get_rotation(image: np.ndarray) -> float:
    """Return angle for rotation.

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
    
    # Drawing the lines on the image.
    x0, y0, x1, y1 = np.median(all_lines, axis=0).astype(int)

    composite = np.zeros_like(image, dtype=np.float64)
    for line in all_lines:
        p1, p2, p3, p4 = line
        partial = cv2.cvtColor(np.zeros_like(image), cv2.COLOR_GRAY2BGR)
        cv2.line(partial, (int(p1), int(p2)), (int(p3), int(p4)), (1, 1, 1), 5, cv2.LINE_AA)
        composite += partial[:, :, 2]

    # use this line to counter rotate the original image (calculate the coeficient compute_linear_coeficient)

    median_image = cv2.cvtColor(np.zeros_like(image), cv2.COLOR_GRAY2BGR)
    cv2.line(median_image, (x0, y0), (x1, y1), (255, 255, 255), 5, cv2.LINE_AA)

    # Convert the image back to grayscale
    # return normalize_image(cv2.cvtColor(median_image, cv2.COLOR_BGR2GRAY))

    m, _ = compute_linear_coefficients(x0, y0, x1, y1)

    composite = normalize_image(composite)

    return np.degrees(np.arctan(m)), median_image, composite