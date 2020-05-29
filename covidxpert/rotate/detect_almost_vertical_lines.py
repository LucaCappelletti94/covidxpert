from itertools import chain
import numpy as np
from ..utils import polar2cartesian
from .get_dominant_lines import get_dominant_lines


def detect_almost_vertical_lines(
    image: np.ndarray,
) -> np.ndarray:
    """Return image containing almost vertical lines.

    Parameters
    ------------------
    image: np.ndarray,
        The image to apply the Hough transform to.

    Returns
    ------------------
    Returns binary image containing vertical lines.
    """

    # Compute almost vertical lines with Hough from given image
    lines = get_dominant_lines(polar2cartesian(
        cv2.HoughLines(image, 1, np.pi / 180, 100, None, 0, 0)[:1000]
    ), *image.shape)

    # Compute almost vertical lines with Hough from given image
    probabilistic_lines = get_dominant_lines(cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=image.shape[1]//10
    )[:1000], *image.shape)

    # Drawing the lines on the image.
    x0, y0, x1, y1 = np.median(
        chain(lines, probabilistic_lines),
        axis=0
    ).astype(int)

    median_image = cv2.cvtColor(np.zeros_like(image), cv2.COLOR_GRAY2BGR)
    cv2.line(median_image, (x0, y0), (x1, y1), (1, 1, 1), 5, cv2.LINE_AA)

    # Convert the image back to grayscale
    return normalize_image(cv2.cvtColor(median_image, cv2.COLOR_BGR2GRAY))
