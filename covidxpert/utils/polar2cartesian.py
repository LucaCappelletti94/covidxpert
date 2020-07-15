from typing import Generator
import numpy as np


def polar2cartesian(lines: np.ndarray) -> Generator:
    """Return given lines converted from polar to cartesian representation.

    The lines numpy array has to be of shape (n, 1, 2), where n is the number
    of samples provided.

    Parameters
    ------------------
    lines: np.ndarray,
        The lines in polar representation to be converted.

    Returns
    -----------------
    Lines in cartesian representation.
    """
    for ((rho, theta),) in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        yield ((x0 - b, y0 + a, x0 + b, y0 - a),)
