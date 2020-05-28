from typing import Tuple
import numpy as np


def get_projected_points(m: float, q: float, x: float, height: int) -> Tuple[float]:
    """Return the points projected on the upper and lower border.

    Parameters
    ---------------------
    m: float,
        Angular coefficient.
    q: float,
        Intersect.
    x: float,
        X value to use when m is infinite.
    height: int,
        Height of the considered image.

    Returns
    ---------------------
    Tuple containing the two projected points.
    """
    x0 = x if np.isinf(m) else -q/m
    y0 = 0
    x1 = x if np.isinf(m) else (height - q) / m
    y1 = height
    return x0, y0, x1, y1
