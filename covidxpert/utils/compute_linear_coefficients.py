from typing import Tuple
import numpy as np
from numba import njit


@njit
def compute_linear_coefficients(
    x0: float,
    y0: float,
    x1: float,
    y1: float
) -> Tuple[float, float]:
    """Return the linear coefficients for the line passing by the given points.

    Parameters
    -------------------
    x0:float,
    y0:float,
    x1:float,
    y1:float

    Returns
    -------------------
    Return the tuple of linear coefficients, the first one being the
    angular coefficient and the second one being the intercept.
    """
    # Instead of using numpy.isclose we use this to have the numbas support.
    if abs(x0-x1)<1e-08:
        m = np.inf
        q = x0
    else:
        m = (y1-y0)/(x1-x0)
        q = y0 - m*x0
    return m, q
