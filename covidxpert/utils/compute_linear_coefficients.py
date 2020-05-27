from typing import Tuple
import numpy np


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
    if (x0 != x1):
        m = (y1-y0)/(x1-x0)
        q = -x0*m+y0
    else:
        m = np.inf
        q = x0
    return m, q
