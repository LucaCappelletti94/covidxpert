from typing import List
import numpy as np
from ..utils import compute_linear_coefficients, get_projected_points


def get_dominant_lines(
    lines: np.ndarray,
    height: int,
    width: int,
    max_inclination: float = 75,
) -> np.ndarray:
    """Return at most k dominant lines within given inclination.

    Parameters
    -------------------
    lines:np.ndarray,
        Array of lines to be parsed.
    k:int=10,
        Maximal amount of lines to be selected.
    max_inclination:float=70,
        Range of inclination to consider.

    Returns
    -------------------
    Array of selected dominand lines.
    """
    for ((x0, y0, x1, y1),) in lines:
        # We retrieve the linear coefficients
        m, q = compute_linear_coefficients(x0, y0, x1, y1)
        # The line is closer to be horizzontal than vertical
        if abs(m) < height/width:
            # We skip this line
            continue
        # Otherwise we get the projection of the points to the lower and upper
        # sides of the image, using the provided heights.
        x0, y0, x1, y1 = get_projected_points(m, q, x0, height)
        # If the middle point of the line does no fall within the central
        # fifth of the image (e.i. the line is vertically inclined but close
        # to either sides)
        if abs((x0+x1) - width)/2 > width/5:
            # We skip this line
            continue
        # Finally, if the lines starting or ending X coordinate is within
        # the initial fifth of the image or the last fifth of the image
        # meaning that it is unlikely to be representing the spinal cord
        # we drop also this line.
        if min(x0, width-x0) < width/5 or min(x1, width-x1) < width/5:
            # We skip this line
            continue
        # If the line has a angular coefficient that is greater than the given
        # one we return the projected points.
        if np.abs(m) >= np.tan(max_inclination):
            yield x0, y0, x1, y1
