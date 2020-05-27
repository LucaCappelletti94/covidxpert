from typing import List
import numpy as np

def get_dominant_lines(
    lines:np.ndarray,
    height:int,
    width:int,
    max_inclination:float=75,
)->np.ndarray:
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
            continue
        x0, y0, x1, y1 = get_projected_points(m, q, height)
        if abs((x0+x1)/2 - width/2) > width/5:
            continue
        if min(x0, width-x0) < width/5:
            continue
        if min(x1, width-x1) < width/5:
            continue
        if np.abs(m)>=np.tan(max_inclination):
            yield x0, y0, x1, y1