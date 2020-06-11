from numba import njit, prange
import numpy as np
from typing import Tuple


@njit
def get_inclined_rectangle(image_shape: Tuple[int, int], x: int, angle: float, width: int) -> np.ndarray:
    """Return inclined rectangle with given image shapes.

    Parameters
    ------------------
    image_shape: Tuple[int, int],
        Shape of the image to generate.
    x: int,
        Abscissa x offset.
    angle: float,
        Angle for the rotation of the rectangle.
    width: int,
        Width of the rectangle.

    Returns
    ------------------
    Return boolean mask with the shape of the requested rectangle.
    """
    rectangle = np.zeros(image_shape, dtype=np.bool_)
    half_width = width//2
    if abs(angle) < 0.1:
        lower_bound = max(0, x-half_width)
        upper_bound = min(image_shape[1], x+half_width)
        rectangle[:, lower_bound:upper_bound] = True
        return rectangle
    m = np.tan(np.radians(angle))
    q = image_shape[0]/2 - m*x
    for y in range(image_shape[0]):
        xi = int((y-q)/m)
        lower_bound = max(0, xi-half_width)
        upper_bound = min(image_shape[1], xi+half_width)
        if upper_bound <= 0:
            continue
        rectangle[y, lower_bound:upper_bound] = True
    return rectangle


def get_rectangle_based_rotation(mask: np.ndarray) -> Tuple[float, float]:
    """Return tuple with best rotation angle and best rotation axis.

    Parameters
    ------------------
    mask: np.ndarray,
        Mask to use for the rotation. The mask should be obtained form the
        spinal cord mask function.

    Returns
    ------------------
    Tuple with best angle and simmetry axis.
    """
    best_score = 0
    best_x = None
    best_angle = None

    spine_width = mask.shape[1]//10

    x_mask = mask.sum(axis=0) > 0
    xs = np.where(x_mask)[0]
    angles = np.linspace(75, 105, num=50)

    for x in xs:
        for angle in angles:
            rectangle = get_inclined_rectangle(
                image_shape=mask.shape,
                x=x,
                angle=angle,
                width=spine_width
            )
            score = mask[rectangle].sum()
            if score > best_score:
                best_score = score
                best_x = x
                best_angle = angle

    return 90 - best_angle, best_x