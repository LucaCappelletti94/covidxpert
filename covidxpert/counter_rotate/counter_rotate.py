import numpy as np
from ..utils import get_thumbnail, rotate_image, get_simmetry_axis, simmetry_loss, darken
from .get_spinal_cord_mask import get_spinal_cord_mask
from .get_rectangle_based_rotation import get_rectangle_based_rotation
from .get_lines_based_rotation import get_lines_based_rotation
from typing import Tuple, List, Union
import cv2


def counter_rotate(
    image: np.ndarray,
    width: int = 256,
    left_factor: float = 0.2,
    right_factor: float = 0.4,
    others: List[np.ndarray] = None
) -> Union[Tuple[np.ndarray, float, int], Tuple[np.ndarray, float, int, List[np.ndarray]]]:
    """Return counter-rotated image to optimize, its angle and its simmetry axis.

    Parameters
    ------------------
    image: np.ndarray,
        The image to counter rotate.
    width: int = 256,
        Width to which resize the image before processing.
    left_factor: float = 0.2,
        Percentage to interpolate from left valley minima to center peak.
    right_factor: float = 0.4,
        Percentage to interpolate from right valley minima to center peak.
    others: List[np.ndarray] = None,
        Optional parameter to specify images to be rotated alongside the
        given original image.

    Returns
    ------------------
    Tuple with counter-rotated image to optimize, its angle and its simmetry axis.
    """
    thumb = get_thumbnail(image, width=width)
    spine = get_spinal_cord_mask(
        thumb,
        left_factor=left_factor,
        right_factor=right_factor
    )

    # TODO: Maybe we can compute the optimal padding from the spine!
    angle0 = 0
    angle1, spine_x = get_rectangle_based_rotation(spine)
    angle2 = get_lines_based_rotation(spine)

    blurred0 = cv2.blur(thumb, (21, 21))  # pylint: disable=no-member
    blurred1 = rotate_image(blurred0, angle1)
    blurred2 = rotate_image(blurred0, angle2)

    spine_x = spine_x/thumb.shape[1]

    x1 = get_simmetry_axis(blurred0, 0.4)
    x2 = get_simmetry_axis(blurred1, 0.4)
    x3 = get_simmetry_axis(blurred2, 0.4)

    losses = [
        simmetry_loss(blurred0, x1),
        simmetry_loss(blurred1, x2),
        simmetry_loss(blurred2, x3)
    ]

    best_rotation = np.argmin(losses)

    best_angle = [
        angle0, angle1, angle2
    ][best_rotation]

    result = rotate_image(image, best_angle), best_angle, spine_x

    if others is None:
        return result

    return (*result, [
        rotate_image(other, best_angle)
        for other in others
    ])
