import numpy as np
from .get_bounding_box_extremes import get_bounding_box_extremes


def transpose_corners(corners: np.ndarray, chull: np.ndarray) -> np.ndarray:
    min_x, min_y, _, _ = get_bounding_box_extremes(chull)
    return corners - (min_x, min_y)
