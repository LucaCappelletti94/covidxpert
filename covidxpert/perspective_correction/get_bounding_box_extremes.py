from typing import Tuple
import numpy as np


def get_bounding_box_extremes(chull: np.ndarray) -> Tuple[float, float, float, float]:
    """Return extremes of given bounding box."""
    indexes = np.where(chull > 0)
    min_x, min_y = np.min(indexes, axis=1)
    max_x, max_y = np.max(indexes, axis=1)
    return min_x, min_y, max_x, max_y
