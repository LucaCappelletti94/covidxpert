from .get_bounding_box_extremes import get_bounding_box_extremes
import numpy as np


def cut_bounding_box(image: np.ndarray, chull: np.ndarray) -> np.ndarray:
    """Cut the image using the given bounding box."""
    min_x, min_y, max_x, max_y = get_bounding_box_extremes(chull)
    return image[min_x:max_x, min_y:max_y]
