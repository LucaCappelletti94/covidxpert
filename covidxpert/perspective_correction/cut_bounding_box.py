import numpy as np


def cut_bounding_box(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Cut the image using the given bounding box."""
    (min_x, min_y), (max_x, max_y) = corners.min(axis=0), corners.max(axis=0)
    return image[min_y:max_y, min_x:max_x]
