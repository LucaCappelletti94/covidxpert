from skimage.feature import corner_harris, corner_peaks
from typing import List
import numpy as np


def get_corners(image: np.ndarray, chull: np.ndarray) -> List:
    """Return image corners."""
    return corner_peaks(corner_harris(chull), min_distance=10, num_peaks=4)
