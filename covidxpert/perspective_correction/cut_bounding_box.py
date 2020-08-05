import numpy as np


def cut_bounding_box(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Cut the image using the given bounding box.
    One is added to the maximum (x and y) to include them in the slicing

    Parameters
    -----------------------
    image: np.ndarray,
        The image to be sliced according to corners
    corners: np.ndarray,
        Corners of the image

    Returns
    ------------------------
    Sliced image
    """
    (min_x, min_y), (max_x, max_y) = corners.min(axis=0), corners.max(axis=0) + 1
    return image[min_y:max_y, min_x:max_x]
