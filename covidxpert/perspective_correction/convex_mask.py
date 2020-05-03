from skimage.morphology import convex_hull_image
import numpy as np


def convex_mask(image: np.ndarray) -> np.ndarray:
    """Return image convex mask."""
    return convex_hull_image(image > 0)
