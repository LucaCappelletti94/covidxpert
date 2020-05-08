import numpy as np


def trim_padding(image: np.ndarray, padding: int) -> np.ndarray:
    """Return the image without the given paddin amount.
    
    Parameters
    -------------------
    image: np.ndarray,
        The image to be trimmed.
    padding: int,
        The amount to be trimmed around the image.

    Returns
    -------------------
    Return the image without the padding.
    """
    return image[padding:-padding, padding:-padding]
