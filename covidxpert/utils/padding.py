import numpy as np


def add_padding(image: np.ndarray, padding: int = 10) -> np.ndarray:
    """Return the image with the given paddin amount.

    Parameters
    -------------------
    image: np.ndarray,
        The image to be padded.
    padding: int,
        The amount to be added around the image.

    Returns
    -------------------
    Return the image wth the padding.
    """
    return np.pad(image, padding, mode="constant", constant_values=image.min())


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
