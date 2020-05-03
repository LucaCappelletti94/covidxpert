import numpy as np


def add_padding(image: np.ndarray, padding: int = 10) -> np.ndarray:
    return np.pad(image, padding, mode="constant")
