from typing import Tuple
import numpy as np
import cv2


def central_peak(image: np.ndarray, use_left_padding: bool = True, use_right_padding: bool = True) -> int:
    """Return central peak of given image.

    The central peak is detected best in blurred images.

    Parameters
    ------------------------
    image: np.ndarray,
        Image from which to detect the central peak.
    use_left_padding: bool = True,
        Wethever to add a left padding mask.
    use_right_padding: bool = True,
        Wethever to add a right padding mask.

    Returns
    ------------------------
    X abscissa of the central peak.
    """
    best_x = np.mean(image, axis=0)
    if use_left_padding:
        best_x[:image.shape[1]//3] = 0
    if use_right_padding:
        best_x[-image.shape[1]//3:] = 0
    return best_x.argmax()


def main_peaks(image: np.ndarray) -> Tuple[int, int, int]:
    """return main peaks of a given image.

    In a chest x-ray, these peaks represent the left chest, the spine cord
    and the right chest peaks.

    These peaks are detected best on a blurred image.

    Parameters
    ------------------
    image: np.ndarray,
        Image from which we need to detect the central peaks.

    Returns
    ------------------
    Triple with left, middle and central peak.
    """
    central = central_peak(image)
    left_padding = central-image.shape[1]//5
    left_peak = central_peak(image[:, :left_padding])
    right_padding = central+image.shape[1]//5
    right_peak = right_padding + central_peak(image[:, right_padding:])
    return left_peak, central, right_peak


def main_valleys(image: np.ndarray, left_factor=0.25, right_factor=0.4) -> Tuple[int, int]:
    """Return the image two main valleys.

    The valleys in a chest xray are meant to represent the left and right lungs.
    These valleys are detected best on a blurred image.

    Parameters
    ----------------------
    image: np.ndarray,
        The image to apply the valleys cut on.
    left_factor: float = 0.2,
        Percentage to interpolate from left valley minima to center peak.
    right_factor: float = 0.4,
        Percentage to interpolate from right valley minima to center peak.

    Returns
    ----------------------
    Tuple with left and right valley (the lungs in a chest xray).
    """
    left_peak, central, right_peak = main_peaks(image)
    inverted_image = image.max() - image
    left_padding = int(central*left_factor+(1-left_factor)*left_peak)
    left_valley = left_padding + central_peak(
        inverted_image[:, left_padding:central],
        use_right_padding=False
    )
    # The right is more towards the center because of the heart
    right_valley = central + central_peak(
        inverted_image[:, central: int(
            right_factor*central+(1-right_factor)*right_peak)],
        use_left_padding=False
    )
    return left_valley, right_valley


def valleys_cut(image: np.ndarray, left_factor: float = 0.25, right_factor: float = 0.4) -> np.ndarray:
    """Return the image with black before and after left and right valleys.
    
    These valleys are detected best on a blurred image.

    Used in get_spinal_cord_mask.py.
    
    Parameters
    ----------------------
    image: np.ndarray,
        The image to apply the valleys cut on.
    left_factor: float = 0.2,
        Percentage to interpolate from left valley minima to center peak.
    right_factor: float = 0.4,
        Percentage to interpolate from right valley minima to center peak.

    Returns
    ----------------------
    Image with areas before and after left and right valleys in black.
    """
    left_valley, right_valley = main_valleys(
        cv2.blur(image, (33, 33)),  # pylint: disable=no-member
        left_factor,
        right_factor
    )
    copy = image.copy()
    copy[:, :left_valley] = 0
    copy[:, right_valley:] = 0
    return copy
