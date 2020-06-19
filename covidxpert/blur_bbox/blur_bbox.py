import numpy as np
from .get_kernel_size import get_kernel_size
from ..utils import trim_padding, add_padding, remove_artefacts
import cv2
from typing import List, Union, Tuple


def count_from_left_side(mask: np.ndarray):
    counter = 0
    for boolean in mask:
        if boolean:
            counter += 1
        else:
            break
    return counter


def count_from_right_side(mask: np.ndarray):
    return count_from_left_side(np.flip(mask, axis=0))


def build_slice(left: int, right: int, maximum: int):
    return slice(left, maximum if right == 0 else right)


def strip_sides(image: np.ndarray, flat_mask: np.ndarray) -> np.ndarray:
    return build_slice(
        count_from_left_side(flat_mask),
        -count_from_right_side(flat_mask),
        flat_mask.size
    )


def strip_black(image: np.ndarray, mask: np.ndarray, v_threshold: float, h_threshold: float) -> np.ndarray:
    vertical_mask = mask.mean(axis=1) <= v_threshold
    horizzontal_mask = mask.mean(axis=0) <= h_threshold

    return image[strip_sides(image, vertical_mask), strip_sides(image, horizzontal_mask)]


def compute_median_threshold(mask: np.ndarray) -> Tuple[float, float]:
    masked_mask = strip_black(mask, mask, 0, 0)
    v_white_median = np.median(masked_mask.mean(axis=0))
    h_white_median = np.median(masked_mask.mean(axis=1))
    return v_white_median/2, h_white_median/2


def get_blur_mask(image: np.ndarray, padding: int):
    blurred = add_padding(image, padding)
    blurred = remove_artefacts(blurred)
    kernel = get_kernel_size(blurred)
    blurred = cv2.medianBlur(blurred, kernel)
    blurred = cv2.threshold(blurred, np.median(
        blurred)/2, 255, cv2.THRESH_BINARY)[1]
    return trim_padding(blurred, padding)


def blur_bbox(image: np.ndarray, padding: int = 50, others: List[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    mask = get_blur_mask(image, padding)
    result = strip_black(image, mask, *compute_median_threshold(mask))

    if others is None:
        return result

    return result, [strip_black(other, mask, *compute_median_threshold(mask)) for other in others]
