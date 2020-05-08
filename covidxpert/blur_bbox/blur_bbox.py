import numpy as np
from .get_kernel_size import get_kernel_size
from ..utils import trim_padding, add_padding, remove_artefacts
import cv2


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


def strip_black(image: np.ndarray, mask: np.ndarray, v_threshold: float, h_threshold: float) -> np.ndarray:
    vertical_mask = mask.mean(axis=1) < v_threshold
    horizzontal_mask = mask.mean(axis=0) < h_threshold

    h_slice = build_slice(
        count_from_left_side(horizzontal_mask),
        -count_from_right_side(horizzontal_mask),
        image.shape[1]
    )
    v_slice = build_slice(
        count_from_left_side(vertical_mask),
        -count_from_right_side(vertical_mask),
        image.shape[0]
    )
    return image[v_slice, h_slice]


def compute_median_threshold(mask: np.ndarray) -> float:
    masked_mask = strip_black(mask, mask, 0, 0)
    v_white_median = np.median(masked_mask.mean(axis=0))
    h_white_median = np.median(masked_mask.mean(axis=1))
    return v_white_median/2, h_white_median/2


def get_blur_mask(image: np.ndarray, padding: int):
    blurred = add_padding(image, padding)
    blurred, _ = remove_artefacts(blurred)
    kernel = get_kernel_size(blurred)
    blurred = cv2.medianBlur(blurred, kernel)
    blurred = cv2.threshold(blurred, np.median(
        blurred)/2, 255, cv2.THRESH_BINARY)[1]
    return trim_padding(blurred, padding)


def blur_bbox(image: np.ndarray, padding: int = 50) -> np.ndarray:
    mask = get_blur_mask(image, padding)
    return strip_black(image, mask, *compute_median_threshold(mask))
