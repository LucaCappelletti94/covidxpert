from ..utils import (
    normalize_image, add_padding, trim_padding, darken, median_mask, trim_flip, fill_small_black_blobs, fill_small_white_blobs,
    fill_lower_max, histogram_based_vertical_thresholding, get_thumbnail, rotate_image)
import numpy as np
import cv2
from typing import List, Union, Tuple

import matplotlib.pyplot as plt


def get_complete_body_mask(image: np.ndarray, width=256) -> np.ndarray:
    # Getting the rotated darkened thumb image
    thumb = get_thumbnail(image, width)
    # Computing mask
    body_mask = normalize_image(median_mask(
        thumb, np.median(thumb[thumb > 0]), factor=6))
    body_mask = add_padding(body_mask, 20)
    body_mask = cv2.morphologyEx(  # pylint: disable=no-member
        body_mask,
        cv2.MORPH_CLOSE,  # pylint: disable=no-member
        np.ones((9, 3))
    )
    body_mask = fill_small_black_blobs(body_mask, 10)
    body_mask = fill_small_white_blobs(body_mask, 10)
    body_mask = trim_padding(body_mask, 20)
    body_mask = cv2.erode(  # pylint: disable=no-member
        body_mask,
        np.ones((3, 3)),
        iterations=20
    )
    body_mask = add_padding(trim_padding(body_mask, 5), 5)
    return cv2.resize(  # pylint: disable=no-member
        body_mask,
        (image.shape[1], image.shape[0])
    )


def get_bounding_box(mask, step=20) -> float:
    best_score = 0
    best_y = 0
    height = mask.shape[0]
    for lower_y in range(step, height, step):
        rectangle = np.zeros_like(mask, dtype=np.bool_)
        rectangle[height-lower_y:height] = True
        score = mask[rectangle].sum() - (~mask[rectangle]).sum()*20
        if score > best_score:
            best_y = lower_y
            best_score = score
    return best_y


def get_body_cut(image: np.ndarray, rotated: np.ndarray, angle: float, simmetry_axis: int, hardness: float = 0.75, width: int = 256, others: List[np.ndarray] = None) -> Union[np.ndarray, np.narray, Tuple[np.ndarray, np.ndarray]]:
    rotated_darken = rotate_image(darken(image), angle)
    body = get_complete_body_mask(rotated_darken, width=width)
    median = np.median(rotated_darken[body != 0])
    copy = rotated_darken.copy()
    copy[body == 0] = 255

    mask = normalize_image(median_mask(copy, median=median, factor=1.25))
    mask = fill_lower_max(mask)
    left, right = trim_flip(mask, int(simmetry_axis*mask.shape[1]))
    mask = left & right
    mask = mask > 0

    best_y = get_bounding_box(mask)

    body_slice = slice(0, int(-best_y*hardness))

    result = rotated[body_slice], rotated_darken[body_slice]
    if others is None:
        return result

    return result, [(other[body_slice], rotate_image(darken(other), angle)[body_slice]) for other in others]
