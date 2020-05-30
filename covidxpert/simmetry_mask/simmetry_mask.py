from covidxpert import load_image, perspective_correction, blur_bbox
from covidxpert.utils import remove_artefacts, normalize_image, add_padding, trim_padding, inpaint
from covidxpert.blur_bbox.blur_bbox import strip_black, compute_median_threshold
from covidxpert.perspective_correction.get_corners import get_cardinal_corner_points
from glob import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Tuple
from ..utils import darken


def fill_lower_max(image: np.ndarray, lower_padding: int = 50) -> np.ndarray:
    half_image = np.zeros_like(image)
    half = half_image.shape[0]//2
    half_image[half:-lower_padding] = image[half:-lower_padding]
    argmax = np.argmax(half_image.mean(axis=1))
    half_image[argmax:] = half_image[argmax]
    image = image.copy()
    image[half_image > 0] = 255
    return image


def get_default_mask(image: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """Apply default mask over given image.

    Parameters
    ----------------
    image:np.ndarray,
        The image over which to apply the mask.
    borders:np.ndarray
        The borders to use.

    Returns
    ----------------
    The mask obtained over the image.
    """
    mask = cv2.threshold(image, np.median(image), 255, cv2.THRESH_BINARY)[1]
    mask[borders > 0] = 255
    width = mask.shape[1]
    mask[:, int(width*0.45):int(width*0.55)] = 255
    return mask


def get_simmetry_mask(image: np.ndarray) -> np.ndarray:
    cleared_artefacts = remove_artefacts(image)
    x = simmetry_axis(cleared_artefacts)
    borders = get_borders(cleared_artefacts, x)
    cut_image, flipped = trim_flip(darken(cleared_artefacts, 2, (9, 9)), x)

    reflected_product = normalize_image(
        cut_image.astype(float) * flipped.astype(float))
    reflected_product_mask = get_default_mask(reflected_product, borders)
    reflected_product_mask = remove_small_artefacts(
        reflected_product_mask, 200)
    reflected_product_mask = cv2.erode(
        reflected_product_mask, kernel=np.ones((3, 3)), iterations=3)
    reflected_product_mask = remove_small_artefacts(
        reflected_product_mask, 200)
    reflected_product_mask = fill_in_small_artefacts(
        reflected_product_mask, 100)
    reflected_product_mask = fill_lower_max(reflected_product_mask)
    second_product_mask = fill_in_small_artefacts(reflected_product_mask, 20)

    darkened_image = darken(cleared_artefacts, 5, (15, 15))
    darkened_image[darkened_image < np.median(darkened_image)] = 0
    darkened_image = remove_small_artefacts(darkened_image, 200)
    cut_image, flipped = trim_flip(darkened_image, x)

    second_product = normalize_image(
        cut_image.astype(float) * flipped.astype(float))
    second_product_mask = get_default_mask(second_product, borders)
    second_product_mask = fill_in_small_artefacts(second_product_mask, 20)
    second_product_mask = cv2.dilate(
        second_product_mask, kernel=np.ones((3, 3)))
    second_product_mask = fill_in_small_artefacts(second_product_mask, 20)
    second_product_mask = remove_small_artefacts(second_product_mask, 100)
    second_product_mask = fill_lower_max(second_product_mask)
    second_product_mask = fill_in_small_artefacts(second_product_mask, 20)

    composite_mask = normalize_image(reflected_product_mask.astype(
        float) + second_product_mask.astype(float))
    composite_mask = cv2.threshold(
        composite_mask, 0, 255, cv2.THRESH_BINARY)[1]
    composite_mask = cv2.medianBlur(composite_mask, 75)
    composite_mask = fill_in_small_artefacts(composite_mask, 20)
    composite_mask = remove_small_artefacts(composite_mask, 20)

    return composite_mask
