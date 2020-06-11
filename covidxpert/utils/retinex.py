from typing import List
import numpy as np
import cv2


def single_scale_retinex(image: np.ndarray, sigma: int) -> np.ndarray:
    """
    Performs single-scale retinex in an image

    Parameters
    -----------------------------
    image: np.ndarray,
        Image to be enhanced
    sigma: int
        Sigma to be applied in the log10 function

    Returns
    -----------------------------
    Retinexed image for a given sigma
    """
    retinex = np.log10(image) - np.log10(cv2.GaussianBlur(image, (0, 0), sigma))

    return retinex


def multi_scale_retinex(image: np.ndarray, sigma_list: List) -> np.ndarray:
    """
    Performs multi-scale retinex by wrapping single-scale retinex

    Parameters
    -----------------------------
    image: np.ndarray,
        Image to be enhanced
    sigma_list: List
        Recommended sigma values: [15, 80, 250]

    Returns
    -----------------------------
    Aggregate retinexed image for a given list of sigmas
    """
    retinex = np.zeros_like(image)
    for sigma in sigma_list:
        retinex += single_scale_retinex(image, sigma)

    return retinex / len(sigma_list)


def automated_msrcr(image: np.ndarray, sigma_list: List) -> np.ndarray:
    """
    Automated image enhancement using retinex

    Parameters
    -----------------------------
    image: np.ndarray,
        Image to be enhanced
    sigma_list: List
        Recommended sigma values: [15, 80, 250]

    Returns
    -----------------------------
    Enhanced image
    """
    image = np.float64(image) + 1.0
    img_retinex = multi_scale_retinex(image, sigma_list)

    unique, count = np.unique(np.int32(img_retinex * 100), return_counts=True)
    zero_count = count[np.argwhere(unique == 0)]

    low_val = unique[0] / 100.0
    high_val = unique[-1] / 100.0
    for u, c in zip(unique, count):
        if u < 0 and c < zero_count * 0.1:
            low_val = u / 100.0
        if u > 0 and c < zero_count * 0.1:
            high_val = u / 100.0
            break

    img_retinex = np.maximum(np.minimum(img_retinex, high_val), low_val)
    img_retinex = (img_retinex - np.min(img_retinex)) / (np.max(img_retinex) - np.min(img_retinex)) * 255
    img_retinex = np.uint8(img_retinex)

    return img_retinex
