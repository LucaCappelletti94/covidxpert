import numpy as np
import cv2
from typing import Tuple


def rotate_image(image: np.ndarray, angle: float, center: Tuple[int, int] = None) -> np.ndarray:
    # TODO! Write the docstring of this method! 
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    if center is None:
        (x, y) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((x, y), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    new_width = int((h * sin) + (w * cos))
    new_height = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (new_width, new_height))
