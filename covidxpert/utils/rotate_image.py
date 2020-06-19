import numpy as np
import cv2


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Return image rotated of given amount.

    Parameters
    ----------------------
    image: np.ndarray,
        Image to be rotate.
    angle: float ,
        Amount of rotation to be applied.

    Returns
    ----------------------
    Rotated image.
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    x = w // 2
    y = h // 2
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((x, y), -angle, 1.0)  # pylint: disable=no-member
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    new_width = int((h * sin) + (w * cos))
    new_height = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (new_width, new_height))  # pylint: disable=no-member
