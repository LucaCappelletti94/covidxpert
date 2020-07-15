import numpy as np
import cv2


def get_thumbnail(image: np.ndarray, width: int):
    if image.shape[1] <= width:
        return image
    height = int(image.shape[0] * width/image.shape[1])
    return cv2.resize(image, (width, height), cv2.INTER_AREA)  # pylint: disable=no-member
