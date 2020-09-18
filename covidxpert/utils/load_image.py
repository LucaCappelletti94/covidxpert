import cv2
import numpy as np
import pydicom as dicom
from .normalize_image import normalize_image


def load_image(path: str) -> np.ndarray:
    """Return normalized image at given path.

    Parameters
    ----------------
    path: str,
        Path to image to be loaded.

    Returns
    ----------------
    Return numpy array containing loaded image.
    """
    if path.endswith('.dcm'):
        ds=dicom.dcmread(path)
        image= ds.pixel_array #cv2.cvtColor(ds.pixel_array, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # pylint: disable=no-member
    return normalize_image(image)
