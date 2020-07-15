import numpy as np
import cv2
from .normalize_image import normalize_image


def difference_of_gaussians_pyramid(
        image: np.ndarray,
        sigma: float = 1,
        start_sigma: float = 5,
        end_sigma: float = 50,
        steps: int = 50
) -> np.ndarray:
    """Returns the inverted binary multiscale difference of Gaussian.

    Parameters
    ------------------
    image: np.ndarray,
        The image for which to obtain the kernel.
    sigma: float,
        standard deviation for smoothing the image (if sigma_image = 0 no smoothing is performed)
        # TODO: add range for this parameter.
    start_sigma: float,
        starting std for Gaussian blurring
        # TODO: add range for this parameter.
    end_sigma: float,
        end-sigma for Gaussian blurring differences of Gaussians
        # TODO: add range for this parameter.
    steps: int,
        Steps of linrange to consider from sigma start to sigma end.
        # TODO: add range for this parameter.

    Raises
    ------------------
    # TODO: Add for what we get an exception.

    Returns
    ------------------
    img_sum: np.ndarray,
        returns the sum image
    """

    # TODO: Add exceptions for invalid parameter (negative etc..)
    check_parameters(image, sigma, start_sigma, end_sigma, steps)

    # Normalizing the provided image.
    image = cv2.normalize(  # pylint: disable=no-member
        image.astype(float),
        None,
        0,
        255,
        cv2.NORM_MINMAX  # pylint: disable=no-member
    )

    # Initializing the background and foreground
    backgrounds = np.zeros_like(image, dtype=np.uint8)
    foregrounds = np.zeros_like(image, dtype=np.uint8)

    # If required
    if sigma > 0:
        # We smooth the input image.
        image = cv2.GaussianBlur(  # pylint: disable=no-member
            image,
            (0, 0),
            sigma,
            cv2.BORDER_REPLICATE  # pylint: disable=no-member
        )

    for sigma in np.linspace(start_sigma, end_sigma, steps):
        # Applying blur to the provided image with the current step sigma.
        blur = cv2.GaussianBlur( # pylint: disable=no-member
            image,
            (0, 0),
            sigma,
            cv2.BORDER_REPLICATE # pylint: disable=no-member
        )
        # Compute diffences between gaussian blur and provided image.
        subtraction = image - blur
        # Summing obtained background mask to backgrounds.
        backgrounds[subtraction < 0] += 1
        # Summing obtained foreground mask to foregrounds.
        foregrounds[subtraction > 0] += 1

    return normalize_image(backgrounds), normalize_image(foregrounds)


def check_parameters(image: np.ndarray, sigma: float, start_sigma: float, end_sigma: float, steps: int):
    if sigma < 0:
        raise ValueError('sigma < 0')
