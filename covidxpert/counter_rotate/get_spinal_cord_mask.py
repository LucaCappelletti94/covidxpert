import numpy as np
from ..utils import difference_of_gaussians_pyramid, normalize_image, median_mask, darken, valleys_cut


def get_spinal_cord_mask(image: np.ndarray, left_factor: float = 0.2, right_factor: float = 0.4) -> np.ndarray:
    """Return mask with the spinal cord.

    Parameters
    ------------------
    image: np.ndarray,
        Chest-xray image.
    left_factor: float = 0.2,
        Percentage to interpolate from left valley minima to center peak.
    right_factor: float = 0.4,
        Percentage to interpolate from right valley minima to center peak.

    Returns
    ------------------
    Return spinal cord mask.
    """
    # Getting the DOG mask from the original image.
    dog_image, _ = difference_of_gaussians_pyramid(darken(image), 1, 5, 50)
    # Applying median mask and normalizing obtained mask.
    dog_image = normalize_image(~median_mask(dog_image))
    # Applying valleys based cut.
    return valleys_cut(dog_image, left_factor, right_factor)
