import cv2
import numpy as np
from colour import cctf_encoding
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear, demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)


def demosaicking(image: np.ndarray, method: str = "bilinear", pattern: str = "RGGB") -> np.ndarray:
    """Returns the demosaicked image given a method.

    Parameters
    -------------------
    image: np.ndarray,
        The image to be demosaicked.
    method: str,
        Demosaicking method to be applied.
    pattern: str,
        Arrangement of the colour filters on the pixel array.
        Possible patterns are: {RGGB, BGGR, GRBG, GBRG}.

    Raises
    ------------------
    ValueError,
        If given method does not exist.

    Returns
    -------------------
    Returns the demosaicked image.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_cfa = mosaicing_CFA_Bayer(image_rgb, pattern=pattern) / 255

    if method == 'bilinear':
        image_demo = (
                (
                    cctf_encoding(
                        demosaicing_CFA_Bayer_bilinear(image_cfa, pattern=pattern)
                    )
                ) * 255
        ).astype(np.uint8)
    elif method == 'malvar':
        image_demo = (
                (
                    cctf_encoding(
                        demosaicing_CFA_Bayer_Malvar2004(image_cfa, pattern=pattern)
                    )
                ) * 255
        ).astype(np.uint8)
    elif method == 'menon':
        image_demo = (
                (
                    cctf_encoding(
                        demosaicing_CFA_Bayer_Menon2007(image_cfa, pattern=pattern)
                    )
                ) * 255
        ).astype(np.uint8)
    else:
        raise ValueError(
            'Given method \'{}\' does not belong to possible methods. '
            'Valid methods are: \'bilinear\', \'malvar\' and \'menon\'.'.format(method)
        )

    return cv2.cvtColor(image_demo, cv2.COLOR_RGB2GRAY)

