import numpy as np
import cv2


def get_masked_image(image: np.ndarray):
    # Applying the threshold
    _, thresholded_mask = cv2.threshold(  # pylint: disable=no-member
        image,
        image.min(),
        255,
        cv2.THRESH_BINARY  # pylint: disable=no-member
    )

    _, output, stats, _ = cv2.connectedComponentsWithStats(  # pylint: disable=no-member
        thresholded_mask,
        connectivity=8
    )

    image_mask = np.zeros((output.shape), dtype=np.uint8)
    image_mask[output == np.argmax(stats[1:, -1]) + 1] = 255

    # We determine the contours of the mask
    contours, _ = cv2.findContours(  # pylint: disable=no-member
        image=image_mask,
        mode=cv2.RETR_TREE,  # pylint: disable=no-member
        method=cv2.CHAIN_APPROX_NONE  # pylint: disable=no-member
    )

    # And fill up the mask within thr contours
    # as they might remain holes within.
    image_mask = cv2.fillPoly(  # pylint: disable=no-member
        image_mask,
        pts=[contours[0]],
        color=255
    )

    # Computing the chull of the mask
    chull = cv2.convexHull(contours[0])  # pylint: disable=no-member

    merged_mask = cv2.fillPoly(  # pylint: disable=no-member
        image_mask, [chull], 255)

    return merged_mask
