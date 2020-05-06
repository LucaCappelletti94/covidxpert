import numpy as np
import cv2

def get_masked_image(image: np.ndarray):
    _, thresholded_mask = cv2.threshold(image, 0, 255, cv2.THRESH_TRIANGLE)

    _, output, stats, _ = cv2.connectedComponentsWithStats(
        thresholded_mask,
        connectivity=8
    )

    max_sizeR = np.argmax(stats[1:, -1])
    image_mask = np.zeros((output.shape), dtype=np.uint8)
    image_mask[output == max_sizeR + 1] = 255

    # We determine the contours of the mask
    contours, _ = cv2.findContours(
        image=image_mask,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_NONE
    )

    # And fill up the mask within thr contours
    # as they might remain holes within.
    image_mask = cv2.fillPoly(
        image_mask,
        pts=[contours[0]],
        color=(255, 255, 255)
    )

    return image_mask