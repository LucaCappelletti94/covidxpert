from covidxpert.rotate import get_rotation
import matplotlib.pyplot as plt
import numpy as np
import cv2


def test_get_rotation():
    line_image = cv2.cvtColor(
        np.ones((500, 500), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.line(line_image, (20, 20), (260, 500), (255, 255, 255), 1, cv2.LINE_AA)
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    line_image[line_image<10] = 0

    rotation, result, composite = get_rotation(line_image)
    fig, axes = plt.subplots(ncols=3)
    axes = axes.flatten()
    axes[0].imshow(line_image, cmap="gray")
    axes[1].imshow(result, cmap="gray")
    axes[2].imshow(composite, cmap="gray")
    fig.savefig("test.jpg")