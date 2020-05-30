from covidxpert.rotate import detect_almost_vertical_lines

import numpy as np
import cv2


def test_detect_almost_vertical_lines():
    line_image = cv2.cvtColor(np.ones((500, 5000), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    line_image = cv2.line(line_image, (0, 0), (100, 100), (1, 1, 1), 5, cv2.LINE_AA)

    returned_image = detect_almost_vertical_lines(cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY))
    # print(list(returned_image))
