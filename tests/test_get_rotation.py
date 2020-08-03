import numpy as np
import cv2
from covidxpert.counter_rotate.get_lines_based_rotation import get_lines_based_rotation, normalize_angle
from covidxpert.utils import compute_linear_coefficients


def test_get_rotation():
    lines = [
        ((250, 0), (250, 500)),
        ((240, 0), (250, 500)),
        ((240, 0), (260, 500)),
        ((250, 0), (260, 500)),
        ((200, 0), (400, 500)),
    ]
    baseline = np.zeros((500, 500), dtype=np.uint8)
    for p1, p2 in lines:
        image = cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR) # pylint: disable=no-member
        cv2.line(image, p1, p2, (255, 255, 255), 5, cv2.LINE_AA) # pylint: disable=no-member
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member

        expected = normalize_angle(np.degrees(np.arctan(
            compute_linear_coefficients(*p1, *p2)[0]
        )))
        rotation = get_lines_based_rotation(image)
        assert np.isclose(expected, rotation, atol=2)

    rotation = get_lines_based_rotation(baseline)
    assert np.isclose(normalize_angle(0), rotation, atol=2)
