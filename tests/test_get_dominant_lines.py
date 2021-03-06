import numpy as np
from covidxpert.counter_rotate.get_dominant_lines import get_dominant_lines


def test_get_dominant_lines():

    valid_line = [
        [[0.5, 0, 0.5, 10]]  # vertical line in the middle
    ]

    assert tuple(*valid_line[0]) == next(get_dominant_lines(valid_line, 10, 1))

    line_on_side = [
        [[0, 0, 10, 1]],  # diagonal
        [[0, 0.5, 10, 0.5]],  # horizzontal line
        [[1.9, 0, 6.1, 1]],  # almost vertical line on the side
        [[0, 0, 1, 1]],  # not in the center
        [[4.1, 0, 5.9, 1]]  # inclined line
    ]

    assert not list(get_dominant_lines(
        line_on_side, 1, 10, max_inclination=89))

    # Fuzzying
    list(get_dominant_lines(
        np.random.uniform(0, 10, size=(1000, 1, 4)),
        height=10,
        width=10
    ))
