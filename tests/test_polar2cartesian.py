from typing import Generator
import numpy as np
from covidxpert.utils import polar2cartesian


def test_polar2cartesian():
    polar_array = np.ones((5, 1, 2))
    cartesian_array = polar2cartesian(polar_array)

    assert isinstance(cartesian_array, Generator)
    for cartesian_point in cartesian_array:
        assert isinstance(cartesian_point, tuple)

    polar_array = np.array([[[5, 60]], [[1000, 190]], [[-5, -15]]])
    cartesian_array_2 = polar2cartesian(polar_array)

    assert next(cartesian_array_2)[0] == ((np.cos(60) * 5) - np.sin(60),
                                          (np.sin(60) * 5) + np.cos(60),
                                          (np.cos(60) * 5) + np.sin(60),
                                          (np.sin(60) * 5) - np.cos(60))
    assert next(cartesian_array_2)[0] == ((np.cos(190) * 1000) - np.sin(190),
                                          (np.sin(190) * 1000) + np.cos(190),
                                          (np.cos(190) * 1000) + np.sin(190),
                                          (np.sin(190) * 1000) - np.cos(190))
    assert next(cartesian_array_2)[0] == ((np.cos(-15) * -5) - np.sin(-15),
                                          (np.sin(-15) * -5) + np.cos(-15),
                                          (np.cos(-15) * -5) + np.sin(-15),
                                          (np.sin(-15) * -5) - np.cos(-15))
