from covidxpert.utils import get_projected_points
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
from tqdm.auto import tqdm
import pytest

from covidxpert.utils.test_utils import static_test


def test_static_get_projected_points():
    l_tests = [{'Input': (0, 0, 0, 0), 'Output': ZeroDivisionError},
               {'Input': (1, 0, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (np.inf, np.inf, np.inf, np.inf),
                'Output': (np.inf, 0, np.inf, np.inf)},
               {'Input': (np.inf, 1, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (-np.inf, 1, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (-1, 2, -3, 4), 'Output': (2.0, 0, -2.0, 4)},
               {'Input': (1, 2, 3, 4), 'Output': (-2.0, 0, 2.0, 4)}]
    static_test(get_projected_points,    l_tests)


def test_fuzzy_get_projected_points2():
    tests_size = (100, 4)
    min_val, max_val = -1000, 1000

    for m, q, x, h in tqdm(
        np.random.uniform(low=min_val, high=max_val, size=tests_size),
        desc="Fuzzying test for get_projected_points in range"):
        if m == 0:
            with pytest.raises(ZeroDivisionError):
                x0, y0, x1, y1 = get_projected_points(
                    m, q, x, int(round(h, 0)))
        else:
            x0, y0, x1, y1 = get_projected_points(m, q, x, h)
            assert np.isclose(x0, x) if np.isinf(m) else np.isclose(x0, -q / m)
            assert np.isclose(y0, 0)
            assert np.isclose(x1, x) if np.isinf(m) else np.isclose(x1, (h - q) / m)
            assert np.isclose(y1, h)
