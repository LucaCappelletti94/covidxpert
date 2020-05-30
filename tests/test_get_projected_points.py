from covidxpert.utils import get_projected_points
from typing import List, Set, Dict, Tuple, Optional
import numpy as np
from tqdm.auto import tqdm
import pytest

from covidxpert.utils.test_utils import static_test

def test_get_projected_points():
    l_tests = [{'Input': (0, 0, 0, 0), 'Output': ZeroDivisionError},
               {'Input': (1, 0, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (np.inf, np.inf, np.inf, np.inf), 'Output': (np.inf, 0, np.inf, np.inf)},
               {'Input': (np.inf, 1, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (-np.inf, 1, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (-1, 2, -3, 4), 'Output': (2.0, 0, -2.0, 4)},
               {'Input': (1, 2, 3, 4), 'Output': (-2.0, 0, 2.0, 4)}]
    static_test(get_projected_points,    l_tests)


def test_get_projected_points2():
    key_in: str = 'Input'
    key_out: str = 'Output'
    l_tests = [{'Input': (0, 0, 0, 0), 'Output': ZeroDivisionError},
               {'Input': (1, 0, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (np.inf, np.inf, np.inf, np.inf), 'Output': (np.inf, 0, np.inf, np.inf)},
               {'Input': (np.inf, 1, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (-np.inf, 1, 0, 0), 'Output': (0, 0, 0, 0)},
               {'Input': (-1, 2, -3, 4), 'Output': (2.0, 0, -2.0, 4)},
               {'Input': (1, 2, 3, 4), 'Output': (-2.0, 0, 2.0, 4)}]

    def lazy_isin(x): return next((True for d in l_tests if x in d), False)
    if not lazy_isin(key_in) or not lazy_isin(key_out):
        raise KeyError(f"{key_in} or {key_out} is not a valid key.")

    for test in l_tests:
        print(test)
        if not isinstance(test[key_out], tuple):
            with pytest.raises(test[key_out]):
                get_projected_points(*test[key_in])
        else:
            result = get_projected_points(*test[key_in])
            assert len(result) == len(test[key_out])
            assert all(np.isclose(x, y) for x, y in zip(result, test[key_out]))
