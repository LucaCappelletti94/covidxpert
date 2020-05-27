from covidxpert.utils import compute_linear_coefficients
import numpy as np
from tqdm.auto import tqdm


def test_compute_linear_coefficients():
    m, q = compute_linear_coefficients(0, 0, 0, 0)
    assert np.isinf(m)
    assert q == 0

    m, q = compute_linear_coefficients(1, 1, 1, 2)
    assert np.isinf(m)
    assert q == 1

    m, q = compute_linear_coefficients(0, 1, 1, 2)
    assert np.isclose(m, 1)
    assert np.isclose(q, 1)

    m, q = compute_linear_coefficients(0, 7, 5, 7)
    assert np.isclose(m, 0)
    assert np.isclose(q, 7)

    m, q = compute_linear_coefficients(0, 5, 1, 10)
    assert np.isclose(m, 5)
    assert np.isclose(q, 5)

    for x0, y0, x1, y1 in tqdm(
        np.random.uniform(-1000, 1000, (100, 4)), 
        desc = "fuzzyng test for computed linear coefficients"
    ):

        m, q = compute_linear_coefficients(x0, y0, x1, y1)
        if np.isclose(x0, x1):
            assert np.isinf(m)
            assert np.isclose(x0, q)
        else:
            assert np.isclose(y0-(m*x0+q), 0)
            assert np.isclose(y1-(m*x1+q), 0)
        
        m, q = compute_linear_coefficients(x1, y0, x0, y1)
        if np.isclose(x0, x1):
            assert np.isinf(m)
            assert np.isclose(x0, q)
        else:
            assert np.isclose(y0-(m*x1+q), 0)
            assert np.isclose(y1-(m*x0+q), 0)
