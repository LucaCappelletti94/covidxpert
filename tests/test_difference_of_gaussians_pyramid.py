import pytest
from covidxpert.utils import difference_of_gaussians_pyramid


def test_difference_of_gaussians_pyramid():
    """Check that proper exceptions are raised for illegal parameters."""
    with pytest.raises(ValueError):
        difference_of_gaussians_pyramid(None, sigma=-1)
