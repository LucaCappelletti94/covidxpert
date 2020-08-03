import pytest
from covidxpert.utils import fill_small_black_blobs, fill_small_white_blobs


def test_fill_artefacts():
    """Check that proper exceptions are raised for illegal parameters."""
    with pytest.raises(ValueError):
        fill_small_white_blobs(None, -1)

    with pytest.raises(ValueError):
        fill_small_black_blobs(None, -1)
