import numpy as np
from covidxpert.utils import get_thumbnail


def test_thumbnail():
    """Check that no resize happens when image is smaller than target."""
    image = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    assert (image == get_thumbnail(image, 200)).all()
