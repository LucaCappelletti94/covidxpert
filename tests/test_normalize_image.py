from covidxpert import load_image
from covidxpert.utils import normalize_image
from tqdm.auto import tqdm
from glob import glob


def test_normalize_image():
    for path in tqdm(glob("tests/test_images/*"), desc="Test normalize image"):
        normalized = normalize_image(load_image(path))

        assert normalized.max() == 255
        assert normalized.min() == 0
