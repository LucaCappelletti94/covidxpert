from glob import glob
from tqdm.auto import tqdm
from covidxpert.utils import load_image


def test_normalize_image():
    for path in tqdm(glob("tests/test_images/*"), desc="Test normalize image"):
        normalized = load_image(path)

        assert normalized.max() == 255
        assert normalized.min() == 0
