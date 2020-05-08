from covidxpert import load_image, perspective_correction
from covidxpert.utils import remove_artefacts, normalize_image
from tqdm.auto import tqdm
from glob import glob


def test_normalize_image():
    for path in tqdm(glob("tests/test_images/*")):
        original = load_image(path)
        cut_image = perspective_correction(original)
        cleared_image = remove_artefacts(cut_image)

        assert cleared_image.max() == 255
        assert cleared_image.min() == 0
