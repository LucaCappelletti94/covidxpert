from covidxpert.utils import load_image, convex_mask
from tqdm.auto import tqdm
from glob import glob


def test_load_image():
    for path in tqdm(glob("sample_dataset/*")):
        convex_mask(load_image(path))
