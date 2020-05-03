from covidxpert.utils import load_image, convex_mask, cut_bounding_box
from .utils import multiprocessing_execution


def job(path: str):
    image = load_image(path)
    cut_bounding_box(image, convex_mask(image))


def test_cut_bounding_box():
    multiprocessing_execution(job)
