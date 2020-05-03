from covidxpert.utils import load_image, convex_mask
from .utils import multiprocessing_execution


def job(path: str):
    convex_mask(load_image(path))


def test_convex_mask():
    multiprocessing_execution(job)
