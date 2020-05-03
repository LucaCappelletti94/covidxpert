from covidxpert.utils import load_image
from covidxpert import perspective_correction
from .utils import multiprocessing_execution


def job(path: str):
    perspective_correction(load_image(path))


def test_perspective_correction():
    multiprocessing_execution(job)
