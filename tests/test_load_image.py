from covidxpert.utils import load_image
from .utils import multiprocessing_execution


def test_load_image():
    multiprocessing_execution(load_image)