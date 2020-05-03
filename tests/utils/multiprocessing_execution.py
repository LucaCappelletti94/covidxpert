from multiprocessing import Pool, cpu_count
from typing import Callable, List
from tqdm.auto import tqdm
from glob import glob


def multiprocessing_execution(test: Callable):
    """Executing test execution in parallel."""
    paths = glob("sample_dataset/*")
    with Pool(cpu_count()) as p:
        list(tqdm(
            p.imap(test, paths),
            total=len(paths),
            desc="Executing test"
        ))
        p.close()
        p.join()