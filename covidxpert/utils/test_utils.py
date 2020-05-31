from typing import List, Set, Dict, Tuple, Optional, Callable
import numpy as np
from tqdm.auto import tqdm
import pytest

def static_test(f: Callable, l_tests: List[Dict[str, Tuple]], 
                key_in:str='Input', key_out:str='Output'):   
    """Validates the 'f' function on the list of tests 'l_tests'

    Parameters
    --------------------
    f: Callable,
        Function to be tested
    l_tests: List[Dict[str, Tuple]],
        List of dictionaries containing inputs for the function 'f' and the expected outputs
    key_in:str='Input'
        Input key for test dictionaries
    key_out:str='Output'
        Output key for test dictionaries
    """
    lazy_isin = lambda x: next((True for d in l_tests if x in d), False)
    if not lazy_isin(key_in) or not lazy_isin(key_out):
        raise KeyError(f"{key_in} or {key_out} is not a valid key.")

    for test in l_tests:
        if not isinstance(test[key_out], tuple):
            with pytest.raises(test[key_out]):
                f(*test[key_in])
        else:
            result = f(*test[key_in])
            assert len(result) == len(test[key_out])
            assert all(np.isclose(x, y) for x, y in zip(result, test[key_out]))
