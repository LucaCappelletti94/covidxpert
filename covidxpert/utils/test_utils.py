from typing import List, Set, Dict, Tuple, Optional, Any
import numpy as np
from tqdm.auto import tqdm
import pytest

def static_test(f: Any, l_tests: List[Dict[str, Tuple]], 
                key_in:str='Input', key_out:str='Output'):        
    lazy_isin = lambda x: next((True for d in l_tests if x in d), False)
    if not lazy_isin(key_in) or not lazy_isin(key_out):
        raise KeyError(f"{key_in} or {key_out} is not a valid key.")

    for test in l_tests:
        print(test)

        if not isinstance(test[key_out], tuple):
            with pytest.raises(test[key_out]):
                f(*test[key_in])
        else:
            result = f(*test[key_in])
            assert len(result) == len(test[key_out])
            assert all(np.isclose(x, y) for x, y in zip(result, test[key_out]))
