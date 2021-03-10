import pytest
from covidxpert import load_efficientnet_model
from tensorflow.python.keras.engine.functional import Functional

def test_efficientnet():
    model = load_efficientnet_model((300,300,3))
    assert type(model) == Functional