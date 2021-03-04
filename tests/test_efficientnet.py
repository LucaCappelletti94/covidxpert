import pytest
from covidexpert import load_efficientnet_model
from tensorflow.keras.models import Model

def test_efficientnet():
    model = load_efficientnet_model((300,300,3))
    assert type(model)==Model