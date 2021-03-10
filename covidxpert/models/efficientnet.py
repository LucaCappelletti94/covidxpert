from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from .metrics import *
from typing import Tuple
from .keras_models import load_keras_model

def top_model_function(h):
   """
   It is used by load_keras_model. 
   It adds layers to the classification model.

   Arguments
   ---------

   h: kerasTensor

   """
   h = BatchNormalization()(h)
   h = Dropout(0.2)(h)
   h = Dense(128, activation="relu")(h)
   h = Dropout(0.2)(h)
   h = Dense(64, activation="relu")(h)

   return h

def load_efficientnet_model(img_shape: Tuple[int, int, int]):
    """ 
    Build an efficientnet.

    Arguments
    ---------

    img_shape: Tuple[int,int, int],
        The shape of image. 

    """
    return load_keras_model( EfficientNetB3,  img_shape, top_model_function, None)
    
