from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, GlobalAveragePooling2D

from typing import Tuple

def load_keras_model(keras_model: Model, img_shape: Tuple[int, int]):
    """Adapt a keras model for our task making them accept a 
        gray-scale image and adding a basic mlp on the top of it.
    
    Arguments
    ---------
    keras_model: tensorflow.keras.models.Model,
        One of the model specified at https://keras.io/api/applications/
        They can be found under tf.keras.applications.
    img_shape: Tuple[int, int],
        The shape of the image.
    """

    i = Input(shape=img_shape)
    # All the models in keras.applications expects Rgb images, so we fix the shape
    # by adding an extra convolution at the start.
    h = Conv2D(3, kernel_size=(1, 1))(i)

    # Initialize the model
    kmodel = keras_model(
        input_shape=(*img_shape[:2], 3),
        include_top=False,
        weights="imagenet",
        classes=2,
        pooling="avg",
    )

    o = Dense(1, activation="sigmoid")(kmodel(h))

    # Compile the model
    model = Model(i, o, name=kmodel.name)
    return model