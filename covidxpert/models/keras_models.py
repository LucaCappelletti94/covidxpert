from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, GlobalAveragePooling2D
from extra_keras_metrics import get_complete_binary_metrics

from typing import Tuple

def load_keras_model(keras_model: Model, img_shape: Tuple[int, int], nadam_kwargs=None):
    """Adapt a keras model for our task making them accept a 
        gray-scale image and adding a basic mlp on the top of it.
    
    Arguments
    ---------
    keras_model: tensorflow.keras.models.Model,
        One of the model specified at https://keras.io/api/applications/
        They can be found under tf.keras.applications.
    img_shape: Tuple[int, int],
        The shape of the image.
    nadam_kwargs: dict,
        The keywords aaguments to be passed to the Nadam Optimizer.
    """
    # Use an empty dict as default avoiding the quirks of having a mutable default.
    if nadam_kwargs is None:
        nadam_kwargs = {}

    i = Input(shape=img_shape)
    # All the models in keras.applications expects Rgb images, so we fix the shape
    # by adding an extra convolution at the start.
    h = Conv2D(3, kernel_size=(1, 1))(i)

    # Initialize the model
    kmodel = keras_model(
        input_tensor=h,
        include_top=False,
        weights="imagenet",
        classes=2,
        pooling="avg",
    )

    o = Dense(1, activation="sigmoid")(kmodel.output)

    # Compile the model
    model = Model(i, o, name=kmodel.name)
    model.compile(
        optimizer=Nadam(**nadam_kwargs),
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )
    return model