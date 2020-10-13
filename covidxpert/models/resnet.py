import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, add

def resnet_block(input_tensor: Input, length: int, **kwargs):
    """Build a resent block.
    
    Arguments
    ---------
    input_tensor: Input,
        The tensorflow input tensor of the model,
    lenght: int,
        How many convolutional layers will be in the block
    **kwargs:
        Arguments to pass to **every** layer of the block"""
    start = hidden = Conv2D(**kwargs)(input_tensor)
    for _ in range(length):
        hidden = Conv2D(padding="same", **kwargs)(hidden)
    return add([start, hidden])

def build_classifier(hidden_layer):
    """Build the classification model from the features exctracted
    by the convolutional layer.
    
    Arguments
    ---------
    hidden_layer: Layer,
        The "input" of the classifier
    """
    o = Dense(128, activation="relu")(hidden_layer)
    o = Dense(64, activation="relu")(o)
    o = Dense(32, activation="relu")(o)
    o = Dense(1, activation="sigmoid")(o)
    return o

def build_resnet(input_shape, resnet_layers):
    """Build a resent as specified in the arguments.
    
    Arguments
    ---------
    input_shape: Tuple[int, int],
        The shape of the input images.
    resnet_layers: Dict[str, *],
        A list of kwargs for each resnet block."""
    i = h = Input(shape=input_shape)
    for layer in resnet_layers:
        h = resnet_block(h, **layer)
    o = build_classifier(Flatten()(h))
    model = Model(i, o)
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )
    return model

def resnet(input_shape):
    return build_resnet(input_shape,
        resnet_layers=[
            {
                "length":3,
                "filters":10,
                "kernel_size":(2, 2)
            },
            {
                "length":4,
                "filters":10,
                "kernel_size":(2, 2)
            }]
        )