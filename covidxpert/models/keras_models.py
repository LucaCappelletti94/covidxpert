from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC

from .metrics import *

def load_keras_model(keras_model, img_shape, top_model_function, weights=None):
    """The function accepts a gray-scale image, it applies a convolutional layer
        with three filters to the image. 
        Then, it applies the Keras model provided as an argument by the user.
        To the output of this model, we apply a Global Average Pooling. 
        The output of the Global Average Pooling is the input of the 
        top_model_function. 
        The top_model_function adds layers to the classification model. 
        On top of the classification model, the function adds a Dense layer 
        with a sigmoid as its activation function.
        Finally, it returns the compiled model ready to be trained. 

    +
    Arguments
    ---------
    keras_model,
        One of the model specified at https://keras.io/api/applications/
        They can be found under tf.keras.applications.
    img_shape: Tuple[int, int],
        The shape of the image.
    weights: str
        The weights of keras model. 
        One of None (random initialization), 'imagenet' (pre-training on 
        ImageNet), or the path to the weights file to be loaded. 
        Defaults to None. 
    top_model_function: Callable[ KerasTensor, KerasTensor]
        Adds layers to the classification model.
    """
    i = Input(shape=img_shape)
    h = Conv2D(3, kernel_size=(1, 1))(i)

    kmodel = keras_model(
        input_tensor=h,
        include_top=False,
        weights=weights,
        classes=2,
    )

    o = GlobalAveragePooling2D()(kmodel.output)
    o = top_model_function(o)
    o = Dense(1, activation="sigmoid")(o)

    model = Model(i, o)

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            Accuracy(name="accuracy"),
            BalancedAccuracy(name="BA"),
            AUC(curve="PR", name="AUPRC"),
            AUC(curve="ROC", name="AUROC"),
            F1Score(name="f1-score"),
            MatthewsCorrelationCoefficinet(name="MCC"),
            Recall(name="recall"),
            Specificity(name="specificity"),
            Precision(name="precision"),
            NegativePredictiveValue(name="NPV"),
            MissRate(name="missrate"),
            FallOut(name="fallout"),
            FalseDiscoveryRate(name="FDR"),
            FalseOmissionRate(name="FOR"),
            PrevalenceThreshold(name="PT"),
            ThreatScore(name="TS"),
            FowlkesMallowsIndex(name="FMI"),
            Informedness(name="informedness"),
            Markedness(name="markedness"),
            PositiveLikelihoodRatio(name="LR+"),
            NegativeLikelihoodRatio(name="LR-"),
            DiagnosticOddsRatio(name="DOR"),
    ])
    return model