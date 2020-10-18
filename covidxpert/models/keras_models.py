from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC

from .metrics import *

def load_keras_model(keras_model, img_shape):
    """Adapt a keras model for our task making them accept a 
        gray-scale image and adding a basic mlp on the top of it.
    
    Arguments
    ---------
    keras_model,
        One of the model specified at https://keras.io/api/applications/
        They can be found under tf.keras.applications.
    img_shape: Tuple[int, int],
        The shape of the image.
    """
    i = Input(shape=img_shape)
    h = Conv2D(3, kernel_size=(1, 1))(i)

    kmodel = keras_model(
        input_tensor=h,
        include_top=False,
        weights=None,
        classes=2,
    )

    o = Flatten()(kmodel.output)
    o = Dense(1, activation="sigmoid")(o)

    model = Model(i, o)

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="PR", name="AUPRC"),
            AUC(curve="ROC", name="AUROC"),
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
            Accuracy(name="accuracy_2"),
            BalancedAccuracy(name="BA"),
            F1Score(name="f1-score"),
            MatthewsCorrelationCoefficinet(name="MCC"),
            FowlkesMallowsIndex(name="FMI"),
            Informedness(name="informedness"),
            Markedness(name="markedness"),
    ])
    return model