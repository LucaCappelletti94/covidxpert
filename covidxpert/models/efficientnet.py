from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from .metrics import *
from typing import Tuple


def load_efficientnet_model(img_shape: Tuple[int, int, int]):
    """ 
    Build an efficientnet.

    Arguments
    ---------

    img_shape: Tuple[int,int, int],
        The shape of image. 

    """
    
    i = Input(shape=img_shape)
    
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_tensor=i
    )
    
    h = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    h = BatchNormalization()(h)
    h = Dropout(0.2)(h)
    h = Dense(128, activation="relu")(h)
    h = Dropout(0.2)(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(1, activation="sigmoid")(h)

    model= Model(i, h)

    model.compile(
        optimizer = "nadam",
        loss = "binary_crossentropy",
        metrics = [
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
        ]
    )

    return model