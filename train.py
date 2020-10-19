#!/bin/python
BATCH_SIZE = 16
IMG_SHAPE=(460, 460)
CROP_SHAPE=(440, 440, 1)
MODELS_DIR = "./weights"
BASE_DIR = "/home/lucacappelletti/processed/"

import os
from IPython import embed
import numpy as np
import pandas as pd
from datetime import datetime
from covidxpert import get_balanced_holdouts
from notipy_me import KerasNotipy
from covidxpert import load_keras_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TerminateOnNaN


# Load the data
df = pd.read_csv("../df_all_common_cols.csv")
# temporary workaround
df.img_path = [".".join(x.split(".")[:-1]) + ".jpg" for x in df.img_path]

# Extract the labels and filenames
filenames = np.array([
    os.path.join(BASE_DIR, x)

    for x in df.img_path.values
])
labels = np.array(df.covid19.values | df.pneumonia.values)

# Compute the re-balancing weights
positives_ratio = np.mean(labels)
class_weights = {
    0:positives_ratio,
    1:1 - positives_ratio
}

# Load the images
test, train = next(get_balanced_holdouts(filenames, labels, IMG_SHAPE, CROP_SHAPE, BATCH_SIZE))

# Load the model
from tensorflow.keras.applications import InceptionResNetV2
model = load_keras_model(InceptionResNetV2, CROP_SHAPE)
model.summary()

ROOT_FOLDER = os.path.join(MODELS_DIR, "InceptionResNetV2")

try:
    # Train the model
    history = model.fit(
        train,
        validation_data=test,
        epochs=1000,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(
                monitor="loss",
                patience=6,
                min_delta=0.001,
                restore_best_weights=False
            ),
            ReduceLROnPlateau(
                monitor="loss",
                patience=3,
                min_delta=0.001
            ),
            ModelCheckpoint(
                os.path.join(ROOT_FOLDER, "model_checkpoint"), 
                monitor='loss', 
                save_freq='epoch',
                save_weights_only=True
            ),
            KerasNotipy("Training Covidxpert Resnet"),
            CSVLogger(os.path.join(ROOT_FOLDER, "model_log.csv")),
            TerminateOnNaN()
        ]
    ).history

    model.save_weights(os.path.join(ROOT_FOLDER, "best_weights_{}.h5".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))
    history.to_csv(os.path.join(ROOT_FOLDER, "model_history.csv"))
except Exception as e:
    print(e)

# leave the shell open to save manually in case of errors
from IPython import embed
embed()
