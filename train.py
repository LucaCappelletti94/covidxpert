#!/bin/python

import os
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, ResNet50V2
import pandas as pd
from covidxpert import main_train_loop

df = pd.read_csv("../df_all_common_cols.csv")

dataset_name = "processed"

df.img_path = [os.path.join("..", dataset_name, x) for x in df.img_path]

performance = main_train_loop(
    keras_model=ResNet50V2,
    model_name="ResNet50V2",
    batch_size=256, # We run InceptionNet with batchsize 128 and for the ResNet50V2 we use 256
    dataset_name=dataset_name,
    dataframe=df,
    img_shape=(480, 480, 1),
)

performance.to_csv("{}.csv".format(dataset_name), index=False)
