#!/bin/python

import os
import silence_tensorflow.auto
import tensorflow as tf
import pandas as pd
from covidxpert import main_train_loop

df = pd.read_csv("../df_all_common_cols.csv")

dataset_name = "processed"

df.img_path = [os.path.join("..", dataset_name, x) for x in df.img_path]

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    performance = main_train_loop(
        dataset_name=dataset_name,
        dataframe=df,
        img_shape=(256, 256, 1),
    )

performance.to_csv("{}.csv".format(dataset_name), index=False)
