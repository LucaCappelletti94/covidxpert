#!/bin/python

import os
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, InceptionResNetV2
import pandas as pd
from covidxpert import main_train_loop
from tqdm.auto import tqdm

dataset_names = [
    "processed",
    "demosaicking_processed",
    "resized_processed",
]

models = {
    "ResNet50V2": dict(
        keras_model=ResNet50V2,
        batch_size=256
    ),
    "InceptionResNetV2": dict(
        keras_model=InceptionResNetV2,
        batch_size=128
    )
}

for dataset_name in tqdm(
    dataset_names,
    desc="Elaborating datasets"
):
    df = pd.read_csv("../df_all_common_cols.csv")
    df.img_path = [os.path.join("..", dataset_name, x) for x in df.img_path]
    for model_name, model_parameters in tqdm(
        models.items(),
        total=len(models),
        desc="Training models on dataset {}".format(dataset_name)
    ):
        performance = main_train_loop(
            **model_parameters,
            model_name=model_name,
            dataset_name=dataset_name,
            dataframe=df,
            img_shape=(480, 480, 1),
        )

        performance.to_csv(
            "{}_{}.csv".format(
                model_name,
                dataset_name
            ),
            index=False
        )
