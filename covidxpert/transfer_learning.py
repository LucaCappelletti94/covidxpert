from typing import Generator, List, Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2, InceptionResNetV2  # , #EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from tqdm.auto import tqdm

from .models import load_keras_model
from .datasets import build_dataset
from cache_decorator import Cache


@Cache(
    [
        "{cache_dir}/{dataset_name}/{task_name}/{holdout_number}/{model_name}/history_{_hash}.csv",
        "{cache_dir}/{dataset_name}/{task_name}/{holdout_number}/{model_name}/model_{_hash}.keras",
        "{cache_dir}/{dataset_name}/{task_name}/{holdout_number}/{model_name}/performance_{_hash}.csv",
    ],
    args_to_ignore=(
        "model",
        "train_data",
        "test_data",
        "verbose",
        "cache_dir",
    )
)
def train(
    model: Model,
    model_name: str,
    dataset_name: str,
    task_name: str,
    holdout_number: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_shape: Tuple[int, int],
    crop_shape: Tuple[int, int] = (256, 256, 1),
    batch_size: int = 1024,
    random_state: int = 31337,
    early_stopping_patience: int = 6,
    early_stopping_min_delta: int = 0.001,
    reduce_lr_on_plateau_patience: int = 3,
    reduce_lr_on_plateau_min_delta: int = 0.001,
    max_epochs: int = 1000,
    restore_best_weights: bool = True,
    verbose: bool = True,
    cache_dir: str = "./results/"
) -> Tuple[pd.DataFrame, Model, pd.DataFrame]:
    """Train the model and returns the history, model and performance csv.

    Arguments
    ---------
    model: Model,
        The keras model to train.
    dataset_name: str,
        The name of the dataset we are using, this is just used for the cache.
    task_name: str,
        The name of the current task, this is just used for the cache.
    holdout_number: int,
        Which holdout we are cucrently training, this is just used for the cache.
    train_data: pd.DataFrame,
        A dataframe to pass to the fit method of the model.
    test_data: pd.DataFrame,
        A dataframe to pass to the fit method of the model.
    early_stopping_patience: int = 6,
        How many epochs the early stopping will wait for the model to improve 
        before stopping.
    early_stopping_min_delta: float = 0.001,
        The minimum improvement the model will need to not be stopped.
    early_stopping_patience: int = 6,
        How many epochs the readuce lr on plateau will wait for the model to improve 
        before reducing the learning rate.
    early_stopping_min_delta: float = 0.001,
        The minimum improvement the model will need to not reduce the learning rate.
    max_epochs: int = 1000,
        Max number of epochs the modell will train for.
    restore_best_weight: bool = True,
        Whether or not to restore at the end the best weights in the training.
    verbose: bool = True,
        If the training will be verbose or not.
    cache_dir: str = "./results/",
        The directory to use for the cache.
    """
    # Convert them to datasets
    train_data = build_dataset(
        train_df.img_path, train_df.label,
        img_shape=img_shape,
        crop_shape=crop_shape,
        batch_size=batch_size,
        random_state=random_state,
    )
    test_data = build_dataset(
        test_df.img_path, test_df.label,
        img_shape=img_shape,
        batch_size=batch_size,
        random_state=random_state,
        augment_images=False,
    )

    history = pd.DataFrame(model.fit(
        train_data,
        validation_data=test_data,
        epochs=max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="loss",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=restore_best_weights,
            ),
            ReduceLROnPlateau(
                monitor="loss",
                patience=reduce_lr_on_plateau_patience,
                min_delta=reduce_lr_on_plateau_min_delta,
            )
        ]
    ).history)

    perf = pd.DataFrame(
        [{
            **dict(zip(
                model.metrics_names,
                model.evaluate(
                    train_data,
                    verbose=verbose
                )
            )),
            "run_type": "training"
        },
            {
            **dict(zip(
                model.metrics_names,
                model.evaluate(
                    test_data,
                    verbose=verbose
                )
            )),
            "run_type": "test"
        }]
    )

    return history, model, perf


def get_task_dataframes(dataframe: pd.DataFrame):
    """Returns a generator with the dataframe to use for each task.

    We choose 3 tasks to use for transfer-learning:
    - Covid OR Pneumonia Vs others
    - Covid Vs Pneumnonia
    - Covid Vs others

    We hope that doing so we will improve the performance of the model since
    the data are un-balanced.
    """
    result = []

    covid_or_pneumonia_vs_other = pd.DataFrame({
        "img_path": dataframe.img_path,
        "label": dataframe.covid19 | dataframe.pneumonia,
    })
    result.append(("covid_or_pneumonia_vs_other", covid_or_pneumonia_vs_other))

    only_covid_or_pneumonia = dataframe[covid_or_pneumonia_vs_other.label.astype(
        bool)]
    covid_vs_pneumonia = pd.DataFrame({
        "img_path": only_covid_or_pneumonia.img_path,
        "label": only_covid_or_pneumonia.covid19,
    })
    result.append(("covid_vs_pneumonia", covid_vs_pneumonia))

    covid_vs_other = pd.DataFrame({
        "img_path": dataframe.img_path,
        "label": dataframe.covid19,
    })
    result.append(("covid_vs_other", covid_vs_other))

    return result


def get_balanced_holdouts(
    dataframe: pd.DataFrame,
    holdout_numbers: int = 1,
    test_size: float = 0.2,
    random_state: int = 42
) -> Generator:
    """Create a generator of holdouts.

    Parameters
    ----------
    dataframe: pd.DataFrame,
        This is the input dataframe, it should have two columns: 
        - image_path: which is the path to the image to load
        - label: the **binary** label of the image.
    img_shape: Tuple[int, int],
        The shape of the images, if the size differs the image will be padded 
        with zeros or cropped.
    crop_shape: Tuple[int, int, int],
        The size of the input images
    batch_size: int,
        Batch size for the training of the model.
    holdout_numbers: int = 1,
        How many holdouts will be done.
    test_size: float = 0.2,
        Which fraction of the dataset will be used to test the model.
    random_state: int = 42,
        The "seed" of the holdouts and data augmentation.
    """
    classes = dataframe[["normal", "covid19",
                         "other", "pneumonia"]].values.argmax(axis=1)

    # use a stratified split to get a train and test split which should have
    # reduced covariate shift
    sss = StratifiedShuffleSplit(
        n_splits=holdout_numbers, test_size=test_size, random_state=random_state)

    for holdout_number, (train_index, test_index) in tqdm(
        enumerate(sss.split(dataframe, classes)),
        desc="Holdout",
        leave=False,
        total=holdout_numbers,
    ):
        # Apply the indices to get the slices
        train_df = dataframe.iloc[train_index]
        test_df = dataframe.iloc[test_index]

        yield holdout_number, train_df, test_df


def get_models_generator(img_shape: Tuple[int, int]):
    return tqdm((
        load_keras_model(model, img_shape)
        for model in [
                ResNet50V2,
                InceptionResNetV2,
                # EfficientNetB4,
                ]
    ),
        desc="Models",
        leave=False,
        total=2,
    )


def main_train_loop(
    dataset_name: str,
    dataframe: pd.DataFrame,
    img_shape: Tuple[int, int],
    crop_shape: Tuple[int, int] = (256, 256, 1),
    nadam_kwargs=None,
    holdout_numbers: int = 10,
    batch_size: int = 256,
    early_stopping_patience: int = 6,
    early_stopping_min_delta: int = 0.001,
    reduce_lr_on_plateau_patience: int = 3,
    reduce_lr_on_plateau_min_delta: int = 0.001,
    max_epochs: int = 1000,
    random_state: int = 31337,
    restore_best_weights: bool = True,
    verbose=True,
    cache_dir="./results/"
):
    """Run the training loop for all the models and tasks on the given dataset.

    Arguments
    ---------
    img_shape: Tuple[int, int],
        The shape of the image.
    nadam_kwargs: dict,
        The keywords aaguments to be passed to the Nadam Optimizer.
    early_stopping_patience: int = 6,
        How many epochs the early stopping will wait for the model to improve 
        before stopping.
    early_stopping_min_delta: float = 0.001,
        The minimum improvement the model will need to not be stopped.
    early_stopping_patience: int = 6,
        How many epochs the readuce lr on plateau will wait for the model to improve 
        before reducing the learning rate.
    early_stopping_min_delta: float = 0.001,
        The minimum improvement the model will need to not reduce the learning rate.
    max_epochs: int = 1000,
        Max number of epochs the modell will train for.
    restore_best_weight: bool = True,
        Whether or not to restore at the end the best weights in the training.
    verbose: bool = True,
        If the training will be verbose or not.
    cache_dir: str = "./results/",
        The directory to use for the cache.
    """
    total_perf = []
    for holdout_number, train_df, test_df in get_balanced_holdouts(dataframe, holdout_numbers):
        for model in get_models_generator(img_shape):
            for (task_name, task_train_df), (_task_name, task_test_df) in tqdm(zip(
                    get_task_dataframes(train_df),
                    get_task_dataframes(test_df),
                ),
                desc="Task",
                total=3,
                leave=False,
            ):
                _history, model, perf = train(
                    model=model,
                    model_name=model.name,
                    dataset_name=dataset_name,
                    task_name=task_name,
                    holdout_number=holdout_number,
                    train_df=task_train_df,
                    test_df=task_test_df,
                    img_shape=img_shape,
                    crop_shape=crop_shape,
                    batch_size=batch_size,
                    random_state=random_state,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
                    reduce_lr_on_plateau_min_delta=reduce_lr_on_plateau_min_delta,
                    max_epochs=max_epochs,
                    restore_best_weights=restore_best_weights,
                    verbose=verbose,
                    cache_dir=cache_dir,
                )
                total_perf.append(perf)

    return pd.concat(total_perf)
