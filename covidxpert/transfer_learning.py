from typing import *
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import AUC
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

from itertools import permutations

@Cache(
    [
        "{cache_dir}/{dataset_name}/{task_name}/{holdout_number}/{model_name}/history_{_hash}.csv",
        "{cache_dir}/{dataset_name}/{task_name}/{holdout_number}/{model_name}/model_{_hash}.keras.tar.gz",
        "{cache_dir}/{dataset_name}/{task_name}/{holdout_number}/{model_name}/performance_{_hash}.csv",
    ],
    args_to_ignore=(
        "model",
        "train_df",
        "val_df",
        "val_df",
        "test_df",
        "verbose",
        "cache_dir",
        "verbose",
    )
)
def train(
    model: Model,
    model_name: str,
    dataset_name: str,
    task_name: str,
    holdout_number: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_shape: Tuple[int, int],
    batch_size: int = 1024,
    random_state: int = 31337,
    early_stopping_patience: int = 6,
    early_stopping_min_delta: int = 0.001,
    reduce_lr_on_plateau_patience: int = 3,
    reduce_lr_on_plateau_min_delta: int = 0.001,
    max_epochs: int = 1000,
    restore_best_weights: bool = True,
    verbose: bool = True,
    cache_dir: str = "./results/",
    nadam_kwargs=None
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
    nadam_kwargs: dict,
        The keywords aaguments to be passed to the Nadam Optimizer.
    """
    # Convert them to datasets
    train_data = build_dataset(
        train_df.img_path, train_df.label,
        img_shape=img_shape,
        batch_size=batch_size,
        random_state=random_state,
    )
    val_data = build_dataset(
        val_df.img_path, val_df.label,
        img_shape=img_shape,
        batch_size=batch_size,
        random_state=random_state,
    )
    test_data = build_dataset(
        test_df.img_path, test_df.label,
        img_shape=img_shape,
        batch_size=batch_size,
        random_state=random_state,
    )

    # Use an empty dict as default avoiding the quirks of having a mutable default.
    nadam_kwargs = nadam_kwargs or {}

    model.compile(
        optimizer=Nadam(**nadam_kwargs),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="PR", name="AUPRC"),
            AUC(curve="ROC", name="AUROC"),
        ]
    )

    history = pd.DataFrame(model.fit(
        train_data,
        validation_data=val_data,
        epochs=max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=restore_best_weights,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
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
                    val_data,
                    verbose=verbose
                )
            )),
            "run_type": "validation"
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


def get_holdouts(df: pd.DataFrame) -> Generator:
    """
    Create a generator of holdouts.

    Parameters
    ----------
    df: pd.DataFrame,
        This is the input dataframe, it should have this columns: 
        - image_path: which is the path to the image to load
        - normal, covid19, pneuomina, other: these columns contain the **binary** label of the image.
    """
    filter_cols = ['img_path', 'normal', 'covid19', 'pneumonia', 'other']
    datasets_to_rotate = ['COVID-19 Radiography Database', 'covid-chestxray-dataset', 'Actualmed-COVID-chestxray-dataset']
    dataset_to_remove = 'all_masks'
    df = df[(df.dataset!=dataset_to_remove)]

    map_exp = {f'holdout_{k}': {'train': val[0][0], 'validation': val[0][1], 'test': val[0][2]}
                for k, val in enumerate(zip(permutations(datasets_to_rotate, r=len(datasets_to_rotate))))
              }
    for k, v in map_exp.items():
        df[k] = df.dataset.apply(lambda x: 'validation' 
                                         if x == map_exp[k]['validation'] 
                                         else ('test' if x == map_exp[k]['test']  else 'train'))
    
    for holdout_number in range(len(map_exp)):
        grp_holdout = df.groupby(f'holdout_{holdout_number}')
        train_df = grp_holdout.get_group('train')[filter_cols]
        val_df = grp_holdout.get_group('validation')[filter_cols]
        test_df = grp_holdout.get_group('test')[filter_cols]
        
        yield holdout_number, train_df, val_df, test_df

def get_balanced_holdouts(
    dataframe: pd.DataFrame,
    holdout_numbers: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.05,
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
    val_size: float = 0.05,
        Which fraction of the training data will be used for the validation of the model.
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

        sub_train_idx, sub_val_idx = next(StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=random_state
        ).split(train_df, classes[train_index]))

        val_df = train_df.iloc[sub_val_idx]
        train_df = train_df.iloc[sub_train_idx]

        yield holdout_number, train_df, val_df, test_df


def main_train_loop(
    keras_model: [[None], Model],
    model_name: str,
    dataset_name: str,
    dataframe: pd.DataFrame,
    img_shape: Tuple[int, int],
    nadam_kwargs=None,
    holdout_numbers: int = 10,
    batch_size: int = 256,
    early_stopping_patience: int = 16,
    early_stopping_min_delta: int = 0.001,
    reduce_lr_on_plateau_patience: int = 2,
    reduce_lr_on_plateau_min_delta: int = 0.001,
    max_epochs: int = 1000,
    random_state: int = 31337,
    restore_best_weights: bool = True,
    verbose: bool = True,
    cache_dir: str = "./results/"
):
    """Run the training loop for all the models and tasks on the given dataset.

    Arguments
    ---------
    img_shape: Tuple[int, int],
        The shape of the image.
    nadam_kwargs: dict,
        The keywords arguments to be passed to the Nadam Optimizer.
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
    for holdout_number, train_df, val_df, test_df in get_holdouts(dataframe):
        total_perf.append(run_holdout(
            keras_model=keras_model,
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            dataset_name=dataset_name,
            holdout_number=holdout_number,
            img_shape=img_shape,
            batch_size=batch_size,
            random_state=random_state,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
            reduce_lr_on_plateau_min_delta=reduce_lr_on_plateau_min_delta,
            max_epochs=max_epochs,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
            cache_dir=cache_dir,
        ))

    return pd.concat(total_perf)


@Cache(
    "{cache_dir}/{dataset_name}/{holdout_number}/{model_name}/perf_{_hash}.csv",
    args_to_ignore=(
        "keras_model",
        "train_df",
        "val_df",
        "test_df",
        "cache_dir",
        "verbose",
    )
)
def run_holdout(
    keras_model: Callable[[None], Model],
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    holdout_number: int,
    img_shape: Tuple[int, int],
    batch_size: int = 256,
    random_state: int = 31337,
    early_stopping_patience: int = 4,
    early_stopping_min_delta: int = 0.001,
    reduce_lr_on_plateau_patience: int = 2,
    reduce_lr_on_plateau_min_delta: int = 0.001,
    max_epochs: int = 1000,
    restore_best_weights: bool = True,
    verbose: bool = True,
    cache_dir: str = "./results/"
) -> pd.DataFrame:
    """
    Arguments
    ---------
    img_shape: Tuple[int, int],
        The shape of the image.
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
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = load_keras_model(keras_model, img_shape)
        for (task_name, task_train_df), (_, task_val_df), (_, task_test_df) in tqdm(zip(
                get_task_dataframes(train_df),
                get_task_dataframes(val_df),
                get_task_dataframes(test_df),
            ),
            desc="Task",
            total=3,
            leave=False,
        ):
            _history, model, perf = train(
                model=model,
                model_name=model_name,
                dataset_name=dataset_name,
                task_name=task_name,
                holdout_number=holdout_number,
                train_df=task_train_df,
                val_df=task_val_df,
                test_df=task_test_df,
                img_shape=img_shape,
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
            perf["holdout_number"] = holdout_number
            perf["task_name"] = task_name
            total_perf.append(perf)
    return pd.concat(total_perf)