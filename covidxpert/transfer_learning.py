from typing import Generator, List, Tuple

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from .load_images import load_images

@Cache()
def train_covid_plus_pneumonia_vs_other(
    model:Model,
    x_train,
    y_train,
    x_val,
    y_val,
):
    train_data = load_images(
        x_train, y_train, 
        img_shape=img_shape, 
        crop_shape=crop_shape,
        batch_size=batch_size,
        random_state=random_state,
    )
    test_data = load_images(
        x_test, y_test, 
        img_shape=img_shape, 
        batch_size=batch_size,
        random_state=random_state,
        augment_images=False,
    )


    model.fit(
        train_data,
        epochs=1000,
        callbacks=[
            EarlyStopping(
                monitor="loss",
                patience=10,
                min_delta=0.001,
                restore_best_weights=False
            ),
            ReduceLROnPlateau(
                monitor="loss",
                patience=5,
                min_delta=0.001
            )
        ]
    )

def get_balanced_holdouts(
    dataframe: pd.DataFrame,
    model: Model,
    img_shape: Tuple[int, int],
    crop_shape: Tuple[int, int, int],
    batch_size: int,
    holdout_numbers: int = 1,
    test_size: float = 0.2,
    random_state: int = 42
) -> Model:
    """Create a generator of holdouts.

    Parameters
    ----------
    filenames: List[str],
        The list of paths of the images.
    labels: List[int],
        The label of each image.
    crop_shape: Tuple[int, int],
        The size of the input images
    img_shape: Tuple[int, int, int],
        The shape of the images, if the size differs the image will be padded 
        with zeros or cropped.
    batch_size: int,
        Batch size for the training of the model.
    holdout_numbers: int=1,
        How many holdouts will be done.
    test_size: float=0.2,
        Which fraction of the dataset will be used to test the model.
    random_state: int=42,
        The "seed" of the holdouts and data augmentation.
    """

    # load the data

    # 
    sss = StratifiedShuffleSplit(
        n_splits=holdout_numbers, test_size=test_size, random_state=random_state)

    for train_index, test_index in sss.split(filenames, labels):
        # Apply the indices to get the slices
        x_train = filenames[train_index]
        y_train = labels[train_index]
        x_test = filenames[test_index]
        y_test = labels[test_index]

        # Get the indices right

        # Build the steps\

        # train
      
        # Convert them to datasets
        train_data = load_images(
            x_train, y_train, 
            img_shape=img_shape, 
            crop_shape=crop_shape,
            batch_size=batch_size,
            random_state=random_state,
        )
        test_data = load_images(
            x_test, y_test, 
            img_shape=img_shape, 
            batch_size=batch_size,
            random_state=random_state,
            augment_images=False,
        )
        yield test_data, train_data
