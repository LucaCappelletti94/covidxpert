from typing import Generator, List, Tuple

from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.data import Dataset

from .load_images import load_images


def get_balanced_holdouts(
    filenames: List[str],
    labels: List[int],
    img_shape: Tuple[int, int],
    batch_size: int,
    holdout_numbers: int = 1,
    test_size: float = 0.2,
    random_state: int = 42
) -> Generator[None, Tuple[Dataset, Dataset], None]:
    """Create a generator of holdouts.

    Parameters
    ----------
    filenames: List[str],
        The list of paths of the images.
    labels: List[int],
        The label of each image.
    img_shape: Tuple[int, int],
        The shape of the images, if the size differs the image will be padded with zeros or cropped.
    batch_size: int,
        Batch size for the training of the model.
    holdout_numbers: int=1,
        How many holdouts will be done.
    test_size: float=0.2,
        Which fraction of the dataset will be used to test the model.
    random_state: int=42,
        The "seed" of the holdouts and data augmentation.
    """
    sss = StratifiedShuffleSplit(
        n_splits=holdout_numbers, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(filenames, labels):
        # Apply the indices to get the slices
        x_train = filenames[train_index]
        y_train = labels[train_index]
        x_test = filenames[test_index]
        y_test = labels[test_index]
        # Convert them to datasets
        train_data = load_images(
            x_train, y_train, image_size=img_shape, batch_size=batch_size)
        test_data = load_images(
            x_test, y_test, image_size=img_shape, batch_size=batch_size)
        yield test_data, train_data
