"""Load and augment the images.
The code takes inspiration by the following tutorial from Andreww Ng:
https://cs230.stanford.edu/blog/datapipeline/#building-an-image-data-pipeline
"""
import tensorflow as tf
from typing import Tuple, List

from .image_loader import setup_image_loader

def build_dataset(
        filenames:List[str], 
        labels:List[int], 
        batch_size:int=1024,  
        img_shape:Tuple[int, int]=(480, 480), 
        random_state:int=1337,
    ) -> tf.data.Dataset:
    """Prepare a keras Dataset with the images and labels.
    
    Parameters
    ---------
    filenames: List[str],
        The list of paths to the images
    labels: List[int],
        The labels of each image.
    batch_size: int,
        The batchsize to use for the training.
    img_shape: Tuple[int, int],
        The size of the input images
    crop_shape: Tuple[int, int, int],
        The size of the result images, this size is passed to the
        random_crop data augmentation function.
    random_state: int,
        The random state that the data augmentation functions will use
    augment_images:bool,
        If the data should be augmented or not.
    """
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # shuffle and repeat before the loading of the images for speed reason
    dataset = dataset.shuffle(
        len(filenames), 
        reshuffle_each_iteration=True,
        seed=random_state
    )

    # Load the images
    dataset = dataset.map(
        setup_image_loader(img_shape), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Set the batch size and set the prefetch so that the CPU prepares images
    # while the GPU is working on the batch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
