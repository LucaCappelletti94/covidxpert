import numpy as np
import tensorflow as tf
import tensorflow.image as tf_image
from typing import Tuple


def setup_data_augmentation(crop_shape:Tuple[int, int, int], random_state:int):
    """Prepare the data argumentation function using the parameters.
    
    Parameters
    ---------
    crop_shape: Tuple[int, int, int],
        The result size of each image (this parameter is passed to
        the random_crop function.
    random_state: int,
        The random seed that each data augmentation function will use.
    """
    def data_augmentation(image: np.array, label: int):
        """Augment the passed image.
        
        Parameters
        ---------
        image: np.array,
            The image as a matrix.
        label: int,
            The label of the image
        """
        image = tf_image.random_flip_left_right(image, seed=random_state)
        #image = tf_image.random_flip_up_down(image, seed=seed)
        image = tf_image.random_crop(
            image, 
            size=crop_shape, 
            seed=seed, 
            name="random_contrast"
        )

        image = tf_image.random_brightness(
            image, 
            max_delta=32.0 / 255.0, 
            seed=seed
        )
        image = tf_image.random_contrast(
            image,
            lower=0.3,
            higher=0.7,
            seed=seed
        )

        #Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label
    return data_augmentation
