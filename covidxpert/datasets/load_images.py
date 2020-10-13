"""Load and augment the images.
The code takes inspiration by https://cs230.stanford.edu/blog/datapipeline/#building-an-image-data-pipeline
"""
import numpy as np
import tensorflow as tf
import tensorflow.image as tf_image
from tensorflow.data.experimental import AUTOTUNE

def parse_function(filename: str, label: int):
    """Read the file and parse it as a black and white jpeg.
    
    Arguments
    ---------
    filename: str,
        The filename of the file to read
    label: int,
        The label of the file"""
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_imagze, or the output shape will be undefined
    image = tf_image.decode_jpeg(image_string, channels=1)

    #This will convert to float values in [0, 1]
    image = tf_image.convert_image_dtype(image, tf.float32)

    # This was present in the example but using the random_crop
    # it shouldn't be needed
    #image = tf_image.resize_images(image, [64, 64])
    return image, label


def data_augmentation(image_size, seed):
    """Prepare the data argumentation function using the parameters.
    
    Arguments
    ---------
    image_size: Tuple[int, int],
        The result size of each image (this parameter is passed to
        the random_crop function.
    seed: int,
        The random seed that each data augmentation function will use.
    """
    def data_augmentation_inner(image, label):
        """Augment the passed image.
        
        Arguments
        ---------
        image: np.array,
            The image as a matrix.
        label: int,
            The label of the image
        """
        image = tf_image.random_flip_left_right(image, seed=seed)
        image = tf_image.random_flip_up_down(image, seed=seed)
        image = tf_image.random_crop(
            image, 
            size=image_size, 
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
    return data_augmentation_inner

def load_images(filenames, labels, batch_size:int=1024, image_size=(256, 256), seed:int=1337) -> tf.data.Dataset:
    """Prepare a keras Dataset with the images and labels.
    
    Arguments
    ---------
    filenames: List[str],
        The list of paths to the images
    labels: List[int],
        The labels of each image.
    batch_size: int,
        The batchsize to use for the training.
    image_size: Tuple[int, int],
        The size of the result images, this size is passed to the
        random_crop data augmentation function.
    seed: int,
        The random seed that the data augmentation functions will use
    """
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # shuffle and repeat before the loading of the images for speed reason
    dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True, seed=seed)
    # Load the images
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    # Augment them
    dataset = dataset.map(data_augmentation(image_size, seed), num_parallel_calls=AUTOTUNE)
    # Set the batch size and set the prefetch so that the CPU prepares images
    # while the GPU is working on the batch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset