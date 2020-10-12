"""Load and augment the images.
The code takes inspiration by https://cs230.stanford.edu/blog/datapipeline/#building-an-image-data-pipeline
"""
import numpy as np
import tensorflow as tf
import tensorflow.image as tf_image

def parse_function(filename, label):
    image_string = tf.read_file(filename)

    #Don't use tf.image.decode_imagze, or the output shape will be undefined
    image = tf_image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf_image.convert_image_dtype(image, tf.float32)

    image = tf_image.resize_images(image, [64, 64])
    return image, label


def data_augmentation(image_size):
    def data_augmentation_inner(image, label):
        image = tf_image.random_flip_left_right(image)
        image = tf_image.random_flip_up_down(image)
        image = tf_image.random_crop(image)

        image = tf_image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf_image.random_contrast(image)

        #Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label
    return data_augmentation_inner

def load_images(filenames, labels, batch_size:int=1024, image_size=(256, 256), seed:int=1337):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # shuffle and repeat before the loading of the images for speed reason
    dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True, seed=seed)
    # Load the images
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    # Augment them
    dataset = dataset.map(data_augmentation(image_size), num_parallel_calls=4)
    # Set the batch size and set the prefetch so that the CPU prepares images
    # while the GPU is working on the batch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset