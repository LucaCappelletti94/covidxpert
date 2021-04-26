import tensorflow as tf
from typing import Tuple

def setup_image_loader(img_shape: Tuple[int, int]):
    """Setup a parse function with the given image size.
    
    Parameters
    ----------
    img_shape: Tuple[int, int],
        The size of the resulting image, if the input image is smaller or bigger it
        will be cropped or padded with zeros.
    """
    def image_loader(filename: str, label: int):
        """Read the file and parse it as a black and white jpeg.
        
        Parameters
        ---------
        filename: str,
            The filename of the file to read
        label: int,
            The label of the file"""
        image_string = tf.io.read_file(filename)

        #Don't use tf.image.decode_imagze, or the output shape will be undefined
        image = tf.image.decode_jpeg(image_string, channels=1)

        #This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # This was present in the example but using the random_crop
        # it shouldn't be needed
        image = tf.image.resize_with_pad(image, img_shape[0], img_shape[1])
        return image, label
    return image_loader
