from .unet import unet
import numpy as np
from typing import Tuple
import cv2


class LungSegmenter:
    def __init__(self, shape: Tuple = (512, 512, 1)):
        self._model = unet(shape)
        self._shape = shape

    def predict(self, image: np.ndarray)->np.ndarray:
        thumb = cv2.resize(
            src=image,
            dsize=self._shape[:2],
            interpolation=cv2.CV_INTER_AREA
        )
        return self._model.predict([thumb.reshape(self._shape)])
