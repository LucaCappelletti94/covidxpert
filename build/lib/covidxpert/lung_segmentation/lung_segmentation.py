from .unet import unet
import numpy as np
from typing import Tuple
import cv2


class LungSegmenter:
    def __init__(self, shape: Tuple = (512, 512, 1)):
        self._model = unet(shape)
        self._shape = shape

    def predict(self, image: np.ndarray)->np.ndarray:
        thumb = cv2.resize(image, self._shape[:2])
        thumb = np.float32(thumb)
        thumb = (thumb - thumb.min()) / (thumb.max() - thumb.min())
        return self._model.predict([thumb.reshape(1, *self._shape)]).reshape(self._shape[:2])
