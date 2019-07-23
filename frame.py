import cv2
import numpy as np
import time
import datetime


class Frame:
    """
    Frame is class for convenient work with bin/numpy images in RGB/BGR format and convertion either
    :param image: bin or numpyarray image
    """
    def __init__(self, image, BGR=False, single_channel=False):
        self._timestamp = time.time()
        if isinstance(image, bytes):
            try:
                self._np_image, self._np_image_bgr = self._image_to_np_array(image, BGR)
                self._corrupted = False
            except Exception as err:
                self._corrupted = True
        else:
            if BGR:
                self._np_image = image[:, :, ::-1]
                self._np_image_bgr = image
            else:
                if single_channel:
                    self._np_image_bgr = image
                else:
                    self._np_image_bgr = image[:, :, ::-1]
                self._np_image = image
            self._corrupted = False

    def _image_to_np_array(self, image, BGR=False):
        img_rgb = img_bgr = None
        if BGR:
            img_rgb = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            img_bgr = img_rgb[:, :, ::-1]
        else:
            img_bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = img_bgr[:, :, ::-1]
        return img_rgb, img_bgr

    def get_time(self, iso=False):
        if iso:
            return datetime.datetime.utcfromtimestamp(self._timestamp).isoformat() + 'Z'
        return self._timestamp

    def is_corrupted(self):
        return self._corrupted

    def as_np_array(self, BGR=False, gray=False):
        np_image = self._np_image_bgr if BGR else self._np_image
        if gray:
            return cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        else:
            return np_image

    def as_binary(self, BGR=False):
        np_image_to_encode = self._np_image if BGR else self._np_image_bgr
        return cv2.imencode('.jpg', np_image_to_encode)[1].tobytes()

    def copy(self):
        return Frame(self._np_image.copy())
