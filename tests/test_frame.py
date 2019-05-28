import sys
import os
import unittest
import numpy as np
import datetime
import pickle
from time import sleep

from frame import Frame


class FrameTest(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(__file__)
        jpeg_chickens = os.path.join(dirname, 'fixtures/chickens.jpg')
        pickle_chickens = os.path.join(dirname, 'fixtures/chickens.pickle')

        with open(jpeg_chickens, 'rb') as file:
            self._chickens = file.read()
        with open(pickle_chickens, 'rb') as file:
            self._pickle_chickens = pickle.load(file)

        self._frame = Frame(self._chickens)
        self._pickle_frame = Frame(self._pickle_chickens)

    def test_init(self):
        self.assertIsInstance(self._frame._timestamp, datetime.datetime)

        self.assertIsInstance(self._pickle_frame._np_image, np.ndarray)
        self.assertIsInstance(self._pickle_frame._np_image_bgr, np.ndarray)

        self.assertEqual(
            self._pickle_frame._np_image[0][0][0],
            self._pickle_frame._np_image_bgr[0][0][2]
        )
        self.assertEqual(
            self._pickle_frame._np_image[0][0][1],
            self._pickle_frame._np_image_bgr[0][0][1]
        )
        self.assertEqual(
            self._pickle_frame._np_image[0][0][2],
            self._pickle_frame._np_image_bgr[0][0][0]
        )

        self.assertEqual(
          self._pickle_frame._np_image[len(self._pickle_frame._np_image) - 1][1919][0],
          self._pickle_frame._np_image_bgr[len(self._pickle_frame._np_image_bgr) - 1][1919][2]
        )
        self.assertEqual(
          self._pickle_frame._np_image[len(self._pickle_frame._np_image) - 1][1919][1],
          self._pickle_frame._np_image_bgr[len(self._pickle_frame._np_image_bgr) - 1][1919][1]
        )
        self.assertEqual(
          self._pickle_frame._np_image[len(self._pickle_frame._np_image) - 1][1919][2],
          self._pickle_frame._np_image_bgr[len(self._pickle_frame._np_image_bgr) - 1][1919][0]
        )

        self.assertFalse(self._frame._corrupted)
        self.assertFalse(self._pickle_frame._corrupted)

    def test_image_to_np_array(self):
        np_img_rgb, np_img_bgr = self._frame._image_to_np_array(self._chickens)

        self.assertIsInstance(np_img_rgb, np.ndarray)
        self.assertIsInstance(np_img_bgr, np.ndarray)

        self.assertEqual(np_img_rgb[0][0][0], np_img_bgr[0][0][2])
        self.assertEqual(np_img_rgb[0][0][1], np_img_bgr[0][0][1])
        self.assertEqual(np_img_rgb[0][0][2], np_img_bgr[0][0][0])

        self.assertEqual(
          np_img_rgb[len(np_img_rgb) - 1][1919][0],
          np_img_bgr[len(np_img_bgr) - 1][1919][2]
        )
        self.assertEqual(
          np_img_rgb[len(np_img_rgb) - 1][1919][1],
          np_img_bgr[len(np_img_bgr) - 1][1919][1]
        )
        self.assertEqual(
          np_img_rgb[len(np_img_rgb) - 1][1919][2],
          np_img_bgr[len(np_img_bgr) - 1][1919][0]
        )

    def test_as_binary(self):
        img_rgb = self._frame.as_binary()
        img_bgr = self._frame.as_binary(BGR=True)
        pickle_img_rgb = self._pickle_frame.as_binary()
        pickle_img_bgr = self._pickle_frame.as_binary(BGR=True)

        self.assertIsInstance(img_rgb, bytes)
        self.assertIsInstance(img_bgr, bytes)
        self.assertIsInstance(pickle_img_rgb, bytes)
        self.assertIsInstance(pickle_img_bgr, bytes)
