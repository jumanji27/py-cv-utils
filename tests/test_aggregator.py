import os
import unittest
from random import randint

from misc.aggregator import Aggregator


class TestMLQueue(unittest.TestCase):
    def setUp(self):
        self._aggregator = Aggregator(5)

    def test_init(self):
        self.assertIsInstance(self._aggregator._state_threshold, int)
        self.assertIsInstance(self._aggregator.state, list)

    def test_append_and_check(self):
        items = [
            None, 1, 1, 1, None, 1, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, 1, None, None,
            None, None, None, 1, 1, None
        ]
        for index, item in enumerate(items):
            self._aggregator.append(item)

            if index == 0:
                self.assertEqual(self._aggregator.check(), None)
                self.assertEqual(self._aggregator.state, [None])
            elif index == 3:
                self.assertEqual(self._aggregator.check(), None)
                self.assertEqual(self._aggregator.state, [None, 1, 1, 1])
            elif index == 6:
                self.assertEqual(self._aggregator.check(), None)
                self.assertEqual(self._aggregator.state, [1, 1, None, 1, None])
            elif index == 11:
                self.assertEqual(self._aggregator.check(), [1, 1, 1, 1, 1])
                self.assertEqual(self._aggregator.state, [1, 1, 1, 1, 1])
            elif index == 16:
                self.assertEqual(self._aggregator.check(), [1, 1, 1, 1, 1])
                self.assertEqual(self._aggregator.state, [1, 1, 1, 1, 1])
            elif index == 19:
                self.assertEqual(self._aggregator.check(), None)
                self.assertEqual(self._aggregator.state, [1, 1, 1, None, 1])
            elif index == 24:
                self.assertEqual(self._aggregator.check(), [None, None, None, None, None])
                self.assertEqual(self._aggregator.state, [None, None, None, None, None])
            elif index == 28:
                self.assertEqual(self._aggregator.check(), None)
                self.assertEqual(self._aggregator.state, [None, None, None, 1, 1])

    def test_reset(self):
        self._aggregator.reset()
        self.assertEqual(self._aggregator.check(), None)
        self.assertEqual(self._aggregator.state, [])

        items = [1, 1, 1, False, False, 1, False]
        for index, item in enumerate(items):
            self._aggregator.append(item)
            if index == 5:
                self._aggregator.reset()
                self.assertEqual(self._aggregator.check(), None)
                self.assertEqual(self._aggregator.state, [])


if __name__ == '__main__':
    unittest.main()
