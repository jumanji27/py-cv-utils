import os
import unittest
import queue
import logging
import sys

from ml_queue import MLQueue


class TestMLQueue(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

        self._small_ml_queue = MLQueue(
            'small_queue_id',
            {
                'max_size': 50,
                'max_batch_size': 20
            }
        )
        self._big_ml_queue = MLQueue(
            'big_queue_id',
            {
                'max_size': 7000,
                'max_batch_size': 250
            }
        )

    def test_init(self):
        self.assertIsInstance(self._small_ml_queue._queue, queue.Queue)
        self.assertEqual(self._small_ml_queue._items_in_percent, 0.5)

    def test_small_queue(self):
        counter = 0
        while counter < 35:
            self._small_ml_queue.put({'test': 'test', 'test_int': 123})
            counter += 1

        self.assertEqual(
            len(self._small_ml_queue),
            35
        )
        self.assertEqual(self._small_ml_queue.get_batch_size(), 19)
        self.assertEqual(self._small_ml_queue._dilution, 14)

        self.assertEqual(self._small_ml_queue._prev_batch_size_load, 96)
        self.assertEqual(self._small_ml_queue._prev_dilution_load, 36)

        dilution_results = self._small_ml_queue.get()
        self.assertEqual(
            len(self._small_ml_queue),
            21
        )
        self.assertIsInstance(dilution_results, list)
        self.assertEqual(
            len(dilution_results), 1
        )
        self.assertEqual(self._small_ml_queue.get_batch_size(), 17)
        self.assertEqual(self._small_ml_queue._dilution, 1)

        batch_size_results = self._small_ml_queue.get()
        self.assertEqual(
            len(self._small_ml_queue),
            4
        )
        self.assertIsInstance(batch_size_results, list)
        self.assertEqual(
            len(batch_size_results), 17
        )
        self.assertEqual(self._small_ml_queue.get_batch_size(), 3)
        self.assertEqual(self._small_ml_queue._dilution, 1)

        single_result = self._small_ml_queue.get(auto_batch_size=False)
        self.assertEqual(
            len(self._small_ml_queue),
            3
        )
        self.assertIsInstance(single_result, list)
        self.assertEqual(
            len(single_result), 1
        )
        self.assertEqual(self._small_ml_queue.get_batch_size(), 3)
        self.assertEqual(self._small_ml_queue._dilution, 1)

        size_result = self._small_ml_queue.get(size=2, auto_batch_size=False)
        self.assertEqual(
            len(self._small_ml_queue),
            1
        )
        self.assertIsInstance(size_result, list)
        self.assertIsInstance(size_result[0], list)
        self.assertIsInstance(size_result[1], list)
        self.assertEqual(
            len(size_result), 2
        )
        self.assertEqual(self._small_ml_queue.get_batch_size(), 1)
        self.assertEqual(self._small_ml_queue._dilution, 1)

    def test_big_queue(self):
        counter = 0
        while counter < 2000:
            self._big_ml_queue.put([
                {'test': 'test', 'test_int': 123}, {'test': 'test', 'test_int': 123}
            ])
            counter += 1

        self.assertEqual(
            len(self._big_ml_queue),
            4000
        )
        self.assertEqual(self._big_ml_queue.get_batch_size(), 250)
        self.assertEqual(self._big_ml_queue._dilution, 1050)

        self.assertEqual(self._big_ml_queue._prev_batch_size_load, 100)
        self.assertEqual(self._big_ml_queue._prev_dilution_load, 10)

        dilution_results = self._big_ml_queue.get()
        self.assertEqual(
            len(self._big_ml_queue),
            2950
        )
        self.assertIsInstance(dilution_results, list)
        self.assertEqual(
            len(dilution_results), 1
        )
        self.assertEqual(self._big_ml_queue.get_batch_size(), 210)
        self.assertEqual(self._big_ml_queue._dilution, 1)

        batch_size_results = self._big_ml_queue.get()
        self.assertEqual(
            len(self._big_ml_queue),
            2740
        )
        self.assertIsInstance(batch_size_results, list)
        self.assertEqual(
            len(batch_size_results), 210
        )
        self.assertEqual(self._big_ml_queue.get_batch_size(), 210)
        self.assertEqual(self._big_ml_queue._dilution, 1)

        single_result = self._big_ml_queue.get(auto_batch_size=False)
        self.assertEqual(
            len(self._big_ml_queue),
            2739
        )
        self.assertIsInstance(single_result, list)
        self.assertEqual(
            len(single_result), 1
        )
        self.assertEqual(self._big_ml_queue.get_batch_size(), 210)
        self.assertEqual(self._big_ml_queue._dilution, 1)

        size_result = self._big_ml_queue.get(size=2, auto_batch_size=False)
        self.assertEqual(
            len(self._big_ml_queue),
            2737
        )
        self.assertIsInstance(size_result, list)
        self.assertEqual(
            len(size_result), 2
        )
        self.assertEqual(self._big_ml_queue.get_batch_size(), 210)
        self.assertEqual(self._big_ml_queue._dilution, 1)

        size_result_with_batch_size = self._big_ml_queue.get(size=2)
        self.assertEqual(
            len(self._big_ml_queue),
            2317
        )
        self.assertIsInstance(size_result_with_batch_size, list)
        self.assertIsInstance(size_result_with_batch_size[0], list)
        self.assertIsInstance(size_result_with_batch_size[1], list)
        self.assertEqual(
            len(size_result_with_batch_size), 2
        )
        self.assertEqual(
            len(size_result_with_batch_size[0]), 210
        )
        self.assertEqual(
            len(size_result_with_batch_size[1]), 210
        )
        self.assertEqual(self._big_ml_queue.get_batch_size(), 165)
        self.assertEqual(self._big_ml_queue._dilution, 1)

    def test_fully_loaded(self):
        self._fully_loaded_ml_queue = MLQueue(
            'fully_loaded_ml_queue_id',
            {
                'max_size': 100,
                'max_batch_size': 1
            }
        )

        counter = 0
        while counter < 50:
            self._fully_loaded_ml_queue.put([
                {'test': 'test', 'test_int': 123}, {'test': 'test', 'test_int': 123}
            ])
            counter += 1

        self.assertEqual(
            len(self._fully_loaded_ml_queue),
            0
        )
        self.assertEqual(
            self._fully_loaded_ml_queue.get(),
            []
        )

        counter = 0
        while counter < 100:
            self._fully_loaded_ml_queue.put([
                {'test': 'test', 'test_int': 123}, {'test': 'test', 'test_int': 123}
            ])
            counter += 1

        self.assertEqual(
            len(self._fully_loaded_ml_queue),
            0
        )
        self.assertEqual(
            self._fully_loaded_ml_queue.get(size=10),
            []
        )

    def tearDown(self):
        logging.disable(logging.NOTSET)


if __name__ == '__main__':
    unittest.main()
