import queue
import logging

logger = logging.getLogger('ml-queue')


# Queue is a way to communicate with ML
class MLQueue:
    def __init__(self, queue_id, config):
        self._id = queue_id
        self._max_size = config['max_size']
        self._queue = queue.LifoQueue(self._max_size)

        self._balancer_threshold = config.get('balancer_threshold', 50)
        self._balancer_step = config.get('balancer_step', 5)
        self._balancer_step_log = config.get('balancer_step_log', 10)
        self._batch_size_float = 0.0
        self._max_batch_size = config['max_batch_size']
        self._batch_size_step = self._max_batch_size / 100
        self._prev_batch_size_load = 0
        self._dilution = 1
        self._dilution_multiplier = config.get('balancer_dilution_backward_purge', 10)
        self._items_in_percent = self._max_size / 100
        self._prev_dilution_load = 0

    def _calculate_batch_size(self, load, diff):
        diff_in_steps = diff * self._batch_size_step
        self._batch_size_float += diff_in_steps
        if self._batch_size_float < 1:  # If diff is too small or negative
            self._batch_size_float = 1.0
        if self._batch_size_float > self._max_batch_size:
            self._batch_size_float -= diff_in_steps

        status = None
        if diff > 0:
            status = 'increased'
        else:
            status = 'decreased'

        if load and not load % self._balancer_step_log:
            logger.info(
                f'{self._id} ML queue is loaded by {load}%, ' +
                f'batch_size is {status} ({self.get_batch_size()} now)'
            )

    def _calculate_dilution(self, load, diff):
        percents_to_clear = load - self._balancer_threshold + self._dilution_multiplier
        if percents_to_clear >= self._balancer_step_log:
            self._dilution = round(self._items_in_percent * percents_to_clear)
        else:
            self._dilution = 1

        status = None
        if diff > 0:
            status = 'increased'
        else:
            status = 'decreased'

        if load and not load % self._balancer_step_log:
            logger.info(
                f'{self._id} ML queue is loaded by {load}%, ' +
                f'dilution is {status} ({self._dilution} now)'
            )

    def _balance(self):
        size = len(self)
        load = round(size / self._max_size * 100)
        batch_size_load = None
        batch_size_threshold_ratio = 100 / self._balancer_threshold
        dilution_load = None
        dilution_threshold_ratio = 100 / (100 - self._balancer_threshold)
        if load < self._balancer_threshold:
            batch_size_load = round(load * batch_size_threshold_ratio)
            dilution_load = 0
        else:
            batch_size_load = 100
            dilution_load = round((load - self._balancer_threshold) * dilution_threshold_ratio)

        batch_size_load_diff = batch_size_load - self._prev_batch_size_load
        batch_size_load_diff_abs = abs(batch_size_load_diff)
        dilution_load_diff = dilution_load - self._prev_dilution_load
        dilution_load_diff_abs = abs(dilution_load_diff)

        if batch_size_load_diff_abs / batch_size_threshold_ratio >= self._balancer_step:
            self._calculate_batch_size(load, batch_size_load_diff)
            self._prev_batch_size_load = batch_size_load

        if dilution_load_diff_abs / dilution_threshold_ratio >= self._balancer_step:
            self._calculate_dilution(load, dilution_load_diff)
            self._prev_dilution_load = dilution_load

        if not len(self):
            self._batch_size_float = 0.0
            self._prev_batch_size_load = 0
            self._dilution = 1
            self._prev_dilution_load = 0

        if self._queue.full():
            self._queue.queue.clear()
            logger.info(f'{self._id} ML queue was fully loaded and cleared')

    def put(self, items):
        if self._queue.full():
            logger.info(f'{self._id} ML queue is fully loaded, your put is skipped')
        else:
            if isinstance(items, list):
                for item in items:
                    self._queue.put(item)
            else:
                self._queue.put(items)
            self._balance()

    def _dilute(self, batch):
        # It's necessary to gather all queue items in array
        # otherwise next put will be earlier than last loop iteration (strange Python behavior)
        # By memory consuption it's the same as solution without aggregation (one by one),
        # tested
        if self._dilution > 1:
            results = []
            for index in range(0, self._dilution):
                results.append(self._queue.get())
            return results[-1:]
        elif batch:
            results = []
            batch_size = self.get_batch_size()
            if batch_size:
                for index in range(0, batch_size):
                    results.append(self._queue.get())
            else:
                results.append(self._queue.get())
            return results
        else:
            return [self._queue.get()]

    def get(self, size=1, auto_batch_size=True, wait=False):
        result = []
        if len(self) or wait:
            if size > 1:
                items = []
                for counter in range(0, size):
                    items.append(
                        self._dilute(auto_batch_size)
                    )
                result = items
            else:
                result = self._dilute(auto_batch_size)
            self._balance()
        return result

    def get_batch_size(self):
        return round(self._batch_size_float)

    def __len__(self):
        return self._queue.qsize()
