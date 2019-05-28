import time
import logging

logger = logging.getLogger('aggregator')


# Aggregate events from ML
class Aggregator:
    def __init__(self, state_threshold, time_gap=None, aggregator_type='frame_counter'):
        self._type = aggregator_type
        self._state_threshold = state_threshold
        if self._type == 'time_gap':
            self._time_gap = time_gap
        self.state = []

    def append(self, item):
        if len(self) == self._state_threshold:
            self.state.pop(0)
        if self._type == 'time_gap':
            item['_timestamp'] = time.time()
        self.state.append(item)

    def _frame_counter_check(self, field_to_check, limit):
        has_distinction = False
        batch = self.state[-limit:]
        for index, item in enumerate(batch):
            if field_to_check:
                current_item_type = type(item[field_to_check])
            else:
                current_item_type = type(item)
            prev_item_type = None
            if index:
                if field_to_check:
                    prev_item_type = type(batch[index - 1][field_to_check])
                else:
                    prev_item_type = type(batch[index - 1])
            else:
                prev_item_type = current_item_type

            if current_item_type != prev_item_type:
                has_distinction = True

        if not has_distinction and len(self) >= limit:
            return batch

    def _time_gap_check(self, field_to_check, limit):
        batch = self.state[-limit:]
        batch_len = len(batch)
        if batch_len and batch_len == limit \
                and time.time() - batch[0]['_timestamp'] <= self._time_gap:
            return self._frame_counter_check(field_to_check, limit)

    def check(self, field_to_check=None, limit=None):
        if not limit:
            limit = self._state_threshold
        if self._type == 'frame_counter':
            return self._frame_counter_check(field_to_check, limit)
        elif self._type == 'time_gap':
            return self._time_gap_check(field_to_check, limit)

    def reset(self):
        self.state = []

    def __len__(self):
        return len(self.state)
