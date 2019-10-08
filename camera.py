import subprocess
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging
import os
import io
import signal
import time
import datetime

import numpy as np

from .frame import Frame  # noqa

logger = logging.getLogger('camera')


class Camera():
    def __init__(self, address, name, interval):
        self._address = address
        self._name = name
        self._interval = interval
        self.frame = None

    def _read(self, path, queue):
        try:
            with io.open(path, 'rb') as stream:
                image = stream.read()
        except Exception as ex:
            logger.info(f'{self._name} can\'t read image, skipped')
            sleep(self._interval)
            return
        frame = Frame(image)
        if not frame.is_corrupted():
            self.frame = frame
            queue.put(self.frame)
        sleep(self._interval)

    def _start_camera_loop(self, **kwargs):
        image_path = os.path.join('/ramdisk/', f'{self._name}.jpg')
        width = kwargs['resolution'][0]
        height = kwargs['resolution'][1]
        fps = kwargs['fps']
        self._process = subprocess.Popen([
            'ffmpeg',
            '-y',
            '-rtsp_transport', 'tcp',
            '-stimeout', '5000000',
            '-i', self._address,
            '-loglevel', kwargs['log_level'],
            '-f', kwargs['format'],
            '-qscale:v', str(kwargs['quality']),
            '-s', f'{width}x{height}',
            '-vf', f'fps=fps={fps}',
            '-threads', '1',
            '-vsync', 'vfr',
            '-updatefirst', '1',
            image_path
        ], preexec_fn=os.setsid)
        sleep(kwargs['loop_delay'])
        logger.info(f'{self._name} is started to work on {fps} fps')

        start_time = time.time()
        while self._process.poll() is None:
            self._read(image_path, kwargs['queue'])

        # Next code works if process of reading images fails and while loop is interrupted
        self.frame = None
        interrupted_time = time.time()
        working_time = str(
            datetime.timedelta(
                (interrupted_time - start_time) / (60 * 60 * 24)
            )
        )
        logger.info(f'{self._name} loop was interrupted, working time: {working_time}')
        if kwargs['restart']:
            sleep(kwargs['restart_delay'])
            self.restart(**kwargs)

    def _kill_process(self):
        try:
            # Ohh SIGTERM and SIGKILL can't kill all processes anyway
            os.killpg(
                os.getpgid(self._process.pid),
                signal.SIGTERM
            )
            logger.info(f'{self._name} camera process is killed')
        except Exception as ex:
            logger.info(f'{self._name} can\'t find and kill camera process')

    def start(self, **kwargs):
        ThreadPoolExecutor().submit(self._start_camera_loop, **kwargs)

    def stop(self):
        self.frame = None
        self._kill_process()

    def restart(self, **kwargs):
        logger.info(f'{self._name} camera is going to restart, let\'s pray')
        self.stop()
        self.start(**kwargs)
