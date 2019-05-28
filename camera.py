import subprocess
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging
import os
import io
import signal

import numpy as np

from .frame import Frame  # noqa

logger = logging.getLogger('camera')


# Camera to read np frames from RTSP streams
class Camera():
    def __init__(self, address, name, interval):
        self._address = address
        self._name = name
        self._interval = interval
        self._restart_if_down = True
        self.frame = None

    def _read(self, path, queue):
        with io.open(path, 'rb') as stream:
            image = stream.read()
        self.frame = Frame(image)
        if not self.frame.is_corrupted():
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
            '-i', self._address,
            '-loglevel', kwargs['log_level'],
            '-f', kwargs['format'],
            '-qscale:v', str(kwargs['quality']),
            '-s', f'{width}x{height}',
            '-vf', f'fps=fps={fps}',
            '-threads', '1',
            '-updatefirst', '1',
            image_path
        ], preexec_fn=os.setsid)
        sleep(kwargs['loop_delay'])
        logger.info(f'{self._name} is started to work on {fps} fps')

        while self._process.poll() is None:
            self._read(image_path, kwargs['queue'])

        if self._restart_if_down:
            sleep(kwargs['restart_delay'])
            self.restart(**kwargs)

    def _kill_process(self):
        try:
            # Ohh SIGTERM can't kill all processes anyway
            os.killpg(
                os.getpgid(self._process.pid),
                signal.SIGTERM
            )
            logger.info(f'{self._name} camera is killed')
        except Exception as ex:
            logger.info(f'{self._name} can\'t find and kill camera process')

    def start(self, **kwargs):
        self._restart_if_down = True
        ThreadPoolExecutor().submit(self._start_camera_loop, **kwargs)

    def stop(self):
        self._restart_if_down = False
        self._kill_process()

    def restart(self, **kwargs):
        logger.info(f'{self._name} camera is going to restart, good luck')
        self._kill_process()
        self.start(**kwargs)
