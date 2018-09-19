# import the necessary packages
from abc import ABC, abstractmethod
from threading import Thread


class VideoSource(ABC):
    def __enter__(self):
        self.frame = None
        self.stopped = False
        self.paused = False
        self._start()
        self._get_frame()
        Thread(target=self.update, args=()).start()
        return self

    def __exit__(self, *_):
        self.stopped = True
        self.paused = False
        self._stop(*_)

    @abstractmethod
    def _start(self, parameter_list):
        raise NotImplementedError

    @abstractmethod
    def _get_frame(self):
        raise NotImplementedError

    @abstractmethod
    def _stop(self, *_):
        raise NotImplementedError

    def read(self):
        return self.frame

    def update(self):
        while True:
            while self.paused:
                if self.stopped:
                    break
            self._get_frame()
