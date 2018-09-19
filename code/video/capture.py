import cv2
from time import sleep
from .video_source import VideoSource


class VideoCapture(VideoSource):
    def __init__(self, src=0):
        super(VideoSource, self).__init__()
        self.src = src

    def _start(self):
        self.stream = cv2.VideoCapture(self.src)
        # self.stream.set(cv2.CAP_PROP_EXPOSURE, 0.0)

    def _stop(self, *_):
        self.stream.release()

    def _get_frame(self):
        (ret, self.frame) = self.stream.read()
        if isinstance(self.src, str):
            # slow down for video
            sleep(1/30)
            if not ret:
                # reset video for looping
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
