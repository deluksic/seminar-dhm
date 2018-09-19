import mss
import numpy as np
from ctypes import windll, Structure, c_long, byref
from .video_source import VideoSource


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return {"x": pt.x, "y": pt.y}


class ScreenCapture(VideoSource):
    def __init__(self, width, height):
        super(VideoSource, self).__init__()
        self.width = width
        self.height = height

    def _start(self):
        self.sct = mss.mss().__enter__()

    def _stop(self, *_):
        self.sct.__exit__(*_)

    def _get_frame(self):
        pos = queryMousePosition()
        monitor = {'top': pos['y'] - self.height//2,
                   'left': pos['x'] - self.width//2,
                   'width': self.width, 'height': self.height}
        self.frame = np.array(self.sct.grab(monitor))
