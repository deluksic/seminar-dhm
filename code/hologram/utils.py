import numpy as np
import cv2

def crop_frame_h(w, frame):
    frame = cv2.resize(frame, (frame.shape[1] * w // frame.shape[0], w))
    return frame[:, frame.shape[1]//2 - w//2:frame.shape[1]//2 + w//2]


def crop_frame_border_h(w, frame):
    frame = cv2.resize(
        frame, (frame.shape[1] * (w//2) // frame.shape[0], w//2))
    frame = frame[:, frame.shape[1]//2 - w//4:frame.shape[1]//2 + w//4]
    newframe = np.zeros((w, w), dtype=frame.dtype)
    newframe[w//4:w-w//4, w//4:w-w//4] = frame
    return newframe


def to01range(img):
    return (img-img.min())/(img.max()-img.min())
