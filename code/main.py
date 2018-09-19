import numpy as np
import cv2
from hologram import inline
from hologram.utils import *
from video.screen import ScreenCapture
from video.capture import VideoCapture
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().wm_withdraw()

w = 512
inline.init(w)
collect_bg = False

rec_depth = 0
rec_depth_step = 1e-4

with VideoCapture(1) as vs:
# with VideoCapture(askopenfilename() or 0) as vs:
# with ScreenCapture(w, w) as vs:
    while 'Forever':
        try:
            img = vs.read()
            img = img[:, :, 1]
            img = crop_frame_h(w, img) / 255
        except Exception as ex:
            print(ex)
            continue

        # normalize with background
        if collect_bg:
            inline.set_bg(img, 0.1)
        inline.normalize(img)

        # Display the picture
        # cv2.imshow('original', img)
        cv2.imshow('screen', to01range(inline.reconstruct(img)))

        # Key presses
        key = cv2.waitKeyEx(1)

        if key == ord(' '):
            key = cv2.waitKey(0)

        if key == ord('a'):
            vs.paused = not vs.paused

        if key == ord('o'):
            inline.refine(img)

        if key == 2490368:  # up key
            rec_depth -= rec_depth_step
            inline.set_reconstruction_depth(rec_depth)

        if key == 2621440:  # down key
            rec_depth += rec_depth_step
            inline.set_reconstruction_depth(rec_depth)

        if key == ord('p'):
            inline.reset_bg()

        if key == ord('b'):
            collect_bg = not collect_bg

        if key == ord('q'):
            cv2.destroyAllWindows()
            Tk().quit()
            break
