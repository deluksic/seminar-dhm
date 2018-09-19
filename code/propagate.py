import numpy as np
import cv2
import os
from hologram import inline
from hologram.utils import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().wm_withdraw()
fname = askopenfilename()

w = 512
inline.init(w)

rec_depth = 0
rec_depth_step = 1e-4

img = None
if str(fname).endswith('.npy'):
    img = np.load(fname)
else:
    img = cv2.imread(fname)
    img = img[:, :, 1]
    img = crop_frame_h(w, img) / 255

cv2.imshow('original', to01range(img))

# collect_bg = False
while 'Forever':
    # Display the picture
    cv2.imshow('screen', to01range(inline.reconstruct(img)))

    # Key presses
    key = cv2.waitKeyEx(0)

    if key == ord('o'):
        inline.refine(img, iters=100)

    if key == 2490368:  # up key
        rec_depth -= rec_depth_step
        inline.set_reconstruction_depth(rec_depth)

    if key == 2621440:  # down key
        rec_depth += rec_depth_step
        inline.set_reconstruction_depth(rec_depth)

    if key == ord('q'):
        cv2.destroyAllWindows()
        Tk().quit()
        break

    if key == ord('e'):
        np.save(os.path.splitext(fname)[0], inline.reconstruct(img))

    if key == ord('m'):
        cv2.imshow('propagator', np.fft.fftshift(
            np.real(inline.prop_gpu.get())))

    # if key == ord('p'):
    #     inline.reset_noise()

    # if key == ord('b'):
    #     collect_bg = not collect_bg

    # filter background
    # if collect_bg:
    #     inline.set_noise(img, 0.1)
    # inline.filter(img)
