import numpy as np
import math
import cv2
import time
from numba import cuda
from pyculib.fft import fft_inplace, ifft_inplace
from video_source import Webcam, ScreenCapture


@cuda.jit
def gpu_mul(target, mask):
    i, j = cuda.grid(2)
    target[i, j] *= mask[i, j]


@cuda.jit(device=True, inline=True)
def fftshift(i, j, w):
    return (w + w//2-i) % w, (w + w//2-j) % w


@cuda.jit('complex64(complex64)', device=True, inline=True)
def cx_exp(arg):
    e = math.exp(arg.real)
    s = math.sin(arg.imag)
    c = math.cos(arg.imag)
    return complex(c, s)*e


@cuda.jit
def gpu_lens(lens, w, f):
    i, j = cuda.grid(2)
    x = 2.0 * (j / w) - 1
    y = 2.0 * (i / w) - 1
    i, j = fftshift(i, j, w)
    lens[i, j] = cx_exp(1j * f * (x**2 + y**2))


@cuda.jit
def gpu_circ(mask, w, cx, cy, r1, r2):
    i, j = cuda.grid(2)
    x = 2.0 * (j / w) - 1
    y = 2.0 * (i / w) - 1
    d = (x-cx)**2 + (y-cy)**2
    if d < r1**2 or d > r2**2:
        i, j = fftshift(i, j, w)
        mask[i, j] *= 0


@cuda.jit
def gpu_shift(ft, target, dx, dy, w):
    i, j = cuda.grid(2)
    orig = ft[i, j]
    target[i, j] += orig / 3
    i, j = fftshift(i, j, w)
    i1 = i - dy
    j1 = j - dx
    if i1 >= 0 and i1 < w and j1 >= 0 and j1 < w:
        i1, j1 = fftshift(i1, j1, w)
        target[i1, j1] += orig / 3
    i2 = i + dy
    j2 = j + dx
    if i2 >= 0 and i2 < w and j2 >= 0 and j2 < w:
        i2, j2 = fftshift(i2, j2, w)
        target[i2, j2] += orig / 3


w = 256
vs = ScreenCapture(w, w).start()

ctype = np.complex64
mask = np.ones((w, w), dtype=ctype)
fft = np.zeros_like(mask)
d_fft = cuda.to_device(fft)
shfft = np.array(fft)
d_shfft = cuda.to_device(shfft)
d_mask = cuda.to_device(mask)
gpu_circ[(w, w), 1](d_mask, w, 0, 0, 0, 0.4)

lens = np.ones_like(mask)
d_lens = cuda.to_device(lens)
gpu_lens[(w, w), 1](d_lens, w, 10)

stime = time.time()
while(True):
    t = time.time() - stime
    # Capture frame-by-frame
    frame = vs.read()

    # scale, gray, display original
    frame = cv2.resize(frame, (w, w))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    d_fft.copy_to_device(np.array(frame / 255, dtype=ctype))
    fft_inplace(d_fft)
    gpu_mul[(w, w), 1](d_fft, d_mask)
    gpu_shift[(w, w), 1](d_fft, d_shfft, 60, 100, w)
    ifft_inplace(d_shfft)
    d_shfft.to_host()
    cv2.imshow('orig_frame', frame)
    # cv2.imshow('fft', np.log(np.abs(np.fft.fftshift(shfft)))/10)
    cv2.imshow('result', np.real(shfft)/(w*w))
    shfft[:,:] = 0
    d_shfft.copy_to_device(shfft)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('h'):
        gpu_mul[(w, w), 1](d_mask, d_lens)

    # kill on q
    if key == ord('q'):
        break

# When everything done, release the capture
vs.stop() 
cv2.destroyAllWindows()
