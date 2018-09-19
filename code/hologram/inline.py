import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from .pyfft.cuda import Plan
from .optics import (propagator, apply_mask, abs2,
                     amp_from_intensity, apply_phase,
                     apply_conj_mask, filter_obj_dom, cx_to_phase)
from .utils import *


w = 256
lmda = 780e-9
side = 1.8e-3 # side length of sensor/specimen
ctype = np.complex64

frame_gpu = None
phase_gpu = None
field_gpu = None
prop_gpu = None

FFT = None

bg = None


def init(size):
    global w, FFT, frame_gpu, phase_gpu, field_gpu, prop_gpu
    w = size
    FFT = Plan((w, w))

    frame_gpu = gpuarray.to_gpu(np.zeros((w, w), dtype=np.float32))
    phase_gpu = gpuarray.to_gpu(np.zeros((w, w), dtype=np.float32))
    field_gpu = gpuarray.to_gpu(np.zeros((w, w), dtype=ctype))
    prop_gpu = gpuarray.to_gpu(np.ones((w, w), dtype=ctype))

    reset_bg()


def reconstruct(frame):
    global w, FFT, frame_gpu, phase_gpu, field_gpu
    frame_gpu.set(frame.astype(np.float32))
    amp_from_intensity(w, frame_gpu, frame_gpu)
    apply_phase(w, field_gpu, frame_gpu, phase_gpu)
    FFT.execute(field_gpu)
    apply_mask(w, field_gpu, field_gpu, prop_gpu)
    FFT.execute(field_gpu, inverse=True)
    abs2(w, frame_gpu, field_gpu)
    result = frame_gpu.get()
    return result


def fourier_only(frame):
    global w, FFT, frame_gpu, phase_gpu, field_gpu
    frame_gpu.set(frame.astype(np.float32))
    amp_from_intensity(w, frame_gpu, frame_gpu)
    apply_phase(w, field_gpu, frame_gpu, phase_gpu)
    FFT.execute(field_gpu)
    abs2(w, frame_gpu, field_gpu)
    result = frame_gpu.get()
    return np.log(np.fft.fftshift(result)) / 10

def set_reconstruction_depth(z):
    global w, prop_gpu, lmda, side
    propagator(w, prop_gpu, lmda, side, z)
    print("Reconstruction depth: ", '%.2f' % (z*1e3), "mm")


def set_bg(frame, decay):
    global bg
    bg *= 1-decay
    bg += frame*decay


def normalize(frame):
    global bg
    frame /= bg


def reset_bg():
    global w, bg
    bg = np.ones((w, w), dtype=np.float32)


def refine(hologram, iters=50):
    global w

    # d_holo is original hologram amplitude
    d_intensity = gpuarray.to_gpu(hologram.astype(np.float32))
    d_holo = gpuarray.to_gpu(np.zeros((w, w), dtype=np.float32))
    amp_from_intensity(w, d_holo, d_intensity)

    cv2.imshow('result', hologram)
    cv2.waitKey(0)

    for kk in range(iters):
        # copy hologram with phase into field
        apply_phase(w, field_gpu, d_holo, phase_gpu)

        # field_gpu is now transmission function t
        FFT.execute(field_gpu)
        apply_mask(w, field_gpu, field_gpu, prop_gpu)
        FFT.execute(field_gpu, inverse=True)

        # filtering in object domain
        filter_obj_dom(w, field_gpu)

        abs2(w, d_intensity, field_gpu)
        intensity = d_intensity.get()
        cv2.imshow('result', intensity / np.max(intensity))
        cv2.waitKey(1)

        # calculating complex-valued wavefront in the detector plane
        FFT.execute(field_gpu)
        apply_conj_mask(w, field_gpu, field_gpu, prop_gpu)
        FFT.execute(field_gpu, inverse=True)

        cx_to_phase(w, phase_gpu, field_gpu)

    # reset phase
    phase_gpu.set(np.zeros((w, w), dtype=np.float32))


if __name__ == "__main__":
    w = 512
    init(w)
    set_focus(0.0342)
    data = []
    with open('hologram/hologram.data', 'r') as fi:
        for line in fi:
            data.append([float(s) for s in line.split('\t')])
        img = np.array(data)
        img = cv2.resize(img, (w, w))
        cv2.imshow('heh', img / np.max(img))
        cv2.waitKey(0)
        refine(img, iters=50)
