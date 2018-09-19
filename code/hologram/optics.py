import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


mod = None
with open('hologram/optics_gpu.c') as source_code:
    data = str(source_code.read())
    mod = SourceModule(data)
assert mod is not None

propagator__gpu = mod.get_function("propagator")
apply_mask__gpu = mod.get_function("apply_mask")
apply_conj_mask__gpu = mod.get_function("apply_conj_mask")
apply_phase__gpu = mod.get_function("apply_phase")
abs2__gpu = mod.get_function("abs2")
amp_from_intensity__gpu = mod.get_function("amp_from_intensity")
filter_obj_dom__gpu = mod.get_function("filter_obj_dom")
cx_to_phase__gpu = mod.get_function("cx_to_phase")


def propagator(w, a_gpu, lmda, area, z):
    assert w % 2 == 0
    propagator__gpu(a_gpu,
                    np.float32(lmda),
                    np.float32(area),
                    np.float32(z),
                    block=(1, 1, 1), grid=(w//2, w//2))


def apply_mask(w, dest_gpu, src_gpu, mask_gpu):
    apply_mask__gpu(dest_gpu,
                    src_gpu,
                    mask_gpu,
                    block=(1, 1, 1), grid=(w, w))


def apply_conj_mask(w, dest_gpu, src_gpu, mask_gpu):
    apply_conj_mask__gpu(dest_gpu,
                         src_gpu,
                         mask_gpu,
                         block=(1, 1, 1), grid=(w, w))


def apply_phase(w, dest_gpu, src_gpu, phase_gpu):
    apply_phase__gpu(dest_gpu,
                     src_gpu,
                     phase_gpu,
                     block=(1, 1, 1), grid=(w, w))


def abs2(w, dest_gpu, src_gpu):
    abs2__gpu(dest_gpu,
              src_gpu,
              block=(1, 1, 1), grid=(w, w))


def cx_to_phase(w, dest_gpu, src_gpu):
    cx_to_phase__gpu(dest_gpu,
                     src_gpu,
                     block=(1, 1, 1), grid=(w, w))


def amp_from_intensity(w, dest_gpu, src_gpu):
    amp_from_intensity__gpu(dest_gpu,
                            src_gpu,
                            block=(1, 1, 1), grid=(w, w))


def filter_obj_dom(w, t_gpu):
    filter_obj_dom__gpu(t_gpu, block=(1, 1, 1), grid=(w, w))
