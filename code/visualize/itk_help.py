import SimpleITK as sitk
import numpy as np
import cv2
import sys

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    return ct_scan

fname = sys.argv[1]
print(fname)
mat = load_itk(fname)
print(mat.shape)
img = mat[2]
cv2.imshow('heh', img)
cv2.waitKey(0)
cv2.destroyAllWindows()