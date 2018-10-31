from __future__ import division
import csv
import os, sys, glob
import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage.interpolation import zoom
from skimage.draw import polygon

INPUT_PATH = ''
TRAINING_PATH = INPUT_PATH + '/VIP_CUP18_TrainingData'
VALIDATION_PATH = INPUT_PATH + '/VIP_CUP18_ValidationData'
TEST_PATH = INPUT_PATH + '/VIP_CUP18_TestData'
OUTPUT_PATH = ''

# check if the slices are uniform spacing (3mm)
def is_uniform_spacing(slices):
    z = [np.around(s.ImagePositionPatient[2], 1) for s in slices] # save every slice's z coordinate (x.ImagePositionPatient[2])
    diff = np.diff(z) # compute spacing
    return np.allclose(diff, [3.0], rtol=1e-5, atol=0)

# Read the image, label from a folder containing dicom files
def read_image(path, new_spacing):
    for root, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(root, '*.dcm'))        
        if len(dcms) > 1: # if there are several .dcm files, they are slices
            slices = [pydicom.dcmread(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) # sort slices by z coordinate (x.ImagePositionPatient[2])
            z = [np.around(s.ImagePositionPatient[2], 1) for s in slices]
            if not is_uniform_spacing(slices):
                return None
            else:
                return []

if __name__ == '__main__':        
    for path in [TRAINING_PATH, VALIDATION_PATH, TEST_PATH]:
        input_path = path        
        subjects = [os.path.join(input_path, name)
                    for name in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, name))]
        for sub in subjects:
            name = os.path.basename(sub)
            image = read_image(sub, new_spacing=[3, 3, 3])
            if image is None:
                print('The slice spacing of {} is non-uniform'.format(name))
                PROBLEM_FILES.append(name)    
    print('PROBLEM FILES: {}'.format(PROBLEM_FILES))