from __future__ import division
import os, sys, glob, re
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
PROBLEM_FILES = ['LUNG1-212', 'LUNG1-258', 'LUNG1-043', 'LUNG1-045', 'LUNG1-083', 'LUNG1-094', 'LUNG1-103', 'LUNG1-104', 'LUNG1-139', 'LUNG1-189', 'LUNG1-307', 'LUNG1-012', 'LUNG1-021', 'LUNG1-039']

# MISSING_SLICE_FILES: some patients' slices which contain contour are missing 
MISSING_SLICE_FILES = ['LUNG1-043', 'LUNG1-045', 'LUNG1-139', 'LUNG1-039']
Z = {'LUNG1-039': -597.4,'LUNG1-043': -561.9, 'LUNG1-045': -487.4, 'LUNG1-139': -395.0}

# MISSING_LABEL_FILES: some patients' PTV1, CTV1 are missing
MISSING_LABEL_FILES = ['LUNG1-057', 'LUNG1-066', 'LUNG1-149', 'LUNG1-173', 'LUNG1-176', 'LUNG1-177', 'LUNG1-252', 'LUNG1-255', 'LUNG1-260', 'LUNG1-278', 'LUNG1-286', 'LUNG1-298', 'LUNG1-305', 'LUNG1-306', 'LUNG1-311', 'LUNG1-315', 'LUNG1-321', 'LUNG1-015']

# missing CTV1, PTV1
TABLE = {'LUNG1-015': {'PTV1': 'PTV1', 'CTV1':  None },'LUNG1-057': {'PTV1': 'PTV1', 'CTV1': 'CTV2'}, 
         'LUNG1-066': {'PTV1': 'PTV1', 'CTV1': 'CTVROK'}, 'LUNG1-149': {'PTV1':  None , 'CTV1':  None }, 
         'LUNG1-173': {'PTV1': 'PTV1', 'CTV1': 'PTV3'}, 'LUNG1-176': {'PTV1': 'PTV1', 'CTV1': 'CTV3'}, 
         'LUNG1-177': {'PTV1': 'PTV1', 'CTV1': 'CTVTUMOR'}, 'LUNG1-252': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-255': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-260': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-278': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-286': {'PTV1': 'PTV1KOPIE', 'CTV1': 'CTV1KOPIE'}, 
         'LUNG1-298': {'PTV1': 'PTVSBRT', 'CTV1': 'CTV1'}, 'LUNG1-305': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-306': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-311': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-315': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-321': {'PTV1': 'PTV1', 'CTV1':  None }}

# for multitasking, we use GTV1 to compensate missing CTV1, PTV1
TABLE = {'LUNG1-015': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-057': {'PTV1': 'PTV1', 'CTV1': 'CTV2'}, 
         'LUNG1-066': {'PTV1': 'PTV1', 'CTV1': 'CTVROK'}, 'LUNG1-149': {'PTV1': 'GTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-173': {'PTV1': 'PTV1', 'CTV1': 'PTV3'}, 'LUNG1-176': {'PTV1': 'PTV1', 'CTV1': 'CTV3'}, 
         'LUNG1-177': {'PTV1': 'PTV1', 'CTV1': 'CTVTUMOR'}, 'LUNG1-252': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-255': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-260': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-278': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-286': {'PTV1': 'PTV1KOPIE', 'CTV1': 'CTV1KOPIE'}, 
         'LUNG1-298': {'PTV1': 'PTVSBRT', 'CTV1': 'CTV1'}, 'LUNG1-305': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-306': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-311': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-315': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-321': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}}


# read contour data
def read_structure(structure, patient_name):
    contours = []    
    if patient_name == 'LUNG1-170': # some contour data in 'PTV1' and "CTV1' are out of the bound, remove them.
        for i in range(len(structure.StructureSetROISequence)):
            name = structure.StructureSetROISequence[i].ROIName.upper() # change name to uppercase        
            name = re.sub('[-_ ]', '', name) # remove all whitespaces and dashs            
            if name == 'PTV1': # PTV means planning tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:
                    if s.ContourData[2] > -347:
                        continue
                    else:
                        contour['contour'].append(s.ContourData)
                contour['name'] = 'PTV1'
                contours.append(contour)            
            if name == 'CTV1': # CTV means clinical tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:
                    if s.ContourData[2] > -347:
                        continue
                    else:
                        contour['contour'].append(s.ContourData)
                contour['name'] = 'CTV1'
                contours.append(contour)            
            if name == 'GTV1': # GTV means gross tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'GTV1'
                contours.append(contour)    
    elif patient_name == 'LUNG1-212': # contour data deviate by 1mm, correct them
        for i in range(len(structure.StructureSetROISequence)):
            name = structure.StructureSetROISequence[i].ROIName.upper() # change name to uppercase        
            name = re.sub('[-_ ]', '', name) # remove all whitespaces and dashs            
            if name == 'PTV1': # PTV means planning tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:
                    bias = np.zeros_like(s.ContourData)
                    for j in range(2, len(bias), 3):
                        bias[j] = -1
                    contour['contour'].append(s.ContourData + bias)
                contour['name'] = 'PTV1'
                contours.append(contour)
            if name == 'CTV1': # CTV means clinical tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:
                    bias = np.zeros_like(s.ContourData)
                    for j in range(2, len(bias), 3):
                        bias[j] = -1
                    contour['contour'].append(s.ContourData + bias)
                contour['name'] = 'CTV1'
                contours.append(contour)            
            if name == 'GTV1': # GTV means gross tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:
                    bias = np.zeros_like(s.ContourData)
                    for j in range(2, len(bias), 3):
                        bias[j] = -1
                    contour['contour'].append(s.ContourData + bias)
                contour['name'] = 'GTV1'
                contours.append(contour)    
    elif patient_name == 'LUNG1-242': # contains two 'PTV1', choose the one that contains 85 contour consequences.
        for i in range(len(structure.StructureSetROISequence)):
            name = structure.StructureSetROISequence[i].ROIName.upper() # change name to uppercase        
            name = re.sub('[-_ ]', '', name) # remove all whitespaces and dashs            
            if name == 'PTV1': # PTV means planning tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                if len(contour['contour']) == 85:
                    contour['name'] = 'PTV1'
                    contours.append(contour)
            if name == 'CTV1': # CTV means clinical tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'CTV1'
                contours.append(contour)            
            if name == 'GTV1': # GTV means gross tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'GTV1'
                contours.append(contour)    
    elif patient_name in MISSING_SLICE_FILES:
        for i in range(len(structure.StructureSetROISequence)):            
            name = structure.StructureSetROISequence[i].ROIName.upper() # change name to uppercase        
            name = re.sub('[-_ ]', '', name) # remove all whitespaces and dashs            
            if name == 'PTV1': # PTV means planning tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:                    
                    if s.ContourData[2] == Z[patient_name]:
                        continue
                    else:
                        contour['contour'].append(s.ContourData)
                contour['name'] = 'PTV1'
                contours.append(contour)
            if name == 'CTV1': # CTV means clinical tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:                    
                    if s.ContourData[2] == Z[patient_name]:
                        continue
                    else:
                        contour['contour'].append(s.ContourData)
                contour['name'] = 'CTV1'
                contours.append(contour)            
            if name == 'GTV1': # GTV means gross tumor volume
                contour = {}
                contour['contour'] = []
                for s in structure.ROIContourSequence[i].ContourSequence:                    
                    if s.ContourData[2] == Z[patient_name]:
                        continue
                    else:
                        contour['contour'].append(s.ContourData)
                contour['name'] = 'GTV1'
                contours.append(contour)
    elif patient_name in MISSING_LABEL_FILES:
        for i in range(len(structure.StructureSetROISequence)):            
            name = structure.StructureSetROISequence[i].ROIName.upper() # change name to uppercase 
            name = re.sub('[-_ ]', '', name) # remove all whitespaces and dashs
            if name == TABLE[patient_name]['PTV1']: # PTV means planning tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'PTV1'
                contours.append(contour)
            if name == TABLE[patient_name]['CTV1']: # CTV means clinical tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'CTV1'
                contours.append(contour)            
            if name == 'GTV1': # GTV means gross tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'GTV1'
                contours.append(contour)
    else:
        for i in range(len(structure.StructureSetROISequence)):
            name = structure.StructureSetROISequence[i].ROIName.upper() # change name to uppercase        
            name = re.sub('[-_ ]', '', name) # remove all whitespaces and dashs            
            if name == 'PTV1': # PTV means planning tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'PTV1'
                contours.append(contour)
            if name == 'CTV1': # CTV means clinical tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'CTV1'
                contours.append(contour)            
            if name == 'GTV1': # GTV means gross tumor volume
                contour = {}
                contour['contour'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                contour['name'] = 'GTV1'
                contours.append(contour)
    assert len(contours) == 3 # make sure that every patient has PTV-1, CTV-1 and GTV-1 data    
    return contours

# generate multiple binary label map from contour data
def get_labels(contours, shape, slices):    
    z = [np.around(s.ImagePositionPatient[2], 1) for s in slices] # save every slice's z coordinate (x.ImagePositionPatient[2])
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]
    labels = []
    for contour in contours: # contour contains several contour sequences        
        label = {}        
        label_map = np.zeros(shape, dtype=np.int8)
        for contour_sequence in contour['contour']: # contour sequence depicts the tumor on the slice
            nodes = np.array(contour_sequence).reshape((-1, 3)) # every (x, y, z) in contour sequence is a node
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0 # make sure that the nodes are in the same slice
            z_index = z.index(np.around(nodes[0, 2], 1)) # search which slice the nodes are locate at
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label_map[rr, cc, z_index] = 1
        label['label'] = label_map
        label['name'] = contour['name']
        labels.append(label)
    return labels

# Some scanners have cylindrical scanning bounds, but the output image is square. The pixels that fall outside of these bounds get the fixed value -2000(-2048). So we need to set these values to air value.
def remove_outlier(pixel_array):    
    image = pixel_array.copy() # pixel_array is read-only
    air_value = pixel_array.min() # the air value (-1000/-1024/0) is the smallest value in Hounsfield Unit (air value is 0 because s.RescaleIntercept is -1024)
    image[image <= -2000] = air_value
    return image

# set pixels' value in (lower, upper)
def threshold(image, lower, upper):
    image[image < lower] = lower
    image[image > upper] = upper
    return image

# resample the image to a defined resolution (default value is 1mm^3, an isotropic resolution)
def resample(image, slices, new_spacing=[1, 1, 1]):    
    spacing = np.array([1, 1, slices[0].SliceThickness], dtype=np.float32) # determine current pixel spacing (assume that row spacing and column spacing is 1mm)    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor    
    image = zoom(image, real_resize_factor, mode='nearest')    
    return image, real_resize_factor
"""
# compute affine matrix(RAS) and consider image resampling (assume that slices are sorted) (reference: https://www.slicer.org/wiki/Coordinate_systems, https://github.com/innolitics/dicom-numpy/blob/master/dicom_numpy/combine_slices.py)
def get_affine_matrix(slices, image_resize_factor):    
    image_orientation = slices[0].ImageOrientationPatient
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    row_spacing, column_spacing = slices[0].PixelSpacing
    if len(slices) > 1:
        slice_positions = [np.dot(slice_cosine, d.ImagePositionPatient) for d in slices]
        slice_positions_diffs = np.diff(sorted(slice_positions))
        slice_spacing = np.mean(slice_positions_diffs)
    else:
        slice_spacing = 0.0
    
    # if the image is resampled, adjust the spacing accordingly
    row_spacing = row_spacing / image_resize_factor[1]
    column_spacing = column_spacing / image_resize_factor[2]
    slice_spacing = slice_spacing / image_resize_factor[0]
    
    transform = np.identity(4, dtype=np.float32)    
    return transform
"""
# read the image, multilabels from a folder containing dicom files
def read_image_labels(path, name, new_spacing):
    for root, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(root, '*.dcm'))
        if len(dcms) == 1: # if there is only one .dcm file, it contains contour   
            structure = pydicom.dcmread(dcms[0])
            contours = read_structure(structure, name)
        elif len(dcms) > 1: # if there are several .dcm files, they are slices
            slices = [pydicom.dcmread(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) # sort slices by z coordinate (x.ImagePositionPatient[2])
            image = np.stack([remove_outlier(s.pixel_array) * s.RescaleSlope + s.RescaleIntercept for s in slices], axis=-1).astype(np.int16)# rescale slices and combine them into a 3D image (reference: https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/)
    labels = get_labels(contours, image.shape, slices)
    image, image_resize_factor = resample(image, slices, new_spacing)
    for label in labels:
        label['label'], label_resize_factor = resample(label['label'], slices, new_spacing)    
    return image, labels

# read the image from a folder containing dicom files
def read_image(path, name, new_spacing):
    for root, dirs, files in os.walk(path):
        dcms = glob.glob(os.path.join(root, '*.dcm'))
        if len(dcms) > 1: # if there are several .dcm files, they are slices
            slices = [pydicom.dcmread(dcm) for dcm in dcms]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2])) # sort slices by z coordinate (x.ImagePositionPatient[2])
            image = np.stack([remove_outlier(s.pixel_array) * s.RescaleSlope + s.RescaleIntercept for s in slices], axis=-1).astype(np.int16)# rescale slices and combine them into a 3D image (reference: https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/)
    image, image_resize_factor = resample(image, slices, new_spacing)
    return image

if __name__ == '__main__':
    output_path = OUTPUT_PATH
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # training, validation data
    for path in [TRAINING_PATH, VALIDATION_PATH]:
        input_path = path
        subjects = [os.path.join(input_path, name)
                    for name in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, name))]
        for sub in subjects:
            name = os.path.basename(sub)
            image, labels = read_image_labels(sub, name, new_spacing=[2, 2, 3])            
            output_affine = np.eye(4)                
            image = nib.Nifti1Image(image, output_affine)
            nib.save(image, output_path + '/' + name + 'image.nii.gz')
            for label in labels:
                _label = nib.Nifti1Image(label['label'], output_affine)
                nib.save(_label, output_path + '/' + name + 'label_' + label['name'] + '.nii.gz')
                   
    # test data    
    input_path = TEST_PATH
    subjects = [os.path.join(input_path, name)
                for name in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, name))]
    for sub in subjects:
        name = os.path.basename(sub)        
        image = read_image(sub, name, new_spacing=[2, 2, 3])        
        output_affine = np.eye(4)
        image = nib.Nifti1Image(image, output_affine)
        nib.save(image, output_path + '/' + name + 'image.nii.gz')  