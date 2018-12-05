import csv
import os

INPUT_PATH = ''
TRAINING_PATH = INPUT_PATH + '/VIP_CUP18_TrainingData'
VALIDATION_PATH = INPUT_PATH + '/VIP_CUP18_ValidationData'
TEST_PATH = INPUT_PATH + '/VIP_CUP18_TestData'
MISSING_LABEL_FILES = ['LUNG1-057', 'LUNG1-066', 'LUNG1-149', 'LUNG1-173', 'LUNG1-176', 'LUNG1-177', 'LUNG1-252', 'LUNG1-255', 'LUNG1-260', 'LUNG1-278', 'LUNG1-286', 'LUNG1-298', 'LUNG1-305', 'LUNG1-306', 'LUNG1-311', 'LUNG1-315', 'LUNG1-321', 'LUNG1-015']

TABLE = {'LUNG1-015': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-057': {'PTV1': 'PTV1', 'CTV1': 'CTV2'}, 
         'LUNG1-066': {'PTV1': 'PTV1', 'CTV1': 'CTVROK'}, 'LUNG1-149': {'PTV1':  None , 'CTV1':  None }, 
         'LUNG1-173': {'PTV1': 'PTV1', 'CTV1': 'PTV3'}, 'LUNG1-176': {'PTV1': 'PTV1', 'CTV1': 'CTV3'}, 
         'LUNG1-177': {'PTV1': 'PTV1', 'CTV1': 'CTVTUMOR'}, 'LUNG1-252': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-255': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-260': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-278': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-286': {'PTV1': 'PTV1KOPIE', 'CTV1': 'CTV1KOPIE'}, 
         'LUNG1-298': {'PTV1': 'PTVSBRT', 'CTV1': 'CTV1'}, 'LUNG1-305': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-306': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-311': {'PTV1': 'PTV1', 'CTV1':  None }, 
         'LUNG1-315': {'PTV1': 'PTV1', 'CTV1':  None }, 'LUNG1-321': {'PTV1': 'PTV1', 'CTV1':  None }}

TABLE = {'LUNG1-015': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-057': {'PTV1': 'PTV1', 'CTV1': 'CTV2'}, 
         'LUNG1-066': {'PTV1': 'PTV1', 'CTV1': 'CTVROK'}, 'LUNG1-149': {'PTV1': 'GTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-173': {'PTV1': 'PTV1', 'CTV1': 'PTV3'}, 'LUNG1-176': {'PTV1': 'PTV1', 'CTV1': 'CTV3'}, 
         'LUNG1-177': {'PTV1': 'PTV1', 'CTV1': 'CTVTUMOR'}, 'LUNG1-252': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-255': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-260': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-278': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-286': {'PTV1': 'PTV1KOPIE', 'CTV1': 'CTV1KOPIE'}, 
         'LUNG1-298': {'PTV1': 'PTVSBRT', 'CTV1': 'CTV1'}, 'LUNG1-305': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-306': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-311': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 
         'LUNG1-315': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}, 'LUNG1-321': {'PTV1': 'PTV1', 'CTV1': 'GTV1'}}

with open('dataset_split_PTV1.csv', 'w') as file:
    writer = csv.writer(file)
    for path, classification in zip([TRAINING_PATH, VALIDATION_PATH, TEST_PATH], ['Training', 'Validation', 'Inference']):
        input_path = path
        input_classification = classification
        subjects = [os.path.join(input_path, name)
                    for name in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, name))]
        for sub in subjects:
            name = os.path.basename(sub)            
            if name in MISSING_LABEL_FILES:
                if TABLE[name]['PTV1'] is not None:
                    _name = name[:5] + name[6:]
                    writer.writerow([_name, input_classification])
            else:
                _name = name[:5] + name[6:]
                writer.writerow([_name, input_classification])

with open('dataset_split_CTV1.csv', 'w') as file:
    writer = csv.writer(file)
    for path, classification in zip([TRAINING_PATH, VALIDATION_PATH, TEST_PATH], ['Training', 'Validation', 'Inference']):
        input_path = path
        input_classification = classification
        subjects = [os.path.join(input_path, name)
                    for name in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, name))]
        for sub in subjects:
            name = os.path.basename(sub)            
            if name in MISSING_LABEL_FILES:
                if TABLE[name]['CTV1'] is not None:
                    _name = name[:5] + name[6:]
                    writer.writerow([_name, input_classification])
            else:
                _name = name[:5] + name[6:]
                writer.writerow([_name, input_classification])

with open('dataset_split_GTV1.csv', 'w') as file:
    writer = csv.writer(file)
    for path, classification in zip([TRAINING_PATH, VALIDATION_PATH, TEST_PATH], ['Training', 'Validation', 'Inference']):
        input_path = path
        input_classification = classification
        subjects = [os.path.join(input_path, name)
                    for name in sorted(os.listdir(input_path)) if os.path.isdir(os.path.join(input_path, name))]
        for sub in subjects:
            name = os.path.basename(sub)            
            _name = name[:5] + name[6:]
            writer.writerow([_name, input_classification])
