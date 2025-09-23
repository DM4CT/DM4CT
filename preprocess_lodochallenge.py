import os
import numpy as np
from tqdm import tqdm
import pydicom
from tifffile import imwrite

data_path = 'your raw data here'
tar_path = 'path to save normalized images'
train_folders = ['L067', 'L096', 'L109', 'L192', 'L286', 'L291', 'L310', 'L333']
test_folders = ['L506']

# Step 1: Collect global min and max across all volumes
global_min, global_max = np.inf, -np.inf

for folder in train_folders + test_folders:
    cur_folder = os.path.join(data_path, folder, 'full_1mm_sharp')
    files = os.listdir(cur_folder)
    files.sort()
    for file in files:
        img = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array.astype(np.float32)
        global_min = min(global_min, img.min())
        global_max = max(global_max, img.max())

# Step 2: Normalize using global min/max and save
for folder in tqdm(train_folders, desc='Processing Train'):
    cur_folder = os.path.join(data_path, folder, 'full_1mm_sharp')
    files = os.listdir(cur_folder)
    files.sort()
    for i, file in enumerate(files):
        img = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array.astype(np.float32)
        img_norm = 2 * (img - global_min) / (global_max - global_min) - 1
        imwrite(os.path.join(tar_path, "train", f"{folder}_{str(i).zfill(3)}.tif"), img_norm)

for folder in tqdm(test_folders, desc='Processing Test'):
    cur_folder = os.path.join(data_path, folder, 'full_1mm_sharp')
    files = os.listdir(cur_folder)
    files.sort()
    for i, file in enumerate(files):
        img = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array.astype(np.float32)
        img_norm = 2 * (img - global_min) / (global_max - global_min) - 1
        imwrite(os.path.join(tar_path, "test", f"{folder}_{str(i).zfill(3)}.tif"), img_norm)
