import os
import numpy as np
from tqdm import tqdm
import pydicom
from tifffile import imwrite

data_path = 'YOUR_DATA_PATH'  # replace with your data path

# define your train and test split
train_folders = ['L067', 'L096', 'L109', 'L192', 'L286', 'L291', 'L310', 'L333']
test_folder = [ 'L506']

for folder in tqdm(train_folders):
    cur_folder = os.path.join(data_path, folder, 'full_1mm_sharp')
    files = os.listdir(cur_folder)
    files.sort()
    volumes = np.zeros((len(files), 512, 512), dtype=np.float32)
    
    for i, file in enumerate(files):
        volumes[i] = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array

    vmin, vmax = volumes.min(), volumes.max()
    # Normalize the volumes to the range [-1, 1]
    volumes = 2 * (volumes - vmin) / (vmax - vmin) - 1
    for i in range(len(files)):
        imwrite(os.path.join(data_path, "train", f"{folder}_{str(i).zfill(3)}.tif"), volumes[i])

for folder in tqdm(test_folder):
    cur_folder = os.path.join(data_path, folder, 'full_1mm_sharp')
    files = os.listdir(cur_folder)
    files.sort()
    volumes = np.zeros((len(files), 512, 512), dtype=np.float32)

    for i, file in enumerate(files):
        volumes[i] = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array

    vmin, vmax = volumes.min(), volumes.max()
    # Normalize the volumes to the range [-1, 1]
    volumes = 2 * (volumes - vmin) / (vmax - vmin) - 1
    for i in range(len(files)):
        imwrite(os.path.join(data_path, "test", f"{folder}_{str(i).zfill(3)}.tif"), volumes[i])
