import os
import numpy as np
from tqdm import tqdm
import pydicom
from tifffile import imwrite, imread
from PIL import Image

data_path = 'YOUR_DATA_PATH'  # replace with your data path

target_path_train = 'YOUR_PATH_TO_TRAIN'  # replace with your path to train
target_path_test = 'YOUR_PATH_TO_TEST'  # replace with your path to test
os.makedirs(target_path_train, exist_ok=True)
os.makedirs(target_path_test, exist_ok=True)

for i in tqdm(range(500, 3500)):
    cur_slice = imread(os.path.join(data_path, f"recon{str(i).zfill(4)}.tif"))
    
    lower, upper = np.percentile(cur_slice, [0.5, 99.5])
    # Clip, center, and normalize
    cur_slice = np.clip(cur_slice, lower, upper)
    cur_slice -= np.mean(cur_slice)
    # Normalize the image to the range [-1, 1]
    cur_slice = 2 * (cur_slice - cur_slice.min()) / (cur_slice.max() - cur_slice.min()) - 1

    # Resize to 512x512
    img = Image.fromarray(cur_slice)
    img = img.resize((512, 512), Image.LANCZOS)

    # Save slice-by-slice, no need to store in `volume`
    if i < 3000:
        imwrite(os.path.join(target_path_train, f"{str(i-500).zfill(4)}.tif"), np.array(img, dtype=np.float32))
    else:
        imwrite(os.path.join(target_path_test, f"{str(i-500).zfill(4)}.tif"), np.array(img, dtype=np.float32))

