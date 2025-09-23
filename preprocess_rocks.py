import astra
import numpy as np
import os
import torch
from tqdm import tqdm
from ct_reconstruction import sirt, fbp
from forward_operators_ct import Operator
from tifffile import imwrite, imread
from scipy.signal import medfilt

# F3_1 training F3_2 testing

def rivers_filter(sinogram, median_filter_size=11):
    if sinogram.ndim == 2:
        # 2D sinogram: average over rows to get a 1D vector
        average_row = np.mean(sinogram, axis=0)
        filtered_row = medfilt(average_row, kernel_size=median_filter_size)
        anomaly = average_row - filtered_row
        sinogram[:] -= anomaly
    elif sinogram.ndim == 3:
        # 3D sinogram: shape (N, rows, cols)
        # Compute the average row for each sinogram in the batch (averaging over axis 1)
        average_row = np.mean(sinogram, axis=1)  # shape (N, cols)
        # Apply median filter along the column axis only:
        # medfilt will treat the 0-th dimension (batch) separately if we specify kernel_size=(1, median_filter_size)
        filtered_row = medfilt(average_row, kernel_size=(1, median_filter_size))
        anomaly = average_row - filtered_row  # shape (N, cols)
        # Subtract the anomaly from each row in each sinogram
        sinogram = sinogram - anomaly[:, None, :]  # broadcasting over the row axis
    else:
        raise ValueError("Input sinogram must be 2D or 3D.")
    return sinogram

dataset_dir = "YOUR_DATA_PATH"  # replace with your data path
save_dir = "YOUR_SAVE_PATH"  # replace with your save path

save_dir_train = f"{save_dir}/train"
save_dir_test = f"{save_dir}/test"
os.makedirs(save_dir_train, exist_ok=True)
os.makedirs(save_dir_test, exist_ok=True)

save_dir_test_sinos = f"{save_dir}/test_sinos"
os.makedirs(save_dir_test_sinos, exist_ok=True)

slice_start = 5
slice_end = 665

############################test rock###############################
target = 'F3_2_mono'
darks = np.zeros((20, slice_end-slice_start, 1231-463), dtype=np.float32)
flats = np.zeros((20, slice_end-slice_start, 1231-463), dtype=np.float32)
for i in range(1,21):
    darks[i-1] = imread(f"{dataset_dir}/{target}/dark_{str(i).zfill(4)}.tif")[slice_start:slice_end,463:1231]
    flats[i-1] = imread(f"{dataset_dir}/{target}/flat_{str(i).zfill(4)}.tif")[slice_start:slice_end,463:1231]


dark_median = np.sort(darks,axis=0)[10]
white_median = np.sort(flats,axis=0)[10]


projs = np.zeros((1200, slice_end-slice_start, 1231-463), dtype=np.float32)
for i in range(1,1201):
    projs[i-1] = imread(f"{dataset_dir}/{target}/tomo_{str(i).zfill(4)}.tif")[slice_start:slice_end,463:1231]

projs = (projs - dark_median) / (white_median - dark_median)
projs[projs<=0] = projs[projs>0].min()
projs = -np.log(projs)

num_angles = projs.shape[0]

projs = np.swapaxes(projs,0,1)

angles = np.linspace(0, np.pi, num_angles,endpoint=False)

vol_geom = astra.create_vol_geom(768, 768, 1)
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 1, 768, angles)
proj_geom = astra.functions.geom_postalignment(proj_geom, -40.5)

A = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)

recon = np.zeros((projs.shape[0], 768, 768), dtype=np.float32)

for i in tqdm(range(recon.shape[0])):
    proj = torch.tensor(projs[i], dtype=torch.float32, device='cuda')
    proj = proj.unsqueeze(0)
    recon_fbp = fbp(A, proj)
    recon[i] = recon_fbp.detach().squeeze().cpu().numpy()

recon_max, recon_min = recon.max(), recon.min()
# Step 2: Compute scaling factor for projections, so that reconstruction is rougly in the range [0, 1]
scale_factor = 2 / (recon_max - recon_min)
projs *= scale_factor  # Apply scaling to projections
projs *= 1.5
# reduce ring artifacts after scaling
projs_corr = rivers_filter(projs, median_filter_size=11)

# reconstruct again the scaled projections, the reconstruction should be roguhly in the range [-1, 1]
# note that the sinogram is ring corrected
for i in tqdm(range(recon.shape[0])):
    proj = torch.tensor(projs_corr[i], dtype=torch.float32, device='cuda')
    proj = proj.unsqueeze(0)
    recon_fbp = fbp(A, proj)
    recon[i] = recon_fbp.detach().squeeze().cpu().numpy()

# save the reconstructions (ring corrected) and the sinograms (scaled pre ring correction)
for i in tqdm(range(recon.shape[0])):
    imwrite(f"{save_dir_test}/{str(i).zfill(3)}.tif", recon[i].astype(np.float32))

test_p05, test_p95 = np.percentile(recon, [5, 95])

############################test sinogram###########################
# test sinogram only use the first flat/dark field instead of the median
dark_first = darks[0]
white_first = flats[0]

# reload original projections
projs = np.zeros((1200, slice_end-slice_start, 1231-463), dtype=np.float32)
for i in range(1,1201):
    projs[i-1] = imread(f"{dataset_dir}/{target}/tomo_{str(i).zfill(4)}.tif")[slice_start:slice_end,463:1231]

projs = (projs - dark_first) / (white_first - dark_first)
projs[projs<=0] = projs[projs>0].min()
projs = -np.log(projs)

num_angles = projs.shape[0]

projs = np.swapaxes(projs,0,1)

for i in tqdm(range(recon.shape[0])):
    proj = torch.tensor(projs[i], dtype=torch.float32, device='cuda')
    proj = proj.unsqueeze(0)
    recon_fbp = fbp(A, proj)
    recon[i] = recon_fbp.detach().squeeze().cpu().numpy()

recon_max, recon_min = recon.max(), recon.min()
scale_factor = 2 / (recon_max - recon_min)
projs *= scale_factor  # Apply scaling to projections
projs *= 1.5

for i in tqdm(range(projs.shape[0])):
    imwrite(f"{save_dir_test_sinos}/{str(i).zfill(3)}.tif", projs[i].astype(np.float32))

############################train rock###############################
target = 'F3_1_mono'
darks = np.zeros((20, slice_end-slice_start, 1231-463), dtype=np.float32)
flats = np.zeros((20, slice_end-slice_start, 1231-463), dtype=np.float32)
for i in range(1,21):
    darks[i-1] = imread(f"{dataset_dir}/{target}/dark_{str(i).zfill(4)}.tif")[slice_start:slice_end,463:1231]
    flats[i-1] = imread(f"{dataset_dir}/{target}/flat_{str(i).zfill(4)}.tif")[slice_start:slice_end,463:1231]

dark_median = np.sort(darks,axis=0)[10]
white_median = np.sort(flats,axis=0)[10]

projs = np.zeros((1200, slice_end-slice_start, 1231-463), dtype=np.float32)
for i in range(1,1201):
    projs[i-1] = imread(f"{dataset_dir}/{target}/tomo_{str(i).zfill(4)}.tif")[slice_start:slice_end,463:1231]

projs = (projs - dark_median) / (white_median - dark_median)
projs[projs<=0] = projs[projs>0].min()
projs = -np.log(projs)

num_angles = projs.shape[0]

projs = np.swapaxes(projs,0,1)

projs_corr = rivers_filter(projs, median_filter_size=11)


angles = np.linspace(0, np.pi, num_angles,endpoint=False)

vol_geom = astra.create_vol_geom(768, 768, 1)
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 1, 768, angles)
proj_geom = astra.functions.geom_postalignment(proj_geom, -41.5)

A = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)

recon = np.zeros((projs.shape[0], 768, 768), dtype=np.float32)

for i in tqdm(range(recon.shape[0])):
    proj = torch.tensor(projs_corr[i], dtype=torch.float32, device='cuda')
    proj = proj.unsqueeze(0)
    recon_fbp = fbp(A, proj)
    recon[i] = recon_fbp.detach().squeeze().cpu().numpy()

# roughly scale to match the test rock reconstruction
train_p05, train_p95 = np.percentile(recon, [5, 95])
recon = (recon - train_p05) / (train_p95 - train_p05)
recon = recon * (test_p95 - test_p05) + test_p05

# save the reconstructions (ring corrected) and the sinograms (scaled pre ring correction)
for i in tqdm(range(recon.shape[0])):
    imwrite(f"{save_dir_train}/{str(i).zfill(3)}.tif", recon[i].astype(np.float32))

