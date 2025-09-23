import astra
import numpy as np
import os
import torch
from tqdm import tqdm
from ct_reconstruction import sirt, fbp
from forward_operators_ct import Operator
from tifffile import imwrite, imread
from scipy.signal import medfilt

# rock1 training rock2 testing

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

cropping_reconstruction = True
dataset_dir = "YOUR_DATA_PATH"  # replace with your data path

for target in ['F3_1_mono', 'F3_2_mono']:

    if target == 'F3_1_mono':
        off_axis_dist = -61
    elif target == 'F3_2_mono':
        off_axis_dist = -62
    else:
        raise ValueError("Invalid target. Choose 'F3_1_mono' or 'F3_2_mono'.")
    save_dir = f"{dataset_dir}/{target}_recon_corr"
    os.makedirs(save_dir, exist_ok=True)


    darks = np.zeros((20, 679, 1653), dtype=np.float32)
    flats = np.zeros((20, 679, 1653), dtype=np.float32)
    for i in range(1,21):
        darks[i-1] = imread(f"{dataset_dir}/{target}/dark_{str(i).zfill(4)}.tif")
        flats[i-1] = imread(f"{dataset_dir}/{target}/flat_{str(i).zfill(4)}.tif")


    dark_median = np.sort(darks, axis=0)[10]
    white_median = np.sort(flats, axis=0)[10]


    projs = np.zeros((1200, 679, 1653), dtype=np.float32)
    for i in range(1,1201):
        projs[i-1] = imread(f"{dataset_dir}/{target}/tomo_{str(i).zfill(4)}.tif")

    projs = (projs - dark_median) / (white_median - dark_median)
    projs[projs<=0] = np.nanmin(projs[projs>0]) #projs[projs>0].min()
    projs = -np.log(projs)
    projs = np.nan_to_num(projs, nan=0.0)

    num_angles = projs.shape[0]

    projs = np.swapaxes(projs,0,1)

    # river method for ring reduction
    projs = rivers_filter(projs, median_filter_size=11)

    angles = np.linspace(0, np.pi, num_angles,endpoint=False)

    vol_geom = astra.create_vol_geom(1653, 1653, 1)
    proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 1, 1653, angles)
    proj_geom = astra.functions.geom_postalignment(proj_geom, off_axis_dist)

    A = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)

    recon = np.zeros((projs.shape[0], 1653, 1653), dtype=np.float32)

    for i in tqdm(range(recon.shape[0])):
        proj = torch.tensor(projs[i], dtype=torch.float32, device='cuda')
        proj = proj.unsqueeze(0)
        recon_fbp = fbp(A, proj)
        recon[i] = recon_fbp.detach().squeeze().cpu().numpy()
        imwrite(os.path.join(save_dir, f"recon_{str(i).zfill(4)}.tif"), recon[i])
    
    if cropping_reconstruction: 
        # cropping and reconstruction
        if target == 'F3_1_mono':
            off_axis_dist = -40.5
        elif target == 'F3_2_mono':
            off_axis_dist = -41.5
        else:
            raise ValueError("Invalid target. Choose 'F3_1_mono' or 'F3_2_mono'.")
        
        save_dir = f"{dataset_dir}/{target}_crop_recon_corr"
        os.makedirs(save_dir, exist_ok=True)

        # cropping area [:,463:1231]

        darks = np.zeros((20, 679, 1231-463), dtype=np.float32)
        flats = np.zeros((20, 679, 1231-463), dtype=np.float32)
        for i in range(1,21):
            darks[i-1] = imread(f"{dataset_dir}/{target}/dark_{str(i).zfill(4)}.tif")[:,463:1231]
            flats[i-1] = imread(f"{dataset_dir}/{target}/flat_{str(i).zfill(4)}.tif")[:,463:1231]

        dark_median = np.sort(darks, axis=0)[10]
        white_median = np.sort(flats, axis=0)[10]

        projs = np.zeros((1200, 679, 1231-463), dtype=np.float32)
        for i in range(1,1201):
            projs[i-1] = imread(f"{dataset_dir}/{target}/tomo_{str(i).zfill(4)}.tif")[:,463:1231]

        projs = (projs - dark_median) / (white_median - dark_median)
        projs[projs<=0] = np.nanmin(projs[projs>0]) #projs[projs>0].min()
        projs = -np.log(projs)
        projs = np.nan_to_num(projs, nan=0.0)

        num_angles = projs.shape[0]

        projs = np.swapaxes(projs,0,1)

        # river method for ring reduction
        projs = rivers_filter(projs, median_filter_size=11)

        angles = np.linspace(0, np.pi, num_angles,endpoint=False)

        vol_geom = astra.create_vol_geom(1231-463, 1231-463, 1)
        proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 1, 1231-463, angles)
        proj_geom = astra.functions.geom_postalignment(proj_geom, off_axis_dist)

        A = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)

        recon = np.zeros((projs.shape[0], 1231-463, 1231-463), dtype=np.float32)

        for i in tqdm(range(recon.shape[0])):
            proj = torch.tensor(projs[i], dtype=torch.float32, device='cuda')
            proj = proj.unsqueeze(0)
            recon_fbp = fbp(A, proj)
            recon[i] = recon_fbp.detach().squeeze().cpu().numpy()
            imwrite(os.path.join(save_dir, f"recon_{str(i).zfill(4)}.tif"), recon[i])
