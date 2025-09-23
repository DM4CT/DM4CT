import os
import astra
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from forward_operators_ct import Operator, PoissonNoise, NoNoise, PoissonNoiseRing
from PIL import Image
import numpy as np
from datasets import TiffDataset
from tifffile import imwrite, imread
from ct_reconstruction import sirt, fbp


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda:0'

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(), 
])

data_path = "YOUR_DATA_PATH"  # replace with your data path
dataset_name = "lodochallenge"
# Create dataset
dataset = TiffDataset(data_path, transform=transform)

num_angles = 40
angles =np.linspace(0, np.pi, num_angles)

n_rows=512
n_cols=512
n_slices=1

vol_geom = astra.create_vol_geom(n_rows, n_cols, n_slices)
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 1, 512, angles)

A = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)
# choose the noiser, e.g. PoissonNoise(0.5, 5000), PoissonNoise(0.5, 10000), NoNoise(), PoissonNoiseRing(0.5, 10000, 0.05, 0.25, 123)
noiser = NoNoise()


vol_geom = astra.create_vol_geom(n_rows, n_cols, n_slices)
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, 1, n_rows, angles)
A = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)

save_dir_fbp = f"classical/{dataset_name}/{num_angles}projs/fbp"
os.makedirs(save_dir_fbp, exist_ok=True)  # Create the directory if it doesn't exist
save_dir_sirt = f"classical/{dataset_name}/{num_angles}projs/sirt"
os.makedirs(save_dir_sirt, exist_ok=True)  # Create the directory if it doesn't exist

for i in tqdm(range(len(dataset))):    
    img = dataset[i]
    img = torch.tensor(img, device='cuda')
    img = img.to(device)
    
    # forward pass
    y = A(img)
    y_n = noiser(y)

    recon_fbp = fbp(A, y_n)
    recon_fbp = recon_fbp.clamp(min=-1, max=1)
    recon_sirt = sirt(A, y_n, num_iterations=100)
    recon_sirt = recon_sirt.clamp(min=-1, max=1)
    recon_fbp = recon_fbp.squeeze().cpu().numpy()
    recon_sirt = recon_sirt.squeeze().cpu().numpy()
    imwrite(os.path.join(save_dir_fbp, f"recon_fbp_{str(i).zfill(3)}.tif"), recon_fbp)
    imwrite(os.path.join(save_dir_sirt, f"recon_sirt_{str(i).zfill(3)}.tif"), recon_sirt)