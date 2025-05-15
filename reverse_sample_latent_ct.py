import os
import astra
import torch
from torchvision import transforms
from pipelines import LDMPipelinePSLD, LDMPipelineReSample, LDMPipelineDiffStateGrad
from condition_methods import PSLD, Resample, DiffStateGrad
import numpy as np
from datasets import TiffDataset
from tifffile import imwrite
from diffusers import UNet2DModel, DDPMScheduler, VQModel
from forward_operators_ct import Operator, PoissonNoise, NoNoise
from schedulers import DDIMSchedulerReSampler

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(), 
])

dataset_name = "lodochallenge"
data_path = f"YOUR_DATA_PATH"  # replace with your data path
vae_path = f"YOUR_VAE_PATH"  # replace with your VAE path
unet_path = f"YOUR_UNET_PATH"  # replace with your UNET path

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

# Load trained VAE
vqvae = VQModel.from_pretrained(vae_path).to('cuda')

unet = UNet2DModel.from_pretrained(unet_path).to('cuda')

import sys
idx = int(sys.argv[1])

if idx == 0:
    # Define the pipeline
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = LDMPipelinePSLD(vqvae=vqvae, unet=unet, scheduler=scheduler)
    measurement_condition = PSLD(A, noiser)
    pipeline.measurement_condition = measurement_condition

    save_dir = f"psld/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')

        # forward pass
        y = A(img)
        y_n = noiser(y)

        img = pipeline(num_inference_steps=1000, measurement=y_n, output_type=np.array, gamma=0.8,omega=0.2).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)

elif idx==1:
     # Define the pipeline
    scheduler = DDIMSchedulerReSampler(num_train_timesteps=1000)
    pipeline = LDMPipelineReSample(vqvae=vqvae, unet=unet, scheduler=scheduler)
    measurement_condition = Resample(A, noiser)
    pipeline.measurement_condition = measurement_condition

    save_dir = f"resample/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')

        # forward pass
        y = A(img)
        y_n = noiser(y)

        img = pipeline(num_inference_steps=200, measurement=y_n, output_type=np.array,pixel_lr=1e-4,
        latent_lr=0.1).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)

elif idx==2:
    # Define the pipeline
    scheduler = DDIMSchedulerReSampler(num_train_timesteps=1000)
    pipeline = LDMPipelineDiffStateGrad(vqvae=vqvae, unet=unet, scheduler=scheduler)
    measurement_condition = DiffStateGrad(A, noiser)
    pipeline.measurement_condition = measurement_condition

    save_dir = f"diffstategrad/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')

        # forward pass
        y = A(img)
        y_n = noiser(y)

        img = pipeline(num_inference_steps=200, measurement=y_n, output_type=np.array,pixel_lr=1e-4,
        latent_lr=0.1,var_cutoff=0.99).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)



