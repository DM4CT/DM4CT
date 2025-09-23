import os
import astra
import torch
from torchvision import transforms
from forward_operators_ct import Operator, NoNoise
from pipelines import DDPMPipelineDPS, DDPMPipelineMCG, DDPMPipelinePGDM, DDPMPipelineDMPlug, DDPMPipelineRedDiff, DDPMPipelineHybridReg
from condition_methods import PosteriorSampling, ManifoldConstraintGradient, PGDM, DMPlug, RedDiff, HybridReg
import numpy as np
from datasets import TiffDataset
from tifffile import imwrite
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset_name = "lodochallenge"
data_path = f"YOUR_DATA_PATH"  # replace with your data path
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

unet = UNet2DModel.from_pretrained(unet_path).to('cuda')

import sys
idx = int(sys.argv[1])

if idx == 0:
    # Use a standard DDPM scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # # Define the pipeline
    pipeline = DDPMPipelineDPS(unet=unet, scheduler=scheduler)
    conditioning_method = PosteriorSampling(operator=A, noiser=noiser, scale=10)
    measurement_condition = conditioning_method
    pipeline.measurement_condition = measurement_condition

    save_dir = f"dps/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')

        # forward pass
        y = A(img)
        y_n = noiser(y)

        img = pipeline(num_inference_steps=1000, measurement=y_n, output_type=np.array).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)

elif idx==1:
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Define the pipeline
    pipeline = DDPMPipelineMCG(unet=unet, scheduler=scheduler)
    conditioning_method = ManifoldConstraintGradient(operator=A, noiser=noiser, scale=2)
    pipeline.measurement_condition = conditioning_method

    save_dir = f"mcg/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')

        # forward pass
        y = A(img)
        y_n = noiser(y)

        img = pipeline(num_inference_steps=1000, measurement=y_n, output_type=np.array).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)

elif idx==2:
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    # Define the pipeline
    pipeline = DDPMPipelinePGDM(unet=unet, scheduler=scheduler)
    conditioning_method = PGDM(operator=A, noiser=noiser)
    pipeline.measurement_condition = conditioning_method

    save_dir = f"pgdm/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')

        # forward pass
        y = A(img)
        y_n = noiser(y)

        img = pipeline(num_inference_steps=100, measurement=y_n, output_type=np.array, scale=0.3).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)

elif idx==3:
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    # Define the pipeline
    pipeline = DDPMPipelineDMPlug(unet=unet, scheduler=scheduler)
    conditioning_method = DMPlug(operator=A, noiser=noiser)
    pipeline.measurement_condition = conditioning_method

    save_dir = f"dmplug/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')
        # forward pass
        y = A(img)
        y_n = noiser(y)
        img = pipeline(num_inference_steps=3, measurement=y_n, epochs=1000, lr=0.005, output_type=np.array).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)

elif idx==4:
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    # Define the pipeline
    dps_pipeline = DDPMPipelineRedDiff(unet=unet, scheduler=scheduler)
    conditioning_method = RedDiff(operator=A, noiser=noiser)
    dps_pipeline.measurement_condition = conditioning_method

    save_dir = f"reddiff/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')
        # forward pass
        y = A(img)
        y_n = noiser(y)
        img = dps_pipeline(num_inference_steps=200, measurement=y_n, sigma=1e-4, loss_measurement_weight=0.5,
                            loss_noise_weight=10000, lr=0.099, output_type=np.array).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)

elif idx==5:
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    # Define the pipeline
    dps_pipeline = DDPMPipelineHybridReg(unet=unet, scheduler=scheduler)
    conditioning_method = HybridReg(operator=A, noiser=noiser)
    dps_pipeline.measurement_condition = conditioning_method

    save_dir = f"hybridreg/{dataset_name}/{num_angles}projs"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(len(dataset)):
        img = dataset[i]
        img = torch.tensor(img, device='cuda')
        # forward pass
        y = A(img)
        y_n = noiser(y)
        img = dps_pipeline(num_inference_steps=200, measurement=y_n, sigma=1e-4, loss_measurement_weight=0.5,
                            loss_noise_weight=10000, lr=0.099, beta=0.9999, output_type=np.array).images[0]
        imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)


