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
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, VQModel, DDPMPipeline, LDMPipeline

dataset_name = "lodochallenge"

typ = 'pixel'

assert typ in ['pixel', 'latent'], "typ must be either 'pixel' or 'latent'"

if typ == 'pixel':
    unet_path = f"trained/{dataset_name}/ddpm/unet" # or your pathto unet
    unet = UNet2DModel.from_pretrained(unet_path).to('cuda')

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)

    img = pipeline(num_inference_steps=1000, output_type=np.array).images[0]
    folder = f"pixel/{dataset_name}"
    os.makedirs(folder, exist_ok=True)  # Create the directory if it doesn't exist
    imwrite(f"{folder}/uncond_gen.tif", img)
else:
    vae_path = f"trained/{dataset_name}/latent/vqvae" # or your path to vae
    unet_path = f"trained/{dataset_name}/latent/unet" # or your path to unet
    vqvae = VQModel.from_pretrained(vae_path).to('cuda')
    unet = UNet2DModel.from_pretrained(unet_path).to('cuda')
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = LDMPipeline(vqvae=vqvae, unet=unet, scheduler=scheduler)

    img = pipeline(num_inference_steps=1000, output_type=np.array).images[0]
    folder = f"latent/{dataset_name}"
    os.makedirs(folder, exist_ok=True)
    imwrite(f"{folder}/uncond_gen.tif", img)

