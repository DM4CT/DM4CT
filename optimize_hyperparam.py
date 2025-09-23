import torch
import astra
import numpy as np
from scipy.optimize import minimize
from torchvision import datasets, transforms
from datasets import TiffDataset
from tifffile import imwrite
from forward_operators_ct import Operator, PoissonNoise, NoNoise, PoissonNoiseRing
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, VQModel
from pipelines import DDPMPipelineDPS, DDPMPipelineMCG, DDPMPipelinePGDM, DDPMPipelineDMPlug, DDPMPipelineRedDiff, DDPMPipelineHybridReg
from condition_methods import PosteriorSampling, ManifoldConstraintGradient, PGDM, DMPlug, RedDiff, HybridReg
from pipelines import LDMPipelinePSLD, LDMPipelineReSample, LDMPipelineDiffStateGrad
from condition_methods import PSLD, Resample, DiffStateGrad
from schedulers import DDIMSchedulerReSampler

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (0,1)
])

data_path = "YOUR_DATASET_PATH"  # Replace with your dataset path
img_idx = 0

# Create dataset
dataset = TiffDataset(data_path, transform=transform)

# Get an image
img = dataset[img_idx]
img = torch.tensor(img, device='cuda')

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

# forward pass
y = A(img)
y_n = noiser(y)


# Define objective function for optimization
def objective_function(params, pipeline_class, param_names, num_images=1):
    """
    Optimize the given pipeline's hyperparameters using multiple images.
    
    params: List of hyperparameter values
    pipeline_class: The diffusion pipeline class
    param_names: Names of the hyperparameters
    num_images: Number of images to use for optimization
    """
    # Convert params into a dictionary
    param_dict = {name: value for name, value in zip(param_names, params)}

    if 'DDPM' in pipeline_class.__name__:   
        unet = UNet2DModel.from_pretrained("trained/ddpm/unet").to('cuda')
    elif 'LDM' in pipeline_class.__name__:   
        unet = UNet2DModel.from_pretrained("trained/latent/unet").to('cuda')
        vqvae = VQModel.from_pretrained("trained/latent/vqvae").to('cuda')
    else:
        raise ValueError(f"Unknown pipeline class")
    
    # Create conditioning method (adjust for different pipelines)
    if issubclass(pipeline_class, DDPMPipelineDPS):
        conditioning_method = PosteriorSampling(operator=A, noiser=noiser, scale=param_dict["scale"]**2)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, DDPMPipelineMCG):
        conditioning_method = ManifoldConstraintGradient(operator=A, noiser=noiser, scale=param_dict["scale"]**2)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, DDPMPipelinePGDM):
        conditioning_method = PGDM(operator=A, noiser=noiser)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, DDPMPipelineDMPlug):
        conditioning_method = DMPlug(operator=A, noiser=noiser)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, DDPMPipelineRedDiff):
        conditioning_method = RedDiff(operator=A, noiser=noiser)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, DDPMPipelineHybridReg):
        conditioning_method = HybridReg(operator=A, noiser=noiser)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, LDMPipelinePSLD):
        conditioning_method = PSLD(operator=A, noiser=noiser)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, LDMPipelineReSample):
        conditioning_method = Resample(operator=A, noiser=noiser)
        scheduler = DDIMSchedulerReSampler(num_train_timesteps=1000)
    elif issubclass(pipeline_class, LDMPipelineDiffStateGrad):
        conditioning_method = DiffStateGrad(operator=A, noiser=noiser)
        scheduler = DDIMSchedulerReSampler(num_train_timesteps=1000)
    else:
        raise ValueError(f"Unknown pipeline class")

    # Initialize pipeline
    if 'DDPM' in pipeline_class.__name__: 
        pipeline = pipeline_class(unet=unet, scheduler=scheduler)
    elif 'LDM' in pipeline_class.__name__:   
        pipeline = pipeline_class(unet=unet, scheduler=scheduler, vqvae=vqvae)
    else:
        raise ValueError(f"Unknown pipeline class")
    pipeline.measurement_condition = conditioning_method

    # Evaluate over multiple images
    loss = 0.0
    for i in range(num_images):
        img = torch.tensor(dataset[i], device='cuda')
        y = A(img)
        y_n = noiser(y)
        if issubclass(pipeline_class, DDPMPipelinePGDM):
            reconstructed = pipeline(num_inference_steps=100, measurement=y_n, output_type=np.array, scale=param_dict["scale"]**2).images[0]
        elif issubclass(pipeline_class, DDPMPipelineDMPlug):
            reconstructed = pipeline(num_inference_steps=3, measurement=y_n, output_type=np.array, lr=param_dict["lr"]).images[0]
        elif issubclass(pipeline_class, DDPMPipelineRedDiff):
            reconstructed = pipeline(num_inference_steps=100, measurement=y_n, output_type=np.array, 
            sigma=param_dict["sigma"], loss_measurement_weight=1/(1+np.exp(-param_dict["loss_measurement_weight"])),
            loss_noise_weight=1-1/(1+np.exp(-param_dict["loss_measurement_weight"])), lr=param_dict["lr"]**2).images[0]
        elif issubclass(pipeline_class, DDPMPipelineHybridReg):
            reconstructed = pipeline(num_inference_steps=100, measurement=y_n, output_type=np.array, 
            sigma=param_dict["sigma"], loss_measurement_weight=1/(1+np.exp(-param_dict["loss_measurement_weight"])),
            loss_noise_weight=1-1/(1+np.exp(-param_dict["loss_measurement_weight"])), lr=param_dict["lr"]**2,
            beta=1/(1+np.exp(-param_dict["beta"]))).images[0]
        elif issubclass(pipeline_class, LDMPipelinePSLD):
            reconstructed = pipeline(num_inference_steps=200, measurement=y_n, output_type=np.array, gamma=1/(1+np.exp(-param_dict['gamma'])), 
                                     omega=1-1/(1+np.exp(-param_dict['gamma']))).images[0]
        elif issubclass(pipeline_class, LDMPipelineReSample):
            reconstructed = pipeline(num_inference_steps=200, measurement=y_n, output_type=np.array, inter_timesteps=5,
                                     pixel_lr=param_dict["pixel_lr"]**2,latent_lr=param_dict["latent_lr"]**2).images[0]
        else:
            reconstructed = pipeline(num_inference_steps=1000, measurement=y_n, output_type=np.array).images[0]

        # Compute loss (e.g., MSE)
        loss += np.mean((reconstructed - img.cpu().numpy()) ** 2)

    return loss / num_images  # Average loss over images

# Define optimization function
def optimize_hyperparams(pipeline_class, param_names, initial_values, num_images=1):
    """
    Optimize hyperparameters for a given pipeline.
    
    pipeline_class: Diffusion pipeline class
    param_names: List of hyperparameter names
    initial_values: Initial values for optimization
    num_images: Number of images to use for optimization
    """
    result = minimize(
        objective_function,
        initial_values,
        args=(pipeline_class, param_names, num_images),
        method="Nelder-Mead",
        # options={"maxiter": 50}
    )

    # Save results to a text file named after the pipeline
    pipeline_name = pipeline_class.__name__
    filename = f"{pipeline_name}_optimized.txt"
    with open(filename, "w") as f:
        for name, value in zip(param_names, result.x):
            if 'beta' in name:
                f.write(f"{name}: {1/(1+np.exp(-value))}\n")
            elif 'scale' in name:
                f.write(f"{name}: {value**2}\n")
            elif 'gamma' in name: 
                f.write(f"{name}: {1/(1+np.exp(-value))}\n")
            elif 'loss_measurement_weight' in name:
                f.write(f"{name}: {1/(1+np.exp(-value))}\n")
            elif 'inter_timesteps' in name:
                f.write(f"{name}: {np.abs(value)}\n")
            elif 'lr' in name:
                f.write(f"{name}: {value**2}\n")
            else:
                f.write(f"{name}: {value}\n")

    print(f"Optimization results saved to {filename}")

    return result

import sys
idx = int(sys.argv[1])

if idx==0:
    opt_result = optimize_hyperparams(
        pipeline_class=DDPMPipelineDPS,
        param_names=["scale"],
        initial_values=[np.sqrt(10)],  # Initial guess
        num_images=1
    )
elif idx==1:
    opt_result = optimize_hyperparams(
        pipeline_class=DDPMPipelineMCG,
        param_names=["scale"],
        initial_values=[np.sqrt(10)],  # Initial guess
        num_images=1
    )
elif idx==2:
    opt_result = optimize_hyperparams(
        pipeline_class=DDPMPipelinePGDM,
        param_names=["scale"],
        initial_values=[np.sqrt(10)],  # Initial guess
        num_images=1
    )
elif idx==3:
    opt_result = optimize_hyperparams(
        pipeline_class=DDPMPipelineDMPlug,
        param_names=["lr"],
        initial_values=[0.1],  # Initial guess
        num_images=1
    )
elif idx==4:
    opt_result = optimize_hyperparams(
        pipeline_class=DDPMPipelineRedDiff,
        param_names=["sigma", "loss_measurement_weight",  "lr"],
        initial_values=[0, 0, 0.1],  # Initial guess
        num_images=1
    )
elif idx==5:
    opt_result = optimize_hyperparams(
        pipeline_class=DDPMPipelineHybridReg,
        param_names=["sigma", "loss_measurement_weight",  "beta","lr"],
        initial_values=[0, 0, 0.97, 0.2, 0.1],  # Initial guess
        num_images=1
    )

elif idx==6:
    opt_result = optimize_hyperparams(
        pipeline_class=LDMPipelinePSLD,
        param_names=["gamma"],
        initial_values=[0],  # Initial guess
        num_images=1
    )


elif idx==7:
    opt_result = optimize_hyperparams(
        pipeline_class=LDMPipelineReSample,
        param_names=["pixel_lr", "latent_lr"],
        initial_values=[1e-1,1e-2],  # Initial guess
        num_images=1
    )

