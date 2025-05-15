from diffusers import UNet2DModel, LDMPipeline, DDPMScheduler, VQModel
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tifffile import imread
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import TiffDataset
import numpy as np
from tifffile import imwrite
from utils import save_checkpoint, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = "lodochallenge"
data_path = f"YOUR_DATA_PATH"  # replace with your data path
vae_path = f"YOUR_VAE_PATH"  # replace with your VAE path
unet_path = f"YOUR_UNET_PATH"  # replace with your UNET path

os.makedirs(vae_path, exist_ok=True)
os.makedirs(unet_path, exist_ok=True)



# Checkpoint paths
vae_checkpoint_path = os.path.join(vae_path, "vae_checkpoint.pth")
best_vae_checkpoint_path = os.path.join(vae_path, "best_vae_checkpoint.pth")
unet_checkpoint_path = os.path.join(unet_path, "unet_checkpoint.pth")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  
])

# Create dataset
dataset = TiffDataset(data_path, transform=transform)
batch_size = 2
training_dataset, validate_dataset = torch.utils.data.random_split(dataset, [(int)(len(dataset)*0.8), len(dataset) - (int)(len(dataset)*0.8)])
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

# ----------- 1️⃣ Train VAE (Autoencoder) -----------
vqvae = VQModel(sample_size=512,
                in_channels=1,
                out_channels=1,
                scaling_factor=1,
                down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
                up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
                block_out_channels=[128, 256, 512],
                layers_per_block=2,
                latent_channels=1,
                num_vq_embeddings=512).to(device)

vae_optimizer = optim.AdamW(vqvae.parameters(), lr=1e-4)


# Select a fixed image from the dataset to track reconstruction quality
fixed_image = dataset[0].unsqueeze(0).to(device) 

# ----------- Set Fixed Noise for Unconditional Sampling -----------
fixed_noise = torch.randn(1, 1, 128, 128).to(device)  # Shape: (batch_size, latent_channels, latent_height, latent_width)

# Function to save VAE reconstructions
def save_vae_reconstruction(epoch, model, image, save_dir="vae_reconstructions"):
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()  # Set to eval mode
    with torch.no_grad():
        latent = model.encode(image).latents
        recon_image = model.decode(latent).sample
    
    model.train()  # Set back to training mode

    # Convert tensors to images
    image = image.cpu().squeeze(0)  # Remove batch dimension
    recon_image = recon_image.cpu().squeeze(0)

    # Plot original and reconstructed images
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(image.permute(1, 2, 0), cmap="gray")  # Original
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_image.permute(1, 2, 0), cmap="gray")  # Reconstructed
    axes[1].set_title(f"Reconstruction (Epoch {epoch+1})")
    axes[1].axis("off")

    plt.savefig(f"{save_dir}/epoch_{epoch+1}.png", bbox_inches="tight")
    plt.close()

    print(f"Saved reconstruction at epoch {epoch+1}")

# ----------- Add Function to Save Generated Images -----------
def save_diffusion_sample(epoch, pipeline, noise, save_dir="diffusion_latent_uncond_samples"):
    os.makedirs(save_dir, exist_ok=True)
    
    pipeline.unet.eval()  # Set UNet to eval mode
    with torch.no_grad():
        # Generate an image from the same noise
        generated_image = pipeline(noise=noise, num_inference_steps=1000, output_type=np.array).images[0] 

    pipeline.unet.train()  # Set UNet back to training mode

    imwrite(f"{save_dir}/epoch_{epoch+1}.tif", generated_image)
    print(f"Saved generated sample at epoch {epoch+1}")

# Load checkpoint if available
start_epoch_vae, best_val_loss = load_checkpoint(vqvae, vae_optimizer, vae_checkpoint_path)

# Early stopping setup
early_stopping_patience = 10  # Stop if no improvement for 10 epochs
best_val_loss = float("inf")
stagnant_epochs = 0

num_epochs_vae = 100  # Adjust based on your dataset
for epoch in range(start_epoch_vae, num_epochs_vae):
    vqvae.train()
    train_loss = 0.0
    for batch in tqdm(training_dataloader):
        vae_optimizer.zero_grad()

        images = batch.to(device)

        # Encode & Decode
        latents = vqvae.encode(images).latents
        out = vqvae.decode(latents)
        recon_images, commit_loss = out.sample, out.commit_loss

        # Loss: Reconstruction loss
        loss = F.mse_loss(recon_images, images) + commit_loss

        # Backpropagation
        loss.backward()
        vae_optimizer.step()
        train_loss += loss.item()

    train_loss /= len(training_dataloader)

    # Validation step
    vqvae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in validate_dataloader:
            images = batch.to(device)
            latents = vqvae.encode(images).latents
            recon_images = vqvae.decode(latents).sample
            loss = F.mse_loss(recon_images, images)
            val_loss += loss.item()
    val_loss /= len(validate_dataloader)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stagnant_epochs = 0
        save_checkpoint(epoch, vqvae, vae_optimizer, best_val_loss, best_vae_checkpoint_path)
    else:
        stagnant_epochs += 1
    
    if stagnant_epochs >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    save_checkpoint(epoch, vqvae, vae_optimizer, best_val_loss, vae_checkpoint_path)

    # Save reconstruction every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_vae_reconstruction(epoch, vqvae, fixed_image)

# Load the best-performing VAE before saving
print("Loading best-performing VAE for final save...")
load_checkpoint(vqvae, vae_optimizer, best_vae_checkpoint_path)
vqvae.save_pretrained(vae_path)
print("Best VAE model saved successfully.")

# ----------- 2️⃣ Train Latent Diffusion Model -----------
# Load trained VAE
vqvae = VQModel.from_pretrained(vae_path).to(device)

dataset = TiffDataset(data_path, transform=transform)
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define UNet for latent diffusion
unet = UNet2DModel(
    sample_size=128,  # Latent space is smaller than original image
    in_channels=1,  # Latent channels
    out_channels=1,  # Latent channels
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)
pipeline = LDMPipeline(vqvae=vqvae, unet=unet, scheduler=scheduler)
pipeline(noise=fixed_noise, num_inference_steps=1000, output_type=np.array).images[0] 
unet_optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

# Load checkpoint if available
start_epoch_unet, _ = load_checkpoint(unet, unet_optimizer, unet_checkpoint_path)

num_epochs_unet = 200
for epoch in range(start_epoch_unet, num_epochs_unet):
    for batch in tqdm(dataloader):
        unet_optimizer.zero_grad()

        images = batch.to(device)

        # Convert to latent space
        with torch.no_grad():
            latents = vqvae.encode(images).latents

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps).sample

        # Loss: Mean Squared Error
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        loss.backward()
        unet_optimizer.step()

    print(f"UNet Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save UNet checkpoint
    save_checkpoint(epoch, unet, unet_optimizer, None, unet_checkpoint_path)

    # Save a generated image every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_diffusion_sample(epoch, pipeline, fixed_noise)

# Save trained UNet
unet.save_pretrained(unet_path)
