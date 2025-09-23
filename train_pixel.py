from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tifffile import imread, imwrite
import torch.nn.functional as F
from tqdm import tqdm
from datasets import TiffDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = "lodochallenge"
data_path = f"YOUR_DATA_PATH"  # replace with your data path
unet_path = f"YOUR_UNET_PATH"  # replace with your UNET path

os.makedirs(unet_path, exist_ok=True)

# Checkpoint path
checkpoint_path = os.path.join(unet_path, "checkpoint.pth")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create dataset
dataset = TiffDataset(data_path, transform=transform)

batch_size = 1
# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def save_diffusion_sample(epoch, pipeline, save_dir="diffusion_uncond_samples"):
    os.makedirs(save_dir, exist_ok=True)
    
    pipeline.unet.eval()  # Set UNet to eval mode
    with torch.no_grad():
        generated_image = pipeline(num_inference_steps=1000, output_type=np.array).images[0]
    
    pipeline.unet.train()  # Set UNet back to training mode

    # Save the image
    imwrite(f"{save_dir}/epoch_{epoch+1}.tif", generated_image)
    print(f"Saved generated sample at epoch {epoch+1}")

# Define UNet with 1-channel input/output
unet = UNet2DModel(
    sample_size=512,  # Adjust based on your dataset
    in_channels=1,  # Grayscale input
    out_channels=1,  # Grayscale output
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
)
unet = unet.to(device)

# Use a standard DDPM scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Define pipeline
pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)

# Define optimizer
optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

num_epochs = 200

# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, path="trained/checkpoint.pth"):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch+1}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, path="trained/checkpoint.pth"):
    global start_epoch  # Modify start_epoch variable
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

# Load checkpoint if available
load_checkpoint(unet, optimizer, checkpoint_path)

for epoch in range(start_epoch, num_epochs):
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        # Get image tensor and noise
        images = batch.to(device)  
        noise = torch.randn_like(images)  # Gaussian noise
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()

        # Add noise to images based on timesteps
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # Predict noise using UNet
        noise_pred = unet(noisy_images, timesteps).sample  # Output: [B, 1, 64, 64]

        # Loss: Mean Squared Error between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save checkpoint after each epoch
    save_checkpoint(epoch, unet, optimizer, checkpoint_path)

    # Save generated sample every 5 epochs
    if epoch==0 or (epoch + 1) % 5 == 0:
        save_diffusion_sample(epoch, pipeline)

# Save model
unet.save_pretrained(unet_path)


