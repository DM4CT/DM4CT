import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tifffile import imwrite
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
from datasets import TiffDataset, PairedTiffDataset  # Custom dataset for (noisy, clean) pairs
from utils import save_checkpoint, load_checkpoint

# -------------------------------
# 🚀 Setup: Paths & Configurations
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"

data_path_low = "YOUR_DATA_PATH_LOW_DOSE"  # replace with your data path
data_path_high = "YOUR_DATA_PATH_HIGH_DOSE"  # replace with your data path
data_path_low_test = "YOUR_DATA_PATH_LOW_DOSE_TEST"  # replace with your data path
dataset_name = "lodochallenge"
num_projs = 40 

save_dir = f"swinir/{dataset_name}/{num_projs}projs"
os.makedirs(save_dir, exist_ok=True)

intermediate_dir = os.path.join(save_dir, "intermediate")
os.makedirs(intermediate_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, "swinir_checkpoint.pth")
best_checkpoint_path = os.path.join(save_dir, "swinir_best.pth")

# -------------------------------
# 📂 Load Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(), 
])

dataset = PairedTiffDataset(
    data_path_low,
    data_path_high,
    transform=transform
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -------------------------------
# 🏗️ Initialize Model
# -------------------------------
config = Swin2SRConfig(
    img_size=(512, 512),
    scale=1,  # No upscaling
    num_channels=1,  # Single-channel CT images
    num_channels_out=1,
    embed_dim=128,  # Reduce embedding dimension (default: 180)
    depths=[2, 2, 2, 2],  # Reduce number of blocks per stage (default: [6, 6, 6, 6])
    num_heads=[2, 2, 2, 2],  # Reduce number of attention heads per stage
    window_size=8,  # Reduce window size (default: 8, can increase if needed)
    mlp_ratio=2.0,  # Reduce MLP ratio (default: 4.0)
    upsampler=None
)

model = Swin2SRForImageSuperResolution(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# -------------------------------
# 🎯 Training Loop with Validation & Checkpointing
# -------------------------------
num_epochs = 200
best_val_loss = float("inf")
early_stopping_patience = 10  # Stop if no improvement in 10 epochs
stagnant_epochs = 0

# Load previous checkpoint if available
start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0

    # ---- Training ----
    for i, (img_lr, img_hr) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]")):
        img_lr, img_hr = img_lr.to(device), img_hr.to(device)

        img_sr = model(img_lr).reconstruction  # Predicted denoised image

        loss_val = criterion(img_sr, img_hr)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (img_lr, img_hr) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Validation]")):
            img_lr, img_hr = img_lr.to(device), img_hr.to(device)

            img_sr = model(img_lr).reconstruction  # Predicted denoised image

            loss_val = criterion(img_sr, img_hr)
            val_loss += loss_val.item()

            # Save intermediate results as a single figure
            if i < 5 and epoch % 10 == 0:  
                original_img = img_lr.squeeze().cpu().numpy()
                denoised_img = img_sr.squeeze().cpu().numpy()
                ground_truth = img_hr.squeeze().cpu().numpy()

                # Create a single figure with 3 images
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(original_img, cmap="gray")
                axes[0].set_title("Original Noisy")
                axes[0].axis("off")

                axes[1].imshow(denoised_img, cmap="gray")
                axes[1].set_title("Denoised (SwinIR)")
                axes[1].axis("off")

                axes[2].imshow(ground_truth, cmap="gray")
                axes[2].set_title("Ground Truth")
                axes[2].axis("off")

                # Save the figure
                fig.savefig(os.path.join(intermediate_dir, f"epoch_{epoch}_sample_{i}.png"))
                plt.close(fig)

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.6f}")

    # ---- Save Checkpoints ----
    save_checkpoint(epoch, model, optimizer,  avg_val_loss, checkpoint_path)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        stagnant_epochs = 0
        save_checkpoint(epoch, model, optimizer, avg_val_loss, best_checkpoint_path)
        print(f"New Best Model Saved (Epoch {epoch})")
    else:
        stagnant_epochs += 1

    # ---- Early Stopping ----
    if stagnant_epochs >= early_stopping_patience:
        print(f"Early Stopping Triggered at Epoch {epoch}")
        break


# ------------------------------
load_checkpoint(model, optimizer, best_checkpoint_path)
model.save_pretrained(save_dir)

model = Swin2SRForImageSuperResolution.from_pretrained(save_dir)
model = model.to(device)
test_dataset = TiffDataset(data_path_low_test, transform=transform)

for i in tqdm(range(len(test_dataset))):
    img = test_dataset[0]
    img = torch.tensor(img, device='cuda')
    img = img.unsqueeze(0)

    img = model(img).reconstruction
    img = img.detach().cpu().numpy()[0,0]
    imwrite(os.path.join(save_dir, f"{str(i).zfill(3)}.tif"),img)
