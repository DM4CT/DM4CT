import os
import astra
import torch
from tqdm import tqdm
from torchvision import transforms
from forward_operators_ct import Operator, PoissonNoise, NoNoise, PoissonNoiseRing
import numpy as np
from datasets import TiffDataset
from tifffile import imwrite
from utils_dip import Skip

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

save_dir = f"dip/{dataset_name}/{num_angles}projs/"
intermediate_dir = f"dip/{dataset_name}/{num_angles}projs/intermediate"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
os.makedirs(intermediate_dir, exist_ok=True)  # Create the directory if it doesn't exist


for i in range(len(dataset)):    
    img = dataset[i]
    img = torch.tensor(img, device='cuda')
    img = img.unsqueeze(0)  # Add batch and channel dimensions
    img = img.to(device)
    
    # forward pass
    y = A(img)
    y_n = noiser(y)

    model = Skip(in_ch=1, out_ch=1, skip_channels=(4, 4, 4, 4), channels=(8, 16, 32, 64))

    random_input = 0.1 * torch.randn((1,n_slices,n_rows,n_cols)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4 , betas=(0.9, 0.999), weight_decay=0)
    model = model.to(device)
    loss = torch.nn.MSELoss()

    epochs = 10000

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        train_output = model(random_input)  # [B, H, W, 3]

        train_projs = A(train_output)
        train_loss = loss(train_projs, y_n) 
        
        train_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            test_output = model(random_input)
            test_output = test_output.squeeze(-1).detach().cpu().numpy()
            imwrite(os.path.join(intermediate_dir,f"epoch_{str(epoch)}.tif"), test_output)


    model.eval()
    with torch.no_grad():
        test_output = model(random_input)

    test_output = test_output.squeeze(-1).detach().cpu().numpy()

    imwrite(os.path.join(save_dir,f"{str(i).zfill(3)}.tif"), test_output)
