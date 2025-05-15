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
from tomobar.supp.suppTools import normaliser
from tomobar.methodsIR import RecToolsIR

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

save_dir = f"mbir/{dataset_name}/{num_angles}projs/"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

for i in tqdm(range(len(dataset))):    
    img = dataset[i]
    img = torch.tensor(img, device='cuda')
    img = img.to(device)
    
    # forward pass
    y = A(img)
    y_n = noiser(y)


    Rectools = RecToolsIR(
    DetectorsDimH=n_cols,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension
    CenterRotOffset=0,#-42,  # Center of Rotation (needs to be found)
    AnglesVec=angles,  # A vector of projection angles in radians
    ObjSize=n_cols,  # The reconstructed object dimensions
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS and SWLS
    device_projector="gpu", # Device to perform reconstruction on
    )

    _data_ = {"projection_norm_data": y_n.cpu().numpy()[0],
        "data_axes_labels_order": ["angles", "detX"]}  
    
    lc = Rectools.powermethod(_data_) # calculate Lipschitz constant

    _regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 1e-3         ,  # Regularisation parameter
    "iterations": 100,  # The number of inner regularisation iterations
    "device_regulariser": "gpu"}

    _algorithm_ = {"iterations": 200, "lipschitz_const": lc, "recon_mask_radius":None} 

    admm_recon = Rectools.ADMM(_data_, _algorithm_, _regularisation_)
    admm_recon = admm_recon[::-1, :]

    os.makedirs(os.path.join(save_dir, 'admm_pdtv'), exist_ok=True)  # Create the directory if it doesn't exist

    imwrite(os.path.join(save_dir, "admm_pdtv", f"{str(i).zfill(3)}.tif"), admm_recon)

    _regularisation_ = {
    "method": "SB_TV",  # Selected regularisation method
    "regul_param": 1e-3,
    "iterations":10,
    "device_regulariser": "gpu"}

    _algorithm_ = {"iterations": 200, "lipschitz_const": lc, "recon_mask_radius":None} 
    # _data_.update({"ringGH_lambda": 1e-4})

    fista_recon = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
    fista_recon = fista_recon[::-1, :]

    os.makedirs(os.path.join(save_dir, 'fista_sbtv'), exist_ok=True)  # Create the directory if it doesn't exist

    imwrite(os.path.join(save_dir, "fista_sbtv", f"{str(i).zfill(3)}.tif"), fista_recon)





