import os
from tifffile import imread
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm import tqdm


def compute_metrics(gt_dir, recon_dir):
    """Compute PSNR and SSIM for a folder of reconstructed images against ground truth."""
    psnrs, ssims = [], []

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".tif")])
    pred_files = sorted([f for f in os.listdir(recon_dir) if f.endswith(".tif")])

    if len(pred_files) > len(gt_files):
        raise ValueError(f"More predictions ({len(pred_files)}) than GT images ({len(gt_files)}). Check your data!")

    # Use only the subset of GT corresponding to predicted images
    gt_files = gt_files[:len(pred_files)]

    for gt_name, pred_name in zip(gt_files, pred_files):
        path_gt = os.path.join(gt_dir, gt_name)
        path_pred = os.path.join(recon_dir, pred_name)

        gt = imread(path_gt).astype(np.float32)
        pred = imread(path_pred).astype(np.float32)

        if pred.ndim == 4:
            pred = pred[0, 0, :, :]
        elif pred.ndim == 3:
            pred = pred[0, :, :]
        if gt.ndim == 3:
            gt = gt[0, :, :]
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {gt_name} and {pred_name}: {gt.shape} vs {pred.shape}")

        if 'rocks' in gt_dir:
            gt_min, gt_max = gt.min(), gt.max()
            gt_range = gt_max - gt_min  
        else:
            gt_range = 2

        psnr_val = psnr(gt, pred, data_range=gt_range)
        ssim_val = ssim(gt, pred, data_range=gt_range)

        psnrs.append(psnr_val)
        ssims.append(ssim_val)

    return np.mean(psnrs), np.mean(ssims)

def evaluate_all(base_dir, dataset, config, gt_root, output_txt="metrics.txt"):
    results = []
    gt_dir = os.path.join(gt_root, "test")

    for method in sorted(os.listdir(base_dir)):
        method_path = os.path.join(base_dir, method, dataset, config)
        if not os.path.isdir(method_path):
            continue

        subdirs = [d for d in os.listdir(method_path) if os.path.isdir(os.path.join(method_path, d))]

        if len(subdirs) > 0 and all(os.path.isdir(os.path.join(method_path, d)) for d in subdirs):
            # Layout: method/dataset/config/{fbp,sirt,...}
            for submethod in subdirs:
                recon_path = os.path.join(method_path, submethod)
                print(f"Evaluating: {method}/{dataset}/{config}/{submethod}")
                avg_psnr, avg_ssim = compute_metrics(gt_dir, recon_path)
                results.append((f"{method}/{submethod}", avg_psnr, avg_ssim))
        else:
            # Layout: method/dataset/config/
            print(f"Evaluating: {method}/{dataset}/{config}")
            avg_psnr, avg_ssim = compute_metrics(gt_dir, method_path)
            results.append((method, avg_psnr, avg_ssim))

    with open(output_txt, "w") as f:
        for method_name, p, s in results:
            f.write(f"{method_name}: PSNR = {p:.4f}, SSIM = {s:.4f}\n")
    print(f"\n Results saved to {output_txt}")

# === Parameters to Set ===
dataset = "rocks" # or "lodochallenge" or "lodoind"
config = "200projs" 
gt_root = f"YOUR_PATH_TO_GT"  # Path to the ground truth images
base_dir = "YOUR_PATH_TO_RECON"  # Path to the directory containing the reconstructions

# === Run Evaluation ===
evaluate_all(base_dir, dataset, config, gt_root, output_txt=f"metrics_{dataset.replace('/','_')}_{config}.txt")