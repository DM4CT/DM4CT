# DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction

## Setup
Create and activate the main Conda environment:

```bash
conda env create -f environment.yml
conda activate diffusers-ct
```

For MBIR reconstruction (due to version conflicts), use the separate environment:
```bash
conda env create -f mbir.yml
conda activate mbir
```

## Datasets
### Simulation Datasets
We benchmark diffusion models on:
* The medical CT dataset: [Low Dose Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/)
* The industrial CT dataset: [LoDoInd](https://zenodo.org/records/10391412)

You can access these datasets through the provided links. Preprocessing scripts are included to enable reproducibility.

### Preprocessing
Use the provided scripts to:
* Perform train/test split
* Rescale intensity values to the range (-1, 1)

Script:
* [preprocess_lodochallenge.py](https://github.com/DM4CT/DM4CT/blob/main/preprocess_lodochallenge.py)
* [preprocess_lodoind.py](https://github.com/DM4CT/DM4CT/blob/main/preprocess_lodoind.py)


### Real-world Synchrotron CT Dataset
We also acquired a synchrotron CT dataset of two rock samples, available at: [Zenodo: Rocks Dataset](https://zenodo.org/records/15420527).

### Preprocessing

* To perform flat-field correction, reduce ring artifacts, and reconstruct the whole volume (flat-field correction->log transformation->ring reduction->reconstruction): [reconstruct_rocks_fbp.py](https://github.com/DM4CT/DM4CT/blob/main/reconstruct_rocks_fbp.py)
* To prepare the rocks dataset for diffusion-based reconstruction (cropping->flat-field correction->log transformation->ring reduction->reconstruction->roughly align value ranges): [preprocess_rocks.py](https://github.com/DM4CT/DM4CT/blob/main/preprocess_rocks.py)


## Pretrained diffusion models

All pretrained models are available [here](https://drive.google.com/drive/folders/1lqbzcQWxfkc1m1MqrSAJN-lyNWsASHx0?usp=sharing)

If you only want to download specific model

lodochallenge:
* [pixel diffusion model](https://drive.google.com/drive/folders/1IQ2ep7n9ARdq_53f7I8y1IsEaPN_Abjx?usp=drive_link) 
* [latent diffusion model](https://drive.google.com/drive/folders/1uWzdoqol5g7vZh8j-WtqTEFN5hBywFSq?usp=drive_link)

lodoind:
* [pixel diffusion model](https://drive.google.com/drive/folders/1lCBPdKIutriMYmFt3Sw54Abgb_n880Xu?usp=drive_link) 
* [latent diffusion model](https://drive.google.com/drive/folders/1A4Xlydhi9u5uMVNf7Qm4naOj_MYC4aBo?usp=drive_link)

rocks:
* [pixel diffusion model](https://drive.google.com/drive/folders/1ktJATnBpgCtttTy1wAAVO6m7MlNplaBj?usp=drive_link)
* [latent diffusion model](https://drive.google.com/drive/folders/1OXMNoA2ty4mB7wnLucp6kMqkM_J9vW_e?usp=drive_link)



### CT Reconstruction Scripts

* Pixel diffusion model: [reverse_sample_pixel_ct.py](https://github.com/DM4CT/DM4CT/blob/main/reverse_sample_pixel_ct.py)
* Latent diffusion model: [reverse_sample_latent_ct.py](https://github.com/DM4CT/DM4CT/blob/main/reverse_sample_latent_ct.py)
* Unconditional generation: [uncond_gen.py](https://github.com/DM4CT/DM4CT/blob/main/uncond_gen.py)


## Other baselines

* INR reconstruction [reverse_sample_ct_inr.py](https://github.com/DM4CT/DM4CT/blob/main/reverse_sample_ct_inr.py)
* DIP reconstruction [reverse_sample_ct_dip.py](https://github.com/DM4CT/DM4CT/blob/main/reverse_sample_ct_dip.py)
* classical CT reconstruction (FBP and SIRT) [reverse_sample_ct_classical.py](https://github.com/DM4CT/DM4CT/blob/main/reverse_sample_ct_classical.py)
* MBIR CT reconstruction (ADMM-PDTV and FISTA-SBTV) [reverse_sample_ct_mbir.py](https://github.com/DM4CT/DM4CT/blob/main/reverse_sample_ct_mbir.py)  
> Note that mbir is a separate environment
* supervised learning with SwinIR [train_reverse_sample_ct_swinir.py](https://github.com/DM4CT/DM4CT/blob/main/train_reverse_sample_ct_swinir.py)

## Training Diffusion Models from Scratch
* pixel diffusion [train_pixel.py](https://github.com/DM4CT/DM4CT/blob/main/train_pixel.py)
* latent diffusion [train_latent.py](https://github.com/DM4CT/DM4CT/blob/main/train_latent.py)
