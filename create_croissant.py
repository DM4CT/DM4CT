import mlcroissant as mlc
import json

# ---- Distribution: Files and how to access them ----
distribution = [
    mlc.FileObject(
        id="f3_1_raw",
        name="F3_1_mono.zip",
        description="Raw projections for rock sample F3_1_mono, including 1200 projections, 20 dark fields, and 20 flat fields.",
        content_url="https://zenodo.org/records/15420527/files/F3_1_mono.zip?download=1",
        encoding_formats=["application/zip"],
        sha256="f1e38c7e37095c0eaa5378f0495276a794f65382d8141af549b7d8e1837983e1"
    ),
    mlc.FileObject(
        id="f3_1_recon",
        name="F3_1_mono_recon_corr.zip",
        description="Reconstructed slices of F3_1_mono after log-transform, flat-field correction, and ring artifact reduction.",
        content_url="https://zenodo.org/records/15420527/files/F3_1_mono_recon_corr.zip?download=1",
        encoding_formats=["application/zip"],
        sha256="a00a00892bba131cadde6e7d682f4ad889dfadb1aeac782e85035e2a46b224eb"
    ),
    mlc.FileObject(
        id="f3_2_raw",
        name="F3_2_mono.zip",
        description="Raw projections for rock sample F3_2_mono, including 1200 projections, 20 dark fields, and 20 flat fields.",
        content_url="https://zenodo.org/records/15420527/files/F3_2_mono.zip?download=1",
        encoding_formats=["application/zip"],
        sha256="1cc1835b73b482dbec0f9c6c9a21efd4400e3569edf6ac5b87cb5ea14e411002"
    ),
    mlc.FileObject(
        id="f3_2_recon",
        name="F3_2_mono_recon_corr.zip",
        description="Reconstructed slices of F3_2_mono after log-transform, flat-field correction, and ring artifact reduction.",
        content_url="https://zenodo.org/records/15420527/files/F3_2_mono_recon_corr.zip?download=1",
        encoding_formats=["application/zip"],
        sha256="3aa9004b0941f4f32635857bc0e6a375a6d27012936e87f32293d9ed2cc0da76"
    )
]


# ---- Metadata ----
metadata = mlc.Metadata(
    name="CT Slice Dataset",
    description="""
        This dataset contains high-energy synchrotron CT data of two rock samples (F3_1_mono and F3_2_mono). For each sample, the following data is provided:

        - **Raw projections**: 1200 projection images, accompanied by 20 dark-field and flat-field measurements.
        - **Reconstructed slices**: Log-transformed and flat-field-corrected reconstructions, with additional ring artifact reduction applied.

        **Acquisition Parameters**:
        - Pixel size: 9.00 mm
        - Angular step: 0.15°
        - Sample-to-detector distance: 150 mm
        - Exposure time: 4 s
        - Number of projections: 1200
        - Energy: 24 keV
        - Filter: none
        - Center of rotation: −61 (F3_1), −62 (F3_2)
        - Ring removal level: 11 (F3_1), 11 (F3_2)
        - Projections were rotated 33° counter-clockwise before reconstruction

        This dataset is designed for benchmarking CT reconstruction algorithms under practical and realistic acquisition conditions.        """,
    url="https://zenodo.org/records/15420527",  # Replace with your DOI
    license="https://creativecommons.org/licenses/by/4.0/",
    distribution=distribution,
    # record_sets=record_sets
)


# ---- Save to file ----
with open("croissant.json", "w") as f:
    content = metadata.to_json()
    content = json.dumps(content, indent=2)
    print(content)
    f.write(content)
    f.write("\n")  # Terminate file with newline
