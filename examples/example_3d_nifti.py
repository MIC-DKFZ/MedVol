"""Example: Create and save a 3D NIfTI image with random data."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create random 3D image (e.g., CT/MRI volume)
array_3d = np.random.rand(128, 128, 64).astype(np.float32)

# Define affine matrix for 3mm isotropic voxels
affine = np.array([
    [3.0, 0.0, 0.0, -192.0],
    [0.0, 3.0, 0.0, -192.0],
    [0.0, 0.0, 3.0, -96.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Save as NIfTI
mv = MedVol(array_3d, affine=affine)
mv.save(script_dir / "data" / "3d_img.nii.gz")
print("Saved 3D NIfTI image")

# Load and verify
mv_loaded = MedVol(script_dir / "data" / "3d_img.nii.gz")
print(f"Loaded: shape={mv_loaded.array.shape}, backend={mv_loaded.backend}")
print(f"Spacing: {mv_loaded.spacing}")
print(f"Origin: {mv_loaded.origin}")
print(f"Coordinate system: {mv_loaded.coordinate_system}")
