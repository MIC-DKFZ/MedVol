"""Example: Create and save a 4D NIfTI image with random data."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create random 4D image (e.g., fMRI with 10 timepoints)
array_4d = np.random.rand(64, 64, 32, 10).astype(np.float32)

# Define affine matrix (5x5 for 4D data)
affine = np.array([
    [2.0, 0.0, 0.0, 0.0, -64.0],
    [0.0, 2.0, 0.0, 0.0, -64.0],
    [0.0, 0.0, 2.0, 0.0, -32.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
])

# Save as NIfTI
mv = MedVol(array_4d, affine=affine)
mv.save(script_dir / "data" / "4d_img.nii.gz")
print("Saved 4D NIfTI image")

# Load and verify
mv_loaded = MedVol(script_dir / "data" / "4d_img.nii.gz")
print(f"Loaded: shape={mv_loaded.array.shape}, backend={mv_loaded.backend}")
