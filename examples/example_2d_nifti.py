"""Example: Create and save a 2D NIfTI image with random data."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create random 2D image (e.g., single-slice MRI)
array_2d = np.random.rand(256, 256).astype(np.float32)

# Save as NIfTI
mv = MedVol(array_2d)
mv.save(script_dir / "data" / "2d_img.nii.gz")
print("Saved 2D NIfTI image")

# Load and verify
mv_loaded = MedVol(script_dir / "data" / "2d_img.nii.gz")
print(f"Loaded: shape={mv_loaded.array.shape}, backend={mv_loaded.backend}")
