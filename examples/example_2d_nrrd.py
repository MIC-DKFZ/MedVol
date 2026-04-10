"""Example: Create and save a 2D NRRD image with random data."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create random 2D image
array_2d = np.random.rand(512, 512).astype(np.float32)

# Save as NRRD
mv = MedVol(array_2d)
mv.save(script_dir / "data" / "2d_img.nrrd")
print("Saved 2D NRRD image")

# Load and verify
mv_loaded = MedVol(script_dir / "data" / "2d_img.nrrd")
print(f"Loaded: shape={mv_loaded.array.shape}, backend={mv_loaded.backend}")
