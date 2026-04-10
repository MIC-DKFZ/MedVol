"""Example: Create and save a 4D NRRD image with random data."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create random 4D image
array_4d = np.random.rand(80, 80, 40, 5).astype(np.float32)

# Define spacing and origin
spacing = [2.5, 2.5, 3.0, 1.0]
origin = [-100.0, -100.0, -60.0, 0.0]

# Save as NRRD
mv = MedVol(array_4d, spacing=spacing, origin=origin)
mv.save(script_dir / "data" / "4d_img.nrrd")
print("Saved 4D NRRD image")

# Load and verify
mv_loaded = MedVol(script_dir / "data" / "4d_img.nrrd")
print(f"Loaded: shape={mv_loaded.array.shape}, backend={mv_loaded.backend}")
