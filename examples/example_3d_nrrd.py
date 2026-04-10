"""Example: Create and save a 3D NRRD image with random data."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create random 3D image
array_3d = np.random.rand(100, 100, 80).astype(np.float32)

# Define spacing, origin, and direction
spacing = [2.5, 2.5, 3.0]
origin = [-125.0, -125.0, -120.0]
direction = np.eye(3)

# Save as NRRD
mv = MedVol(array_3d, spacing=spacing, origin=origin, direction=direction)
mv.save(script_dir / "data" / "3d_img.nrrd")
print("Saved 3D NRRD image")

# Load and verify
mv_loaded = MedVol(script_dir / "data" / "3d_img.nrrd")
print(f"Loaded: shape={mv_loaded.array.shape}, backend={mv_loaded.backend}")
print(f"Spacing: {mv_loaded.spacing}")
