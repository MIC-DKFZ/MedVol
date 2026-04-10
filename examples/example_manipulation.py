"""Example: Direct array manipulation after loading."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Load existing image
mv = MedVol(script_dir / "data" / "3d_img.nii.gz")
print(f"Original shape: {mv.array.shape}")
print(f"Original range: [{mv.array.min():.3f}, {mv.array.max():.3f}]")

# Scale intensity
mv.array = mv.array * 2.0
print(f"\nAfter scaling by 2.0:")
print(f"New range: [{mv.array.min():.3f}, {mv.array.max():.3f}]")

# Crop (slice)
mv.array = mv.array[10:50, 10:50, 10:30]
print(f"\nAfter cropping [10:50, 10:50, 10:30]:")
print(f"New shape: {mv.array.shape}")

# Update spacing
mv.spacing = [3.0, 3.0, 3.0]
print(f"\nAfter updating spacing to [3.0, 3.0, 3.0]:")
print(f"New spacing: {mv.spacing}")

# Save modified volume
mv.save(script_dir / "data" / "modified.nii.gz")
print("\nSaved modified volume")
