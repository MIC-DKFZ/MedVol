"""Example: Using different backends (SimpleITK, nibabel, pynrrd)."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create 3D image with random data
array_3d = np.random.rand(80, 80, 60).astype(np.float32)

# Save using nibabel (default for .nii.gz)
mv = MedVol(array_3d)
mv.save(script_dir / "data" / "backend_test.nii.gz")
print("Saved with nibabel (default for .nii.gz)")

# Load with explicit backend override
print("\nLoading with different backends:")

mv_nib = MedVol(script_dir / "data" / "backend_test.nii.gz", backend="nibabel")
print(f"  nibabel: shape={mv_nib.array.shape}, backend={mv_nib.backend}")

mv_simpleitk = MedVol(script_dir / "data" / "backend_test.nii.gz", backend="simpleitk")
print(f"  SimpleITK: shape={mv_simpleitk.array.shape}, backend={mv_simpleitk.backend}")

# NRRD uses pynrrd
mv_nrrd = MedVol(script_dir / "data" / "3d_img.nrrd")
print(f"  pynrrd (NRRD): shape={mv_nrrd.array.shape}, backend={mv_nrrd.backend}")
