"""Example: Working with headers."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create 3D image
array_3d = np.random.rand(64, 64, 32).astype(np.float32)

# Save and load with nibabel
mv = MedVol(array_3d)
mv.save(script_dir / "data" / "header_test.nii.gz")

# Load and inspect header
mv_loaded = MedVol(script_dir / "data" / "header_test.nii.gz")
print(f"Backend: {mv_loaded.backend}")
print(f"Header type: {type(mv_loaded.header)}")
print(f"Header: {mv_loaded.header}")

# Modify header (backend-specific)
if mv_loaded.backend == "nibabel":
    # nibabel header
    mv_loaded.header['pixdim'][1:4] = [2.5, 2.5, 3.0]
    print("\nModified nibabel header pixdim:", mv_loaded.header['pixdim'])

# Save with custom header
mv_loaded.save(script_dir / "data" / "header_modified.nii.gz")
print("\nSaved with modified header")
