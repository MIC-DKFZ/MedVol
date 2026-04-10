"""Example: Loading without canonicalization (native geometry)."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Create 3D image with LPS+ orientation (DICOM convention)
array_3d = np.random.rand(64, 64, 32).astype(np.float32)

# LPS+ affine (SimpleITK/DICOM convention)
affine_lps = np.array([
    [-2.0, 0.0, 0.0, 64.0],
    [0.0, -2.0, 0.0, 64.0],
    [0.0, 0.0, 2.0, -32.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Save as NIfTI with LPS+ orientation
mv_lps = MedVol(array_3d, affine=affine_lps, coordinate_system="LPS+", canonicalize=False)
mv_lps.save(script_dir / "data" / "native_lps.nii.gz")
print("Saved LPS+ image (non-canonical)")

# Load with canonicalization (default - converts to RAS+)
mv_canonical = MedVol(script_dir / "data" / "native_lps.nii.gz")
print(f"\nCanonicalized (RAS+):")
print(f"  Coordinate system: {mv_canonical.coordinate_system}")
print(f"  Affine:\n{mv_canonical.affine}")

# Load without canonicalization (keeps native LPS+)
mv_native = MedVol(script_dir / "data" / "native_lps.nii.gz", canonicalize=False)
print(f"\nNative (LPS+, no canonicalization):")
print(f"  Coordinate system: {mv_native.coordinate_system}")
print(f"  Affine:\n{mv_native.affine}")
