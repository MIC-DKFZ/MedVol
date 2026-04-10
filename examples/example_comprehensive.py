"""Example: Comprehensive showcase of MedVol functionality."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

print("=" * 60)
print("MedVol Comprehensive Example")
print("=" * 60)

# 1. Create 3D volume with specific geometry
print("\n1. Creating 3D volume with non-isotropic voxels...")
array_3d = np.random.rand(100, 100, 60).astype(np.float32)
spacing = [1.5, 1.5, 2.5]
origin = [-75.0, -75.0, -75.0]
direction = np.eye(3)

mv = MedVol(array_3d, spacing=spacing, origin=origin, direction=direction)
print(f"   Shape: {mv.array.shape}")
print(f"   Spacing: {mv.spacing}")
print(f"   Origin: {mv.origin}")
print(f"   Coordinate system: {mv.coordinate_system}")
print(f"   Backend: {mv.backend}")

# 2. Save as NIfTI
print("\n2. Saving as NIfTI...")
mv.save(script_dir / "data" / "comprehensive.nii.gz")
print("   Saved to examples/data/comprehensive.nii.gz")

# 3. Reload and inspect
print("\n3. Reloading and inspecting...")
mv_loaded = MedVol(script_dir / "data" / "comprehensive.nii.gz")
print(f"   Backend: {mv_loaded.backend}")
print(f"   Shape: {mv_loaded.array.shape}")
print(f"   Coordinate system: {mv_loaded.coordinate_system}")

# 4. Convert to LPS+
print("\n4. Converting to LPS+ coordinate system...")
geometry = mv_loaded.get_geometry("LPS+")
print(f"   LPS+ affine:\n{geometry['affine']}")
print(f"   LPS+ spacing: {geometry['spacing']}")
print(f"   LPS+ origin: {geometry['origin']}")

# 5. Get array in LPS+
print("\n5. Getting array in LPS+ coordinate system...")
array_lps = mv_loaded.get_array("LPS+")
print(f"   LPS+ array shape: {array_lps.shape}")

# 6. Modify volume
print("\n6. Modifying volume properties...")
mv_loaded.spacing = [2.0, 2.0, 2.5]
print(f"   Updated spacing: {mv_loaded.spacing}")

# 7. De-oblique
print("\n7. Checking if oblique...")
print(f"   Is oblique: {mv_loaded.get_geometry()['oblique']}")

# 8. Access all geometry properties
print("\n8. All geometry properties:")
print(f"   Affine:\n{mv_loaded.affine}")
print(f"   Rotation:\n{mv_loaded.rotation}")
print(f"   Shear: {mv_loaded.shear}")

# 9. Create 2D example
print("\n9. Creating 2D image...")
array_2d = np.random.rand(256, 256).astype(np.float32)
mv_2d = MedVol(array_2d)
mv_2d.save(script_dir / "data" / "comprehensive_2d.nii.gz")
print(f"   2D shape: {mv_2d.array.shape}")

# 10. Create 4D example
print("\n10. Creating 4D image (time series)...")
array_4d = np.random.rand(64, 64, 32, 10).astype(np.float32)
mv_4d = MedVol(array_4d, spacing=[2.0, 2.0, 2.0, 1.0])
mv_4d.save(script_dir / "data" / "comprehensive_4d.nii.gz")
print(f"   4D shape: {mv_4d.array.shape}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
