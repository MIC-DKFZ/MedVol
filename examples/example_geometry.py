"""Example: Working with geometry properties (spacing, origin, direction, rotation, shear)."""

import numpy as np
from medvol import MedVol

# Create 3D image with specific geometry
array_3d = np.random.rand(100, 100, 80).astype(np.float32)

# Define non-isotropic spacing
spacing = [1.5, 1.5, 2.0]
origin = [-75.0, -75.0, -80.0]

# Create rotation (30 degrees around z-axis)
theta = np.radians(30)
rotation = np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta), np.cos(theta), 0.0],
    [0.0, 0.0, 1.0]
])

# Create MedVol with spacing and rotation
mv = MedVol(array_3d, spacing=spacing, origin=origin)
mv.direction = rotation

print("Geometry properties:")
print(f"  Spacing: {mv.spacing}")
print(f"  Origin: {mv.origin}")
print(f"  Direction:\n{mv.direction}")
print(f"  Rotation:\n{mv.rotation}")
print(f"  Shear: {mv.shear}")
print(f"  Coordinate system: {mv.coordinate_system}")

# Modify spacing
mv.spacing = [2.0, 2.0, 2.5]
print(f"\nAfter updating spacing to [2.0, 2.0, 2.5]:")
print(f"  New spacing: {mv.spacing}")
print(f"  Affine:\n{mv.affine}")
