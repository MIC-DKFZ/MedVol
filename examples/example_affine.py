"""Example: Custom affine transformations."""

import numpy as np
from medvol import MedVol

# Create 3D image
array_3d = np.random.rand(64, 64, 32).astype(np.float32)

# Define custom affine with scaling, translation, and rotation
spacing = [2.0, 2.0, 3.0]
origin = [-64.0, -64.0, -48.0]
rotation = np.eye(3)

mv = MedVol(array_3d, spacing=spacing, origin=origin, direction=rotation)

print("Original affine:")
print(mv.affine)

# Apply translation
mv.origin = [0.0, 0.0, 0.0]
print("\nAfter moving origin to [0, 0, 0]:")
print(mv.affine)

# Apply non-uniform scaling
mv.spacing = [1.0, 1.5, 2.0]
print("\nAfter scaling [1.0, 1.5, 2.0]:")
print(mv.affine)

# Reset affine
new_affine = np.array([
    [1.0, 0.0, 0.0, -32.0],
    [0.0, 1.0, 0.0, -32.0],
    [0.0, 0.0, 1.0, -16.0],
    [0.0, 0.0, 0.0, 1.0]
])
mv.affine = new_affine
print("\nAfter setting custom affine:")
print(mv.affine)
