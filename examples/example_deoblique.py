"""Example: De-obliquing geometry (removing oblique rotations)."""

import numpy as np
from medvol import MedVol

# Create 3D image with oblique orientation (30 degree rotation)
array_3d = np.random.rand(64, 64, 32).astype(np.float32)

theta = np.radians(30)
rotation = np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta), np.cos(theta), 0.0],
    [0.0, 0.0, 1.0]
])

mv = MedVol(array_3d, spacing=[2.0, 2.0, 2.5], origin=[-64.0, -64.0, -40.0])
mv.direction = rotation

print("Original (oblique) geometry:")
print(f"  Coordinate system: {mv.coordinate_system}")
print(f"  Direction:\n{mv.direction}")
print(f"  Is oblique: {mv.get_geometry()['oblique']}")

# Get de-oblied geometry
geometry_deoblique = mv.get_geometry(deoblique=True)
print("\nDe-oblied geometry:")
print(f"  Direction:\n{geometry_deoblique['direction']}")
print(f"  Is oblique: {geometry_deoblique['oblique']}")
