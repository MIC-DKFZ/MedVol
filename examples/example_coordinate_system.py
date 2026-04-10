"""Example: Converting between coordinate systems (RAS+, LPS+)."""

import numpy as np
from pathlib import Path
from medvol import MedVol

# Get script directory for paths
script_dir = Path(__file__).parent

# Load from file (default is RAS+)
mv_ras = MedVol(script_dir / "data" / "3d_img.nii.gz")
print("RAS+ coordinate system:")
print(f"  Coordinate system: {mv_ras.coordinate_system}")
print(f"  Affine:\n{mv_ras.affine}")

# Get geometry in LPS+ without modifying internal state
geometry_lps = mv_ras.get_geometry("LPS+")
print("\nLPS+ geometry (converted from RAS+):")
print(f"  Coordinate system: {geometry_lps['coordinate_system']}")
print(f"  Affine:\n{geometry_lps['affine']}")

# Convert array to LPS+
array_lps = mv_ras.get_array("LPS+")
print(f"\nArray shape in LPS+: {array_lps.shape}")

# Note: The internal MedVol object remains in RAS+
print(f"\nInternal coordinate system unchanged: {mv_ras.coordinate_system}")
