# MedVol

[![PyPI version](https://img.shields.io/pypi/v/medvol)](https://pypi.org/project/medvol/)
[![Python versions](https://img.shields.io/pypi/pyversions/medvol)](https://pypi.org/project/medvol/)
[![Build status](https://img.shields.io/github/actions/workflow/status/MIC-DKFZ/MedVol/test_and_deploy.yml?branch=main)](https://github.com/MIC-DKFZ/MedVol/actions)
[![License](https://img.shields.io/github/license/MIC-DKFZ/MedVol)](https://github.com/MIC-DKFZ/MedVol/blob/main/LICENSE)

A lightweight Python wrapper that unifies **SimpleITK**, **nibabel**, and **pynrrd** under a single, simple API for reading and writing 2‑D, 3‑D, and 4‑D medical images in NIfTI (`.nii/.nii.gz`) and NRRD (`.nrrd`) formats.


## ✨ Features

- **Unified API** – one `MedVol` class works with all three back‑ends.
- **Automatic backend selection** based on file extension:
  - `.nii/.nii.gz` → `nibabel`
  - `.nrrd` → `pynrrd`
- **Explicit backend override** via the `backend=` argument.
- **Canonical `RAS+` orientation** by default (no interpolation).
- Optional **de‑obliquing** via `get_geometry(deoblique=True)`.
- Geometry is stored in a single source of truth – the affine matrix.
- Convenient derived properties: `spacing`, `origin`, `direction`, `rotation`, `shear`, `coordinate_system`.
- Direct access to the raw backend header through `header`.


## 📦 Installation

```bash
pip install medvol
```


## 🚀 Quick start

```python
from medvol import MedVol

# Load the bundled 3‑D example image (NIfTI).
img = MedVol("examples/data/3d_img.nii.gz")

print("Backend:", img.backend)
print("Shape:", img.array.shape)
print("Coordinate system:", img.coordinate_system)
print("Spacing:", img.spacing)
print("Origin:", img.origin)
print("Direction:\n", img.direction)
print("Affine:\n", img.affine)
print("Rotation:\n", img.rotation)
print("Header type:", type(img.header).__name__)

# Access the centre voxel value.
center = tuple(s // 2 for s in img.array.shape)
print("Center voxel:", img.array[center])
```

To inspect the native geometry without canonicalisation:

```python
native = MedVol(
    "examples/data/3d_img.nii.gz",
    backend="simpleitk",
    canonicalize=False,
)
print("Native coordinate system:", native.coordinate_system)
```

See the runnable demo at `examples/example_showcase_3d_nifti.py`.


## Contributing

Contributions are welcome! Please open a pull request with clear changes and add tests when appropriate.

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.
