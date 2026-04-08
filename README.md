# MedVol

`medvol` is a small wrapper around `SimpleITK`, `nibabel`, and `pynrrd` that
provides one interface for reading and writing 2D, 3D, and 4D NIfTI and NRRD
images.

## Features

- One `MedVol` API across `SimpleITK`, `nibabel`, and `pynrrd`
- Automatic backend selection by extension
  - NIfTI defaults to `nibabel`
  - NRRD defaults to `pynrrd`
- Explicit backend override with `backend=...`
- `affine` as the single source of truth
- Derived geometry properties:
  - `spacing`
  - `origin`
  - `direction`
  - `translation`
  - `rotation`
  - `scale`
  - `shear`
  - `coordinate_system`
- Raw backend-native header access via `header`

## Installation

```bash
pip install medvol
```

## Example

```python
from medvol import MedVol

# Uses the bundled 3D NIfTI example.
image = MedVol("examples/data/3d_img.nii.gz")

print("Backend:", image.backend)
print("Shape:", image.array.shape)
print("Coordinate system:", image.coordinate_system)
print("Spacing:", image.spacing)
print("Origin:", image.origin)
print("Direction:\n", image.direction)
print("Affine:\n", image.affine)
print("Rotation:\n", image.rotation)
print("Header type:", type(image.header).__name__)
print("Center voxel:", image.array[tuple(size // 2 for size in image.array.shape)])
```

See [example_showcase_3d_nifti.py](/home/k539i/Documents/projects/medvol/examples/example_showcase_3d_nifti.py) for a runnable version.

## Notes

- Geometry and array layout are backend-native. The same file can expose
  different `array`, `affine`, and `coordinate_system` values depending on the
  backend used to load it.
- For 4D NIfTI, `nibabel` and the `SimpleITK` NIfTI path only support
  block-separable affines where the spatial axes do not couple to the 4th axis.
  Unsupported 5x5 affines raise a `ValueError`.

## Development

Run the test suite with:

```bash
pytest -q
```

## License

Distributed under the terms of the Apache Software License 2.0.
