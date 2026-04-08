from pathlib import Path

import numpy as np

from medvol import MedVol


def showcase_3d_nifti() -> None:
    filepath = Path("examples/data/3d_img.nii.gz")
    image = MedVol(filepath)

    print("File:", filepath)
    print("Backend:", image.backend)
    print("Shape:", image.array.shape)
    print("Dtype:", image.array.dtype)
    print("Coordinate system:", image.coordinate_system)
    print("Spacing:", image.spacing)
    print("Origin:", image.origin)
    print("Direction:\n", image.direction)
    print("Affine:\n", image.affine)
    print("Translation:", image.translation)
    print("Scale:", image.scale)
    print("Rotation:\n", image.rotation)
    print("Shear:\n", image.shear)
    print("Raw header type:", type(image.header).__name__)
    print("Intensity range:", (float(np.min(image.array)), float(np.max(image.array))))
    print("Center voxel value:", image.array[tuple(size // 2 for size in image.array.shape)])


if __name__ == "__main__":
    showcase_3d_nifti()
