from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from medvol.backends.base import BackendLoadResult
from medvol.geometry import (
    CANONICAL_AXIS_LABELS,
    CoordinateContext,
    SNAP_ATOL,
    canonical_coordinate_context,
    canonicalize_array_and_affine,
    convert_affine_world_basis,
    validate_affine,
)


class NibabelBackend:
    name = "nibabel"

    def load(self, filepath: Path) -> BackendLoadResult:
        image = nib.load(str(filepath))
        array = np.asanyarray(image.dataobj)
        if array.ndim not in {2, 3, 4}:
            raise ValueError("NiBabel backend supports only 2D, 3D, and 4D arrays.")

        if array.ndim == 2:
            affine = np.eye(3, dtype=float)
            affine[:2, :2] = image.affine[:2, :2]
            affine[:2, 2] = image.affine[:2, 3]
        elif array.ndim == 3:
            affine = image.affine.astype(float)
        else:
            affine = np.eye(5, dtype=float)
            affine[:3, :3] = image.affine[:3, :3]
            affine[:3, 4] = image.affine[:3, 3]
            zooms = image.header.get_zooms()
            affine[3, 3] = float(zooms[3]) if len(zooms) > 3 else 1.0
            affine[3, 4] = float(image.header["toffset"])

        coordinate_context = CoordinateContext(
            axis_labels=(("L", "R"), ("P", "A"), ("I", "S")),
            anatomical_ndim=min(array.ndim, 3),
            anatomical_axes=tuple(range(min(array.ndim, 3))),
        )
        return BackendLoadResult(
            array=array,
            affine=validate_affine(affine, array.ndim),
            header=image.header.copy(),
            coordinate_context=coordinate_context,
            backend=self.name,
        )

    def save(self, filepath: Path, medvol) -> None:
        header = medvol.header.copy() if medvol.backend == self.name and medvol.header is not None else None
        affine = convert_affine_world_basis(
            medvol.affine,
            medvol._coordinate_context,
            CANONICAL_AXIS_LABELS[: min(medvol.ndims, 3)],
            atol=SNAP_ATOL,
        )
        array = medvol.array
        context = canonical_coordinate_context(
            medvol.ndims,
            anatomical_ndim=min(medvol.ndims, 3),
            anatomical_axes=tuple(range(min(medvol.ndims, 3))),
        )

        if medvol.ndims == 2:
            nib_affine = np.eye(4, dtype=float)
            nib_affine[:2, :2] = affine[:2, :2]
            nib_affine[:2, 3] = affine[:2, 2]
            image = nib.Nifti1Image(array, nib_affine, header=header)
        elif medvol.ndims == 3:
            image = nib.Nifti1Image(array, affine, header=header)
        elif medvol.ndims == 4:
            array, affine, context = canonicalize_array_and_affine(
                array,
                affine,
                context,
                atol=SNAP_ATOL,
            )
            linear = affine[:-1, :-1]
            if not np.allclose(linear[:3, 3], 0.0, atol=SNAP_ATOL) or not np.allclose(
                linear[3, :3], 0.0, atol=SNAP_ATOL
            ):
                raise ValueError(
                    "4D NIfTI serialization requires a block-separable 5x5 affine without spatial/4th-axis coupling."
                )
            if linear[3, 3] <= 0:
                raise ValueError("4D NIfTI serialization requires a positive 4th-axis scale.")

            nib_affine = np.eye(4, dtype=float)
            nib_affine[:3, :3] = linear[:3, :3]
            nib_affine[:3, 3] = affine[:3, 4]
            image = nib.Nifti1Image(array, nib_affine, header=header)
            zooms = image.header.get_zooms()
            image.header.set_zooms(zooms[:3] + (float(linear[3, 3]),))
            image.header["toffset"] = float(affine[3, 4])
        else:
            raise ValueError("NiBabel backend supports only 2D, 3D, and 4D arrays.")

        nib.save(image, str(filepath))
