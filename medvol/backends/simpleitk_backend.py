from __future__ import annotations

import locale
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk
import numpy as np

from medvol.backends.base import BackendLoadResult
from medvol.geometry import CoordinateContext, SNAP_ATOL, compose_affine, decompose_affine


@contextmanager
def temporary_c_locale():
    old_locale = locale.setlocale(locale.LC_NUMERIC, None)
    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
        yield
    finally:
        locale.setlocale(locale.LC_NUMERIC, old_locale)


class SimpleITKBackend:
    name = "simpleitk"
    _lps_from_ras = np.diag([-1.0, -1.0, 1.0, 1.0])

    @staticmethod
    def _load_geometry(img: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim = img.GetDimension()
        permutation = np.arange(dim)[::-1]
        spacing = np.asarray(img.GetSpacing(), dtype=float)[permutation]
        origin = np.asarray(img.GetOrigin(), dtype=float)
        direction = np.asarray(img.GetDirection(), dtype=float).reshape(dim, dim)[:, permutation]
        return spacing, origin, direction

    @staticmethod
    def _is_nifti(filepath: Path) -> bool:
        return filepath.suffix == ".nii" or filepath.suffixes[-2:] == [".nii", ".gz"]

    def _load_nifti_4d(self, filepath: Path, image: sitk.Image) -> BackendLoadResult:
        nib_image = nib.load(str(filepath))
        if nib_image.ndim != 4:
            raise ValueError("Expected a 4D NIfTI image.")

        spatial_affine = self._lps_from_ras @ nib_image.affine
        affine = np.zeros((5, 5), dtype=float)
        affine[-1, -1] = 1.0
        affine[:3, 1] = spatial_affine[:3, 2]
        affine[:3, 2] = spatial_affine[:3, 1]
        affine[:3, 3] = spatial_affine[:3, 0]
        affine[:3, 4] = spatial_affine[:3, 3]
        zooms = nib_image.header.get_zooms()
        affine[3, 0] = float(zooms[3]) if len(zooms) > 3 else 1.0
        affine[3, 4] = float(nib_image.header["toffset"])

        header = {key: image.GetMetaData(key) for key in image.GetMetaDataKeys()}
        coordinate_context = CoordinateContext(
            axis_labels=(("R", "L"), ("A", "P"), ("I", "S")),
            anatomical_ndim=3,
            anatomical_axes=(1, 2, 3),
        )
        return BackendLoadResult(
            array=sitk.GetArrayFromImage(image),
            affine=affine,
            header=header,
            coordinate_context=coordinate_context,
            backend=self.name,
        )

    def load(self, filepath: Path) -> BackendLoadResult:
        with temporary_c_locale():
            image = sitk.ReadImage(str(filepath))
        if self._is_nifti(filepath) and image.GetDimension() == 4:
            return self._load_nifti_4d(filepath, image)

        array = sitk.GetArrayFromImage(image)
        if image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("SimpleITK vector images are not supported.")
        if array.ndim != image.GetDimension():
            raise ValueError("SimpleITK image dimensionality does not match array dimensionality.")

        spacing, origin, direction = self._load_geometry(image)
        affine = compose_affine(spacing, origin, direction)
        header = {key: image.GetMetaData(key) for key in image.GetMetaDataKeys()}
        coordinate_context = CoordinateContext(
            axis_labels=(("R", "L"), ("A", "P"), ("I", "S")),
            anatomical_ndim=min(array.ndim, 3),
            anatomical_axes=tuple(range(min(array.ndim, 3))),
        )
        return BackendLoadResult(
            array=array,
            affine=affine,
            header=header,
            coordinate_context=coordinate_context,
            backend=self.name,
        )

    def save(self, filepath: Path, medvol) -> None:
        if self._is_nifti(filepath) and medvol.ndims == 4:
            linear = medvol.affine[:-1, :-1]
            if not np.allclose(medvol.affine[:3, 0], 0.0, atol=SNAP_ATOL) or not np.allclose(
                medvol.affine[3, 1:4], 0.0, atol=SNAP_ATOL
            ):
                raise ValueError(
                    "4D NIfTI serialization with the SimpleITK backend requires a block-separable affine in backend-native axis order."
                )
            if medvol.affine[3, 0] <= 0:
                raise ValueError("4D NIfTI serialization requires a positive 4th-axis scale.")

            spatial_lps = np.eye(4, dtype=float)
            spatial_lps[:3, 0] = linear[:3, 3]
            spatial_lps[:3, 1] = linear[:3, 2]
            spatial_lps[:3, 2] = linear[:3, 1]
            spatial_lps[:3, 3] = medvol.affine[:3, 4]
            spatial_ras = self._lps_from_ras @ spatial_lps

            image = nib.Nifti1Image(
                medvol.array.transpose(3, 2, 1, 0),
                spatial_ras,
            )
            zooms = image.header.get_zooms()
            image.header.set_zooms(zooms[:3] + (float(medvol.affine[3, 0]),))
            image.header["toffset"] = float(medvol.affine[3, 4])
            nib.save(image, str(filepath))
            return

        spacing, origin, direction = decompose_affine(medvol.affine)
        permutation = np.arange(medvol.ndims)[::-1]

        image = sitk.GetImageFromArray(medvol.array, isVector=False)
        image.SetSpacing(spacing[permutation].tolist())
        image.SetOrigin(origin.tolist())
        image.SetDirection(direction[:, permutation].flatten().tolist())

        if medvol.backend == self.name and isinstance(medvol.header, dict):
            for key, value in medvol.header.items():
                image.SetMetaData(str(key), str(value))

        with temporary_c_locale():
            sitk.WriteImage(image, str(filepath), useCompression=True)
