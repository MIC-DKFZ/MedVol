from __future__ import annotations

from pathlib import Path
from typing import Any

import nrrd
import numpy as np

from medvol.backends.base import BackendLoadResult
from medvol.geometry import CoordinateContext, context_to_space_name, validate_affine


NRRD_SPACE_LABELS = {
    "right-anterior-superior": (("L", "R"), ("P", "A"), ("I", "S")),
    "left-posterior-superior": (("R", "L"), ("A", "P"), ("I", "S")),
    "right-anterior-superior-time": (("L", "R"), ("P", "A"), ("I", "S")),
    "left-posterior-superior-time": (("R", "L"), ("A", "P"), ("I", "S")),
}


def _infer_non_spatial_scale(header: dict[str, Any], axis: int) -> float:
    spacings = header.get("spacings")
    if spacings is not None and len(spacings) > axis and spacings[axis] not in {None, "none"}:
        return float(spacings[axis])
    thicknesses = header.get("thicknesses")
    if thicknesses is not None and len(thicknesses) > axis and thicknesses[axis] not in {None, "none"}:
        return float(thicknesses[axis])
    return 1.0


class PynrrdBackend:
    name = "pynrrd"

    def load(self, filepath: Path) -> BackendLoadResult:
        array, header = nrrd.read(str(filepath))
        if array.ndim not in {2, 3, 4}:
            raise ValueError("pynrrd backend supports only 2D, 3D, and 4D arrays.")

        ndim = array.ndim
        affine = np.eye(ndim + 1, dtype=float)
        directions = header.get("space directions")
        if directions is not None:
            space_dim = None
            spatial_axes: list[int] = []
            for direction in directions:
                if direction is None or (
                    isinstance(direction, str) and direction.lower() == "none"
                ):
                    continue
                vector = np.asarray(direction, dtype=float)
                if np.isnan(vector).all():
                    continue
                space_dim = vector.shape[0]
                break
            if space_dim is None:
                space_dim = ndim

            non_spatial_row = space_dim
            for axis in range(ndim):
                direction = directions[axis]
                if direction is None or (
                    isinstance(direction, str) and direction.lower() == "none"
                ):
                    if non_spatial_row >= ndim:
                        raise ValueError("Not enough dimensions to place a non-spatial NRRD axis.")
                    affine[non_spatial_row, axis] = _infer_non_spatial_scale(header, axis)
                    non_spatial_row += 1
                    continue
                vector = np.asarray(direction, dtype=float)
                if np.isnan(vector).all():
                    if non_spatial_row >= ndim:
                        raise ValueError("Not enough dimensions to place a non-spatial NRRD axis.")
                    affine[non_spatial_row, axis] = _infer_non_spatial_scale(header, axis)
                    non_spatial_row += 1
                    continue
                affine[: vector.shape[0], axis] = vector
                spatial_axes.append(axis)
        else:
            spacings = header.get("spacings")
            if spacings is None:
                spacings = np.ones((ndim,), dtype=float)
            spatial_axes = list(range(min(3, ndim)))
            for axis in range(ndim):
                affine[axis, axis] = float(spacings[axis])

        origin = header.get("space origin")
        if origin is not None:
            origin_array = np.asarray(origin, dtype=float)
            affine[: origin_array.shape[0], -1] = origin_array

        space_name = str(header.get("space", "")).strip().lower()
        axis_labels = NRRD_SPACE_LABELS.get(space_name)
        coordinate_context = None
        if axis_labels is not None:
            anatomical_axes = tuple(spatial_axes[: min(3, len(spatial_axes))])
            coordinate_context = CoordinateContext(
                axis_labels=axis_labels,
                anatomical_ndim=len(anatomical_axes),
                anatomical_axes=anatomical_axes,
            )

        return BackendLoadResult(
            array=array,
            affine=validate_affine(affine, ndim),
            header=header.copy(),
            coordinate_context=coordinate_context,
            backend=self.name,
        )

    def save(self, filepath: Path, medvol) -> None:
        source_header = medvol.header if medvol.backend == self.name and isinstance(medvol.header, dict) else {}
        header = dict(source_header)
        geometry_keys = {
            "space directions",
            "space origin",
            "sizes",
            "dimension",
            "spacings",
        }
        for key in geometry_keys:
            header.pop(key, None)

        affine = medvol.affine
        ndim = medvol.ndims
        header["space directions"] = [affine[:-1, axis].copy() for axis in range(ndim)]
        header["space origin"] = affine[:-1, -1].copy()
        header["dimension"] = ndim
        header["sizes"] = medvol.array.shape
        space_name = context_to_space_name(medvol._coordinate_context, ndim)
        if space_name is not None:
            header["space"] = space_name
        else:
            header.pop("space", None)
        nrrd.write(str(filepath), medvol.array, header=header)
