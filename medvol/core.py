from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medvol.geometry import (
    SNAP_ATOL,
    affine_to_rotation,
    affine_to_shear,
    canonical_coordinate_context,
    canonicalize_array_and_affine,
    compose_affine,
    coordinate_system_from_affine,
    deoblique_affine,
    decompose_affine,
    normalize_backend_name,
    validate_affine,
)
from medvol.registry import get_backend, resolve_backend


class MedVol:
    def __init__(
        self,
        source: np.ndarray | str | Path,
        *,
        affine: Sequence[Sequence[float]] | None = None,
        spacing: Sequence[float] | None = None,
        origin: Sequence[float] | None = None,
        direction: Sequence[Sequence[float]] | None = None,
        header: Any = None,
        backend: str | None = None,
        canonicalize: bool = True,
        remove_obliqueness: bool = False,
    ) -> None:
        if remove_obliqueness and not canonicalize:
            raise ValueError("remove_obliqueness requires canonicalize=True.")

        self._coordinate_context = None
        self._header = None
        self._backend = normalize_backend_name(backend)
        self._canonicalize = canonicalize
        self._remove_obliqueness = remove_obliqueness

        if isinstance(source, (str, Path)):
            if any(value is not None for value in (affine, spacing, origin, direction)):
                raise ValueError(
                    "affine, spacing, origin, and direction cannot be set when loading from a file."
                )
            resolved_backend = resolve_backend(source, self._backend)
            result = get_backend(resolved_backend).load(Path(source))
            self._array = self._validate_array(result.array)
            self._affine = validate_affine(result.affine, self.ndims)
            self._header = result.header
            self._coordinate_context = result.coordinate_context
            self._apply_orientation_policy()
            self._backend = result.backend
            return

        self._array = self._validate_array(source)
        if affine is not None and any(value is not None for value in (spacing, origin, direction)):
            raise ValueError("Use either affine or spacing/origin/direction, not both.")

        if affine is not None:
            self._affine = validate_affine(affine, self.ndims)
        else:
            spacing_values = np.ones((self.ndims,), dtype=float) if spacing is None else spacing
            origin_values = np.zeros((self.ndims,), dtype=float) if origin is None else origin
            direction_values = np.eye(self.ndims, dtype=float) if direction is None else direction
            self._affine = compose_affine(
                spacing_values,
                origin_values,
                direction_values,
                atol=SNAP_ATOL,
            )
        self._header = header
        self._coordinate_context = canonical_coordinate_context(self.ndims)
        self._apply_orientation_policy()

    def _apply_orientation_policy(self) -> None:
        if self._canonicalize:
            self._array, self._affine, self._coordinate_context = canonicalize_array_and_affine(
                self._array,
                self._affine,
                self._coordinate_context,
                atol=SNAP_ATOL,
            )
        if self._remove_obliqueness:
            self._affine = deoblique_affine(self._affine, atol=SNAP_ATOL)

    @staticmethod
    def _validate_array(array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise ValueError("source must be a NumPy array or a filepath.")
        if array.ndim not in {2, 3, 4}:
            raise ValueError("Array must be 2D, 3D, or 4D.")
        return array

    @property
    def ndims(self) -> int:
        return self._array.ndim

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        array = self._validate_array(value)
        if hasattr(self, "_affine") and array.ndim != self.ndims:
            raise ValueError("array ndim cannot change without recreating the MedVol object.")
        self._array = array

    @property
    def affine(self) -> np.ndarray:
        return self._affine.copy()

    @affine.setter
    def affine(self, value: Sequence[Sequence[float]]) -> None:
        self._affine = validate_affine(value, self.ndims)

    @property
    def spacing(self) -> np.ndarray:
        spacing, _, _ = decompose_affine(self._affine)
        return spacing

    @spacing.setter
    def spacing(self, value: Sequence[float]) -> None:
        spacing = np.asarray(value, dtype=float)
        if spacing.shape != (self.ndims,):
            raise ValueError(f"spacing must have shape {(self.ndims,)}.")
        if np.any(spacing <= 0):
            raise ValueError("spacing must contain positive nonzero values.")
        _, origin, direction = decompose_affine(self._affine)
        self._affine = compose_affine(spacing, origin, direction)

    @property
    def origin(self) -> np.ndarray:
        _, origin, _ = decompose_affine(self._affine)
        return origin

    @origin.setter
    def origin(self, value: Sequence[float]) -> None:
        origin = np.asarray(value, dtype=float)
        if origin.shape != (self.ndims,):
            raise ValueError(f"origin must have shape {(self.ndims,)}.")
        spacing, _, direction = decompose_affine(self._affine)
        self._affine = compose_affine(spacing, origin, direction)

    @property
    def direction(self) -> np.ndarray:
        _, _, direction = decompose_affine(self._affine)
        return direction

    @direction.setter
    def direction(self, value: Sequence[Sequence[float]]) -> None:
        direction = np.asarray(value, dtype=float)
        if direction.shape != (self.ndims, self.ndims):
            raise ValueError(
                f"direction must have shape {(self.ndims, self.ndims)}."
            )
        spacing, origin, _ = decompose_affine(self._affine)
        self._affine = compose_affine(spacing, origin, direction)

    @property
    def translation(self) -> np.ndarray:
        return self.origin

    @property
    def scale(self) -> np.ndarray:
        return self.spacing

    @property
    def rotation(self) -> np.ndarray:
        return affine_to_rotation(self._affine)

    @property
    def shear(self) -> np.ndarray:
        return affine_to_shear(self._affine)

    @property
    def coordinate_system(self) -> str | None:
        return coordinate_system_from_affine(self._affine, self._coordinate_context)

    @property
    def header(self) -> Any:
        return self._header

    @header.setter
    def header(self, value: Any) -> None:
        self._header = value

    @property
    def backend(self) -> str | None:
        return self._backend

    def save(self, filepath: str | Path, *, backend: str | None = None) -> None:
        resolved_backend = resolve_backend(filepath, backend)
        get_backend(resolved_backend).save(Path(filepath), self)

    def __repr__(self) -> str:
        coordinate_system = self.coordinate_system
        return (
            f"MedVol(shape={self.array.shape}, backend={self.backend!r}, "
            f"coordinate_system={coordinate_system!r})"
        )
