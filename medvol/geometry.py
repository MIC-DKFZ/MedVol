from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


SNAP_ATOL = 1e-6


@dataclass(frozen=True)
class CoordinateContext:
    axis_labels: tuple[tuple[str, str], ...]
    anatomical_ndim: int
    anatomical_axes: tuple[int, ...] | None = None


def as_float_array(values: Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    return array


def snap_values(values: np.ndarray | Sequence[float], atol: float = SNAP_ATOL) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    for decimals in range(7):
        rounded = np.round(array, decimals=decimals)
        mask = np.abs(array - rounded) <= atol
        array[mask] = rounded[mask]
    return array


def validate_affine(affine: Sequence[Sequence[float]], ndim: int, atol: float = SNAP_ATOL) -> np.ndarray:
    affine_array = np.asarray(affine, dtype=float)
    expected_shape = (ndim + 1, ndim + 1)
    if affine_array.shape != expected_shape:
        raise ValueError(f"affine must have shape {expected_shape}.")

    expected_last_row = np.zeros((ndim + 1,), dtype=float)
    expected_last_row[-1] = 1.0
    if not np.allclose(affine_array[-1], expected_last_row, atol=atol):
        raise ValueError("affine must be homogeneous with last row [0, ..., 0, 1].")

    linear = affine_array[:-1, :-1]
    if np.linalg.matrix_rank(linear, tol=atol) != ndim:
        raise ValueError("affine linear block must be invertible.")

    return affine_array.astype(np.float64, copy=False)


def compose_affine(
    spacing: Sequence[float],
    origin: Sequence[float],
    direction: Sequence[Sequence[float]],
    *,
    atol: float = SNAP_ATOL,
) -> np.ndarray:
    spacing_array = as_float_array(spacing, (len(spacing),), "spacing")
    if np.any(spacing_array <= 0):
        raise ValueError("spacing must contain positive nonzero values.")

    ndim = spacing_array.shape[0]
    origin_array = as_float_array(origin, (ndim,), "origin")
    direction_array = as_float_array(direction, (ndim, ndim), "direction")

    affine = np.eye(ndim + 1, dtype=np.float64)
    affine[:-1, :-1] = direction_array @ np.diag(spacing_array)
    affine[:-1, -1] = origin_array
    return validate_affine(snap_values(affine, atol=atol), ndim, atol=atol)


def decompose_affine(affine: np.ndarray, *, atol: float = SNAP_ATOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affine_array = np.asarray(affine, dtype=float)
    ndim = affine_array.shape[0] - 1
    linear = affine_array[:-1, :-1]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= atol):
        raise ValueError("affine linear block must have positive nonzero column norms.")

    direction = linear / spacing
    origin = affine_array[:-1, -1]
    return (
        snap_values(spacing, atol=atol),
        snap_values(origin, atol=atol),
        snap_values(direction, atol=atol),
    )


def affine_to_rotation(affine: np.ndarray, *, atol: float = SNAP_ATOL) -> np.ndarray:
    linear = np.asarray(affine, dtype=float)[:-1, :-1]
    u, _, vh = np.linalg.svd(linear)
    rotation = u @ vh
    return snap_values(rotation, atol=atol)


def affine_to_shear(affine: np.ndarray, *, atol: float = SNAP_ATOL) -> np.ndarray:
    affine_array = np.asarray(affine, dtype=float)
    rotation = affine_to_rotation(affine_array, atol=atol)
    spacing, _, _ = decompose_affine(affine_array, atol=atol)
    shear = rotation.T @ affine_array[:-1, :-1] @ np.diag(1.0 / spacing)
    return snap_values(shear, atol=atol)


def coordinate_system_from_affine(
    affine: np.ndarray,
    context: CoordinateContext | None,
    *,
    atol: float = SNAP_ATOL,
) -> str | None:
    if context is None or context.anatomical_ndim <= 0:
        return None

    axis_labels = context.axis_labels[: context.anatomical_ndim]
    axis_indices = (
        tuple(range(context.anatomical_ndim))
        if context.anatomical_axes is None
        else context.anatomical_axes
    )
    ndim = len(axis_labels)
    linear = np.asarray(affine, dtype=float)[:ndim, axis_indices]
    magnitudes = np.abs(linear).copy()
    orientation = [""] * ndim
    used_rows: set[int] = set()

    for column in range(ndim):
        best_row = None
        best_value = -np.inf
        for row in range(ndim):
            if row in used_rows:
                continue
            value = magnitudes[row, column]
            if value > best_value:
                best_value = value
                best_row = row

        if best_row is None or best_value <= atol:
            return None

        used_rows.add(best_row)
        sign = linear[best_row, column]
        if abs(sign) <= atol:
            return None
        negative_label, positive_label = axis_labels[best_row]
        orientation[column] = positive_label if sign > 0 else negative_label

    return "".join(orientation) + "+"


def direction_and_spacing_to_linear(
    direction: Sequence[Sequence[float]],
    spacing: Sequence[float],
) -> np.ndarray:
    direction_array = np.asarray(direction, dtype=float)
    spacing_array = np.asarray(spacing, dtype=float)
    return direction_array @ np.diag(spacing_array)


def normalize_backend_name(backend: str | None) -> str | None:
    if backend is None:
        return None
    normalized = backend.lower()
    if normalized not in {"simpleitk", "nibabel", "pynrrd"}:
        raise ValueError(f"Unsupported backend '{backend}'.")
    return normalized


def is_homogeneous_row(row: Iterable[float], *, atol: float = SNAP_ATOL) -> bool:
    row_array = np.asarray(tuple(row), dtype=float)
    expected = np.zeros_like(row_array)
    expected[-1] = 1.0
    return np.allclose(row_array, expected, atol=atol)
