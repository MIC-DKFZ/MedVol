from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform


SNAP_ATOL = 1e-6


@dataclass(frozen=True)
class CoordinateContext:
    axis_labels: tuple[tuple[str, str], ...]
    anatomical_ndim: int
    anatomical_axes: tuple[int, ...] | None = None


CANONICAL_AXIS_LABELS = (
    ("L", "R"),
    ("P", "A"),
    ("I", "S"),
)

SIMPLEITK_AXIS_LABELS = (
    ("R", "L"),
    ("A", "P"),
    ("I", "S"),
)


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


def canonical_coordinate_context(
    ndim: int,
    anatomical_ndim: int | None = None,
    anatomical_axes: tuple[int, ...] | None = None,
) -> CoordinateContext:
    spatial_ndim = min(ndim, 3) if anatomical_ndim is None else anatomical_ndim
    if anatomical_axes is None:
        anatomical_axes = tuple(range(spatial_ndim))
    return CoordinateContext(
        axis_labels=CANONICAL_AXIS_LABELS[:spatial_ndim],
        anatomical_ndim=spatial_ndim,
        anatomical_axes=anatomical_axes,
    )


def convert_affine_world_basis(
    affine: np.ndarray,
    source_context: CoordinateContext | None,
    target_axis_labels: Sequence[tuple[str, str]],
    *,
    atol: float = SNAP_ATOL,
) -> np.ndarray:
    if source_context is None or source_context.anatomical_ndim <= 0:
        return np.asarray(affine, dtype=float)

    affine_array = np.asarray(affine, dtype=float)
    converted = affine_array.copy()
    for axis in range(source_context.anatomical_ndim):
        source_labels = tuple(source_context.axis_labels[axis])
        target_labels = tuple(target_axis_labels[axis])
        if source_labels == target_labels:
            sign = 1.0
        elif source_labels == target_labels[::-1]:
            sign = -1.0
        else:
            raise ValueError(
                "Cannot convert affine between incompatible coordinate contexts."
            )
        converted[axis, :] *= sign

    return snap_values(converted, atol=atol)


def convert_affine_to_ras(
    affine: np.ndarray,
    context: CoordinateContext | None,
    *,
    ndim: int,
    atol: float = SNAP_ATOL,
) -> tuple[np.ndarray, CoordinateContext | None]:
    if context is None or context.anatomical_ndim <= 0:
        return np.asarray(affine, dtype=float), context

    ras_affine = convert_affine_world_basis(
        affine,
        context,
        CANONICAL_AXIS_LABELS[: context.anatomical_ndim],
        atol=atol,
    )
    return (
        ras_affine,
        canonical_coordinate_context(
            ndim,
            anatomical_ndim=context.anatomical_ndim,
            anatomical_axes=context.anatomical_axes,
        ),
    )


def _full_index_transform(
    old_shape: tuple[int, ...],
    anatomical_axes: tuple[int, ...],
    orientation_transform: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    ndim = len(old_shape)
    extras = [axis for axis in range(ndim) if axis not in anatomical_axes]
    ordered_axes = [
        anatomical_axes[int(source_axis)]
        for source_axis in orientation_transform[:, 0].astype(int)
    ] + extras

    transform = np.zeros((ndim + 1, ndim + 1), dtype=float)
    transform[-1, -1] = 1.0
    for new_axis, old_axis in enumerate(ordered_axes):
        if new_axis < len(anatomical_axes):
            flip = int(orientation_transform[new_axis, 1])
            if flip == -1:
                transform[old_axis, new_axis] = -1.0
                transform[old_axis, -1] = old_shape[old_axis] - 1
            else:
                transform[old_axis, new_axis] = 1.0
        else:
            transform[old_axis, new_axis] = 1.0

    return ordered_axes, transform


def canonicalize_array_and_affine(
    array: np.ndarray,
    affine: np.ndarray,
    context: CoordinateContext | None,
    *,
    atol: float = SNAP_ATOL,
) -> tuple[np.ndarray, np.ndarray, CoordinateContext | None]:
    ndim = array.ndim
    if context is None or context.anatomical_ndim <= 0:
        return array, np.asarray(affine, dtype=float), context

    ras_affine, ras_context = convert_affine_to_ras(
        affine, context, ndim=ndim, atol=atol
    )
    assert ras_context is not None

    anatomical_axes = (
        tuple(range(ras_context.anatomical_ndim))
        if ras_context.anatomical_axes is None
        else ras_context.anatomical_axes
    )
    sub_affine = np.eye(ras_context.anatomical_ndim + 1, dtype=float)
    sub_affine[: ras_context.anatomical_ndim, : ras_context.anatomical_ndim] = (
        ras_affine[: ras_context.anatomical_ndim, anatomical_axes]
    )
    sub_affine[: ras_context.anatomical_ndim, -1] = ras_affine[
        : ras_context.anatomical_ndim, -1
    ]

    current_orientation = io_orientation(sub_affine, tol=atol)
    target_orientation = axcodes2ornt(tuple("RAS"[: ras_context.anatomical_ndim]))
    orientation_transform = ornt_transform(current_orientation, target_orientation)

    ordered_axes, index_transform = _full_index_transform(
        array.shape,
        anatomical_axes,
        orientation_transform,
    )

    canonical_array = np.transpose(array, axes=ordered_axes)
    for new_axis, flip in enumerate(orientation_transform[:, 1].astype(int)):
        if flip == -1:
            canonical_array = np.flip(canonical_array, axis=new_axis)

    canonical_affine = snap_values(ras_affine @ index_transform, atol=atol)
    canonical_affine = validate_affine(canonical_affine, ndim, atol=atol)
    canonical_context = canonical_coordinate_context(
        ndim,
        anatomical_ndim=ras_context.anatomical_ndim,
        anatomical_axes=tuple(range(ras_context.anatomical_ndim)),
    )
    return canonical_array, canonical_affine, canonical_context


def deoblique_affine(
    affine: np.ndarray,
    *,
    atol: float = SNAP_ATOL,
) -> np.ndarray:
    affine_array = np.asarray(affine, dtype=float)
    ndim = affine_array.shape[0] - 1
    linear = affine_array[:-1, :-1]
    scales = np.linalg.norm(linear, axis=0)
    if np.any(scales <= atol):
        raise ValueError("Cannot deoblique an affine with zero-length axes.")

    deobliqued = np.eye(ndim + 1, dtype=float)
    deobliqued[:-1, :-1] = np.diag(scales)
    deobliqued[:-1, -1] = affine_array[:-1, -1]
    return validate_affine(snap_values(deobliqued, atol=atol), ndim, atol=atol)


def context_to_space_name(
    context: CoordinateContext | None,
    ndim: int,
) -> str | None:
    if context is None or context.anatomical_ndim < 3:
        return None

    axis_labels = tuple(context.axis_labels[:3])
    if axis_labels == CANONICAL_AXIS_LABELS:
        base = "right-anterior-superior"
    elif axis_labels == SIMPLEITK_AXIS_LABELS:
        base = "left-posterior-superior"
    else:
        return None

    if ndim == 4:
        return base + "-time"
    return base


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
