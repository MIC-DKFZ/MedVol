"""Opt-in resampling of tilted or irregularly spaced DICOM stacks (policies ``dicom_tilt='resample'``
and ``dicom_irregular_spacing='resample'``, spec section 5).

The default policies keep the data as acquired (``dicom_tilt='exact'`` stores a sheared grid) or
refuse it (``dicom_irregular_spacing='error'``). Resampling changes the voxel values and is only
done when it is asked for; everything it does is reported.

Target grid (one rule for both policies):

- The in-plane grid is the one of the first slice along the slice normal: same origin
  (its ImagePositionPatient), same row and column directions, same pixel spacing and size.
- Slices are placed along the slice normal ``n = row direction x column direction``, so the
  grid has no shear.
- The slice spacing is the **mean** spacing along the normal, ``(h_last - h_first) / (N - 1)``,
  where ``h`` is the projection of ImagePositionPatient on the normal. The target therefore has as
  many slices as the source, its first and last slice planes are the source's first and last
  slice planes, and nothing along the normal is extrapolated. (The smallest spacing would add
  slices that are all interpolated and makes the grid depend on the closest pair of slices.)

Interpolation, in world space, per output voxel centre ``P``:

1. Along the normal: find the two source slices whose planes bracket ``P``
   (``h_a <= h(P) <= h_b``); the weight is ``w = (h(P) - h_a) / (h_b - h_a)``.
2. In each of the two slices: project ``P`` along the normal onto the slice plane and express
   it in that slice's own pixel coordinates, using that slice's own ImagePositionPatient. For a
   tilted stack this is where the in-plane shift between slices is accounted for; the projected
   point generally falls between pixel centres and is sampled bilinearly.
3. The value is ``(1 - w) * value_a + w * value_b``.

This is trilinear interpolation over the (sheared or irregular) source grid; it reproduces any
function that is linear in world position exactly, which is what the tests check. Output voxels
whose projection falls outside a needed source slice (the in-plane shift of a tilted stack moves
the source field of view away from the first slice's grid) get the fill value: ``values.fill``
if the source defines one (PixelPaddingValue), else NaN, which then becomes ``values.fill``.

The result is floating point: float32 when the stored values are integers of at most 16 bits or
float32, float64 otherwise (32-bit integers, float64). Interpolation is computed in float64.

The report gives the method, the target spacing, the largest distance from an output voxel centre
to the nearest source voxel centre, the largest misplacement that relabelling the source slices
onto the target grid without resampling would have caused, and the number of output voxels
without source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Neighbours whose bilinear or linear weight is at most this do not count for coverage.
WEIGHT_EPSILON = 1e-9


@dataclass
class Resampling:
    """How one stack is resampled onto a regular grid without shear.

    Returned by ``dicom._sorted_stack`` in place of the slice step vector when a resampling
    policy applies; the image builders then resample the stacked slices with ``apply``.
    """

    reasons: Tuple[str, ...]          # "gantry tilt", "irregular slice spacing"
    positions: np.ndarray             # (N, 3) ImagePositionPatient of the source slices, sorted along the normal
    normal: np.ndarray                # unit slice normal
    spacing: float                    # target slice spacing along the normal (mean spacing)
    source_step: Optional[np.ndarray] = None   # step vector of a regular (tilted) source grid; None if irregular

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def origin(self) -> np.ndarray:
        return self.positions[0]

    @property
    def step(self) -> np.ndarray:
        return self.normal * self.spacing

    @property
    def distances(self) -> np.ndarray:
        return self.positions @ self.normal


def plan(positions: np.ndarray, normal: np.ndarray, reasons: Tuple[str, ...],
         source_step: Optional[np.ndarray] = None) -> Resampling:
    """The resampling of slices at ``positions`` (sorted along ``normal``) onto the mean-spacing grid."""
    positions = np.asarray(positions, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    distances = positions @ normal
    spacing = float((distances[-1] - distances[0]) / (len(positions) - 1))
    return Resampling(tuple(reasons), positions, normal, spacing, source_step)


def output_dtype(dtype: np.dtype) -> np.dtype:
    """float32 for integers of at most 16 bits and float32, float64 otherwise."""
    dtype = np.dtype(dtype)
    if dtype == np.float32 or (dtype.kind in "iub" and dtype.itemsize <= 2):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _shift_sample(image: np.ndarray, dj: float, di: float) -> Tuple[np.ndarray, np.ndarray]:
    """``image`` (rows, columns) sampled bilinearly at (j + dj, i + di) for every pixel (j, i).

    Returns (values in float64, valid mask): a pixel is valid when every neighbour with a weight
    above ``WEIGHT_EPSILON`` lies inside the image.
    """
    rows, columns = image.shape
    j0, i0 = int(np.floor(dj)), int(np.floor(di))
    fj, fi = dj - j0, di - i0
    jj = np.arange(rows) + j0
    ii = np.arange(columns) + i0
    weights_j = ((jj, 1.0 - fj), (jj + 1, fj))
    weights_i = ((ii, 1.0 - fi), (ii + 1, fi))
    values = np.zeros((rows, columns), dtype=np.float64)
    valid = np.ones((rows, columns), dtype=bool)
    source = image.astype(np.float64, copy=False)
    for rj, wj in weights_j:
        inside_j = (rj >= 0) & (rj < rows)
        if wj > WEIGHT_EPSILON:
            valid &= inside_j[:, None]
        if wj == 0.0:
            continue
        cj = np.clip(rj, 0, rows - 1)
        for ri, wi in weights_i:
            inside_i = (ri >= 0) & (ri < columns)
            if wi > WEIGHT_EPSILON and wj > WEIGHT_EPSILON:
                valid &= inside_i[None, :]
            if wi == 0.0:
                continue
            ci = np.clip(ri, 0, columns - 1)
            values += (wj * wi) * source[np.ix_(cj, ci)]
    return values, valid


def apply(volume: np.ndarray, resampling: Resampling, orientation: np.ndarray, pixel_spacing: Tuple[float, float],
          fill: Optional[float]) -> Tuple[np.ndarray, Dict[str, float]]:
    """``volume`` (..., N, rows, columns) in source slice order resampled onto the target grid.

    ``orientation``: ImageOrientationPatient (column direction first, then row direction, as in
    DICOM: ``orientation[:3]`` is the direction of increasing column index). ``fill``: the value of
    output voxels without source data (NaN if None). Returns (array, statistics).
    """
    orientation = np.asarray(orientation, dtype=np.float64)
    column_direction, row_direction = orientation[:3], orientation[3:]
    row_spacing, column_spacing = (float(v) for v in pixel_spacing)
    n = resampling.count
    if volume.shape[-3] != n:
        raise ValueError(f"the volume has {volume.shape[-3]} slices, the resampling plan {n}")
    dtype = output_dtype(volume.dtype)
    fill_value = np.nan if fill is None else float(fill)
    distances = resampling.distances
    h0 = distances[0]
    flat = volume.reshape(-1, *volume.shape[-3:])
    out = np.empty(flat.shape, dtype=dtype)
    interpolation_distance = 0.0
    outside = 0
    for t in range(n):
        target_origin = resampling.origin + t * resampling.step
        h = h0 + t * resampling.spacing
        a = int(np.clip(np.searchsorted(distances, h, side="right") - 1, 0, n - 2))
        gap = distances[a + 1] - distances[a]
        w = float(np.clip((h - distances[a]) / gap, 0.0, 1.0))
        needed: List[Tuple[int, float]] = [(a, 1.0 - w), (a + 1, w)]
        nearest = []
        offsets = {}
        for slice_index, weight in needed:
            delta = target_origin - resampling.positions[slice_index]
            dj = float(delta @ row_direction) / row_spacing
            di = float(delta @ column_direction) / column_spacing
            offsets[slice_index] = (dj, di)
            fj, fi = dj - np.floor(dj), di - np.floor(di)
            in_plane = np.hypot(min(fj, 1 - fj) * row_spacing, min(fi, 1 - fi) * column_spacing)
            along = abs(h - distances[slice_index])
            nearest.append(float(np.hypot(in_plane, along)))
        interpolation_distance = max(interpolation_distance, min(nearest))
        for m in range(flat.shape[0]):
            values = np.zeros(flat.shape[-2:], dtype=np.float64)
            valid = np.ones(flat.shape[-2:], dtype=bool)
            for slice_index, weight in needed:
                if weight == 0.0:
                    continue
                sampled, inside = _shift_sample(flat[m, slice_index], *offsets[slice_index])
                values += weight * sampled
                if weight > WEIGHT_EPSILON:
                    valid &= inside
            values[~valid] = fill_value
            out[m, t] = values
            if m == 0:
                outside += int(np.count_nonzero(~valid))
    relabel = float(np.max(np.linalg.norm(
        resampling.positions - (resampling.origin + np.outer(np.arange(n), resampling.step)), axis=1)))
    stats = {
        "spacing": resampling.spacing,
        "interpolation_distance": interpolation_distance,
        "relabel_error": relabel,
        "outside_per_volume": outside,
        "voxels_per_volume": int(np.prod(flat.shape[-3:])),
    }
    return out.reshape(volume.shape[:-3] + out.shape[-3:]), stats


def describe(resampling: Resampling, stats: Dict[str, float], fill: Optional[float]) -> List[str]:
    """Report lines for a resampling."""
    fill_text = "NaN" if fill is None or (isinstance(fill, float) and np.isnan(fill)) else repr(fill)
    lines = [
        f"{' and '.join(resampling.reasons)}: resampled onto a regular grid without shear "
        f"(in-plane grid of the first slice, {resampling.count} slices along the normal at the mean spacing "
        f"{stats['spacing']:.6g} mm; trilinear interpolation in world space)",
        f"resampling: the largest distance from an output voxel centre to the nearest source voxel centre is "
        f"{stats['interpolation_distance']:.6g} mm; relabelling the source slices onto this grid without "
        f"resampling would misplace voxels by up to {stats['relabel_error']:.6g} mm",
    ]
    if stats["outside_per_volume"]:
        lines.append(f"resampling: {stats['outside_per_volume']} of {stats['voxels_per_volume']} output voxels per volume "
                     f"lie outside the source slices and are set to {fill_text}")
    return lines


def method(resampling: Resampling) -> str:
    """The value recorded under the policy names in the report's choices."""
    return f"resample (trilinear in world space, mean spacing {resampling.spacing:.6g} mm, no shear)"
