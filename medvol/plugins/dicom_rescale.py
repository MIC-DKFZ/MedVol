"""Per-slice and per-frame rescale (policy ``dicom_rescale``, spec section 5).

One MLArray holds one ``values.scale`` / ``values.offset``. When RescaleSlope or RescaleIntercept
differ between the slices of a series (phase 1) or the frames of a multi-frame object (phase 2),
policy ``dicom_rescale`` decides:

- ``error`` (default): the read is refused.
- ``apply``: the array holds the physical values ``slope * stored + intercept`` of each slice with
  that slice's own slope and intercept. ``values.scale`` and ``values.offset`` are not set
  (the values are already physical); the unit (e.g. ``hounsfield_unit``) is kept. The products are
  computed in float64 and stored as **float32 when every value is exactly representable in
  float32**, otherwise as float64, so the conversion itself never rounds. The choice is reported.

Writing back (``invert``): the original stored integers are recovered per slice with that slice's
own slope and intercept from ``source`` (``round((physical - intercept) / slope)``), and the result
is accepted only if applying the forward rescale again reproduces the array bit for bit and the
integers fit the stored dtype. If not (the data was changed to values the original rescale cannot
express, or resampled), the write is **refused**: MedVol does not choose a new rescale or round
values implicitly. Data that changed but still inverts exactly is written, with new SOP Instance
UIDs as for any changed data.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Pair = Tuple[float, float]


class RescaleError(ValueError):
    """Physical values cannot be written back as the stored integers of the source."""


def pair_or_identity(pair: Optional[Pair]) -> Pair:
    return (1.0, 0.0) if pair is None else (float(pair[0]), float(pair[1]))


def to_physical(stored: np.ndarray, pairs: Sequence[Optional[Pair]], axis: int = 0) -> Tuple[np.ndarray, str]:
    """``slope * stored + intercept`` per index along ``axis`` (float32 if exact, else float64).

    Returns (array, report text).
    """
    stored = np.asarray(stored)
    slopes = np.array([pair_or_identity(p)[0] for p in pairs], dtype=np.float64)
    intercepts = np.array([pair_or_identity(p)[1] for p in pairs], dtype=np.float64)
    shape = [1] * stored.ndim
    shape[axis] = len(pairs)
    physical = stored.astype(np.float64) * slopes.reshape(shape) + intercepts.reshape(shape)
    as32 = physical.astype(np.float32)
    if np.array_equal(as32.astype(np.float64), physical):
        return as32, "float32 (every value exactly representable)"
    return physical, "float64 (float32 cannot represent every value exactly)"


def stored_dtype(bits_allocated: int, pixel_representation: int) -> np.dtype:
    kind = "i" if int(pixel_representation) == 1 else "u"
    return np.dtype(f"<{kind}{int(bits_allocated) // 8}")


def invert(physical: np.ndarray, pairs: Sequence[Optional[Pair]], dtype: np.dtype, axis: int = 0) -> np.ndarray:
    """The stored integers that ``to_physical`` maps exactly onto ``physical``; raises RescaleError otherwise."""
    physical = np.asarray(physical)
    if physical.dtype.kind != "f":
        raise RescaleError(f"expected physical values in a float array, got {physical.dtype}")
    slopes = np.array([pair_or_identity(p)[0] for p in pairs], dtype=np.float64)
    intercepts = np.array([pair_or_identity(p)[1] for p in pairs], dtype=np.float64)
    shape = [1] * physical.ndim
    shape[axis] = len(pairs)
    if np.any(slopes == 0):
        raise RescaleError("a RescaleSlope of 0 cannot be inverted")
    if not np.all(np.isfinite(physical)):
        raise RescaleError("the array holds NaN or infinite values, which have no stored integer")
    candidate = np.rint((physical.astype(np.float64) - intercepts.reshape(shape)) / slopes.reshape(shape))
    info = np.iinfo(dtype)
    if candidate.min(initial=0) < info.min or candidate.max(initial=0) > info.max:
        raise RescaleError(f"the stored values would leave the range of {dtype.name}")
    forward = (candidate * slopes.reshape(shape) + intercepts.reshape(shape)).astype(physical.dtype)
    mismatch = forward != physical
    if np.any(mismatch):
        count = int(np.count_nonzero(mismatch))
        raise RescaleError(f"{count} values are not slope * integer + intercept for the rescale of their slice or "
                           "frame (the data changed, e.g. by resampling or editing); writing them as stored integers "
                           "would round them, so the write is refused")
    return candidate.astype(dtype)


def distinct(pairs: List[Optional[Pair]]) -> bool:
    return len({pair_or_identity(p) for p in pairs}) > 1
