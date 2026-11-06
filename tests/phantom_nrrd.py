"""Extra NRRD phantoms: any axis layout (vector-first, time, ...), every space name, written with pynrrd.

Like ``phantom.py``, the files are written with the reference library (pynrrd), never with MedVol.
Each voxel's value is its Fortran-order linear index in the file, so ``AxisTruth.world_of_values``
recovers the file index and, from the spatial file axes, the true LPS position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import phantom

_LAS = np.diag([1.0, -1.0, 1.0])

#: every NRRD space name -> 3x3 matrix taking LPS coordinates to the space's coordinates
SPACES = dict(phantom.NRRD_FRAMES)
SPACES.update({
    "3D-left-handed": np.eye(3),                       # ITK takes its numbers as LPS (KB ch04 §2.2)
    "LPS": np.eye(3), "RAS": phantom.FLIP, "LAS": _LAS,
})
TIME_SPACES = {
    "left-posterior-superior-time": np.eye(3), "right-anterior-superior-time": phantom.FLIP,
    "left-anterior-superior-time": _LAS, "scanner-xyz-time": np.eye(3), "3D-right-handed-time": np.eye(3),
    "3D-left-handed-time": np.eye(3), "RAST": phantom.FLIP, "LPST": np.eye(3), "LAST": _LAS,
}


@dataclass
class AxisTruth(phantom.Truth):
    """Ground truth for files whose spatial axes are ``spatial`` (file axes i, j, k in that order)."""

    spatial: tuple = (0, 1, 2)

    def world_of_values(self, values: np.ndarray) -> np.ndarray:
        index = np.array(np.unravel_index(values.astype(np.int64), self.shape, order="F"))[list(self.spatial)]
        return (self.affine_lps @ np.vstack([index, np.ones(index.shape[1])]))[:3].T


def write(path, truth: AxisTruth, kinds=None, space="left-posterior-superior", time_axis=None, time=(0.0, 1.0),
          extra=None, detached=False, big_endian=False, encoding="raw", directions=None):
    """An NRRD with the spatial axes ``truth.spatial``; other axes get ``none`` directions.

    For ``*-time`` spaces, ``time_axis`` gets the direction ``(0, 0, 0, step)`` and the origin's 4th
    component is the time of its first sample (``time = (start, step)``).
    ``directions`` overrides the per-axis direction rows (NaN row = ``none``).
    """
    import nrrd

    ndim = len(truth.shape)
    timed = space in TIME_SPACES
    to_space = TIME_SPACES[space] if timed else SPACES[space]
    world = 4 if timed else 3
    linear = to_space @ truth.affine_lps[:3, :3]
    origin = list(to_space @ truth.affine_lps[:3, 3])
    rows = np.full((ndim, world), np.nan)
    for column, axis in enumerate(truth.spatial):
        rows[axis, :3] = linear[:, column]
        if timed:
            rows[axis, 3] = 0.0
    if timed:
        rows[time_axis] = [0.0, 0.0, 0.0, time[1]]
        origin.append(time[0])
    if directions is not None:
        rows = np.asarray(directions, dtype=float)
    header = {
        "space": space,
        "space directions": rows,
        "space origin": np.array(origin),
        "kinds": kinds or ["domain"] * ndim,
        "encoding": encoding,
    }
    header.update(extra or {})
    data = truth.volume()
    if big_endian:
        data = data.astype(data.dtype.newbyteorder(">"))
    nrrd.write(str(path), data, header, detached_header=detached)
    return path


def write_plain(path, truth: AxisTruth, header):
    """An NRRD with exactly the given header fields (no space unless given)."""
    import nrrd

    nrrd.write(str(path), truth.volume(), dict(header))
    return path
