"""Geometry helpers shared by all plugins (spec section 6).

Conventions:
- MLArray arrays are in C order, last axis = the format's fastest-varying axis (``[k, j, i]``).
- An integer index is the centre of its voxel everywhere.
- Patient spaces are built directly from the format's native fields. Changing the world frame
  (LPS <-> RAS) is a sign change of matrix rows, never an array flip.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: increases_toward word -> (LPS axis index, sign)
WORD_TO_LPS = {
    "left": (0, 1.0), "right": (0, -1.0),
    "posterior": (1, 1.0), "anterior": (1, -1.0),
    "superior": (2, 1.0), "inferior": (2, -1.0),
}
FRAME_WORDS = {
    "dicom_patient": ("left", "posterior", "superior"),
    "ras": ("right", "anterior", "superior"),
}
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


class GeometryError(ValueError):
    """Geometry that cannot be interpreted or represented."""


def homogeneous(linear: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    linear = np.asarray(linear, dtype=np.float64)
    rows, cols = linear.shape
    matrix = np.zeros((rows + 1, cols + 1))
    matrix[:rows, :cols] = linear
    matrix[:rows, cols] = translation
    matrix[rows, cols] = 1.0
    return matrix


def patient_space(matrix_lps: np.ndarray, array_axes: Sequence[str], world_frame: str = "dicom_patient",
                  name: str = "patient", reference: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """A medical space from a matrix that maps ``array_axes`` (columns) to LPS (rows).

    ``world_frame`` chooses the stored frame: ``dicom_patient`` (LPS) or ``ras``.
    """
    matrix_lps = np.asarray(matrix_lps, dtype=np.float64)
    if matrix_lps.shape[0] != 4:
        raise GeometryError(f"A patient space has 3 axes; got a matrix with {matrix_lps.shape[0] - 1} rows.")
    matrix = matrix_lps.copy()
    if world_frame == "ras":
        matrix[:3] = LPS_TO_RAS @ matrix[:3]
    words = FRAME_WORDS[world_frame]
    space = {
        "name": name,
        "axes": [{"name": axis, "unit": "millimeter", "increases_toward": word} for axis, word in zip("xyz", words)],
        "from_array": {"array_axes": list(array_axes), "matrix": matrix.tolist()},
    }
    if reference is not None:
        space = {"name": name, "reference": dict(reference), **{k: v for k, v in space.items() if k != "name"}}
    return space


def lps_matrix(space) -> np.ndarray:
    """The from-array matrix of a medical space with its rows in LPS order.

    The space axes must name the three patient directions with ``increases_toward``
    (any order and sign: LPS, RAS, LAS, ...). Raises GeometryError otherwise.
    """
    words = [axis.increases_toward for axis in space.axes]
    if len(words) != 3 or any(word not in WORD_TO_LPS for word in words):
        raise GeometryError(f"Space {space.name!r} is not a patient space: its axes increase toward {words}.")
    targets = [WORD_TO_LPS[word] for word in words]
    if sorted(axis for axis, _ in targets) != [0, 1, 2]:
        raise GeometryError(f"Space {space.name!r} does not cover left-right, posterior-anterior and superior-inferior once each: {words}.")
    for axis in space.axes:
        if axis.unit not in ("millimeter", None) and axis.unit_code not in ("mm",):
            raise GeometryError(f"Space {space.name!r} axis {axis.name!r} is in {axis.unit!r}, not millimeter.")
    matrix = space.matrix()
    out = np.zeros_like(matrix)
    for row, (lps_axis, sign) in enumerate(targets):
        out[lps_axis] = sign * matrix[row]
    out[-1] = matrix[-1]
    return out


def column_order(space, axes_in_order: Sequence[str]) -> np.ndarray:
    """Columns of ``space``'s matrix (without translation) reordered to ``axes_in_order``."""
    array_axes = list(space.from_array.array_axes)
    missing = [name for name in axes_in_order if name not in array_axes]
    if missing:
        raise GeometryError(f"Axes {missing} are not mapped by space {space.name!r} ({array_axes}).")
    return np.array([array_axes.index(name) for name in axes_in_order])


def has_shear(linear: np.ndarray, tolerance: float = 1e-6) -> bool:
    """True when the columns of a linear map are not mutually orthogonal."""
    columns = linear / np.linalg.norm(linear, axis=0)
    gram = columns.T @ columns
    return bool(np.max(np.abs(gram - np.eye(gram.shape[0]))) > tolerance)


def is_oblique(linear: np.ndarray, tolerance: float = 1e-6) -> bool:
    """True when some column is not parallel to a world axis."""
    columns = np.abs(linear / np.linalg.norm(linear, axis=0))
    return bool(np.any((columns > tolerance) & (columns < 1 - tolerance)))


def world_corners(matrix: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """World coordinates of the corner voxel centres of a grid (columns of ``matrix`` follow ``shape``)."""
    shape = list(shape)
    corners = np.array(np.meshgrid(*[[0, max(n - 1, 0)] for n in shape], indexing="ij")).reshape(len(shape), -1)
    points = np.vstack([corners, np.ones(corners.shape[1])])
    return (matrix @ points)[:-1].T


def closest_axes(linear: np.ndarray) -> Tuple[List[int], List[float]]:
    """For each column, the world axis it points along most and the sign (a permutation if unambiguous)."""
    rows = [int(np.argmax(np.abs(col))) for col in linear.T]
    signs = [float(np.sign(linear[r, c])) for c, r in enumerate(rows)]
    return rows, signs
