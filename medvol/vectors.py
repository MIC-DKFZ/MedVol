"""The frame of vector components (spec section 5, policy ``vector_frame``).

A component axis whose components are world vectors carries ``labels``: for each component, the world
direction its value increases toward, in the ``increases_toward`` vocabulary of spaces. In the DICOM
patient frame these are ``["left", "posterior", "superior"]``, in RAS ``["right", "anterior", "superior"]``.

Stored values are left as they are wherever the frame can be recorded. They change only when a format
needs it, and every such change is reported:
- an NRRD ``measurement frame`` that is a general rotation is applied on read (spec 7.2);
- a writer that can only hold some frames re-expresses the components (a signed permutation).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from medvol.geometry import FRAME_WORDS, WORD_TO_LPS

#: vector_frame policy value -> component labels
POLICY_WORDS: Dict[str, Tuple[str, str, str]] = {"lps": FRAME_WORDS["dicom_patient"], "ras": FRAME_WORDS["ras"]}
#: the frames a reader can be told with vector_frame, by their labels
NAMED_FRAMES: Dict[Tuple[str, ...], str] = {words: name.upper() for name, words in POLICY_WORDS.items()}
POLICY_OPTIONS = "'lps' or 'ras'"


class VectorFrameError(ValueError):
    """The frame of vector components is undeclared, or cannot be applied as the policy asks."""


def letters(words: Sequence[str]) -> str:
    """Display letters of a frame (``LPS``), never stored."""
    return "".join(word[0].upper() for word in words)


def describe(words: Sequence[str]) -> str:
    return f"{letters(words)} ({', '.join(words)})"


def basis(words: Sequence[str]) -> Optional[np.ndarray]:
    """3x3 matrix whose column c is the LPS unit vector component c points along (None: not a patient frame)."""
    if words is None or len(words) != 3 or any(word not in WORD_TO_LPS for word in words):
        return None
    targets = [WORD_TO_LPS[word] for word in words]
    if sorted(axis for axis, _ in targets) != [0, 1, 2]:
        return None
    out = np.zeros((3, 3))
    for column, (axis, sign) in enumerate(targets):
        out[axis, column] = sign
    return out


def words_of(matrix: np.ndarray) -> Optional[Tuple[str, str, str]]:
    """Labels of a basis (columns in LPS) that is a signed permutation with entries exactly 0 and +-1, else None."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isin(matrix, (-1.0, 0.0, 1.0))):
        return None
    if not np.array_equal(np.abs(matrix).sum(axis=0), np.ones(3)) or not np.array_equal(np.abs(matrix).sum(axis=1), np.ones(3)):
        return None
    lookup = {target: word for word, target in WORD_TO_LPS.items()}
    words = []
    for column in matrix.T:
        axis = int(np.flatnonzero(column)[0])
        words.append(lookup[(axis, float(column[axis]))])
    return tuple(words)


def frame_axis(meta) -> Tuple[Optional[int], Optional[Tuple[str, ...]], List[str]]:
    """(array index, labels) of the component axis whose labels name a patient frame, and the names of
    component axes whose frame is unknown (no patient-direction labels)."""
    found, words, unknown = None, None, []
    for index, axis in enumerate(meta.axes):
        if axis.type != "component":
            continue
        labels = tuple(axis.labels) if axis.labels is not None else None
        if labels is not None and basis(labels) is not None and found is None:
            found, words = index, labels
        else:
            unknown.append(axis.name)
    return found, words, unknown


def report_unknown(names: Sequence[str], report) -> None:
    for name in names:
        report.warn(f"component axis {name!r} has no frame labels: the frame of its components is unknown")


def with_labels(axes: List[Dict], index: int, words: Sequence[str]) -> List[Dict]:
    """Axis dicts with ``labels`` set on axis ``index``."""
    out = [dict(axis) for axis in axes]
    out[index]["labels"] = list(words)
    return out


def undeclared(format_name: str, why: str) -> VectorFrameError:
    return VectorFrameError(f"{format_name}: {why}. Declare the frame of the vector components with policy "
                            f"vector_frame={POLICY_OPTIONS} (vector_frame='from_file' needs a file that declares it).")


def transform(array: np.ndarray, axis: int, matrix: np.ndarray) -> np.ndarray:
    """``matrix @ v`` for the vector ``v`` along ``axis`` of every element; floating data only, dtype kept."""
    if array.dtype.kind != "f":
        raise VectorFrameError(f"vector components of dtype {array.dtype.name} cannot be rotated without changing "
                               "the dtype; convert the data to floating point first")
    moved = np.moveaxis(np.asarray(array, dtype=np.float64), axis, -1)
    out = moved @ np.asarray(matrix, dtype=np.float64).T
    return np.ascontiguousarray(np.moveaxis(out, -1, axis).astype(array.dtype, copy=False))


def permute(array: np.ndarray, axis: int, source: Sequence[str], target: Sequence[str]) -> np.ndarray:
    """Components in frame ``source`` re-expressed in frame ``target`` (both patient frames): a signed
    permutation, exact for any signed dtype and float data."""
    mapping = basis(target).T @ basis(source)          # target coordinates of the source components
    rows = [int(np.flatnonzero(row)[0]) for row in mapping]
    signs = [float(mapping[r, c]) for r, c in enumerate(rows)]
    if any(s < 0 for s in signs) and array.dtype.kind == "u":
        raise VectorFrameError(f"unsigned components ({array.dtype.name}) cannot change sign; "
                               "convert the data to a signed or floating type first")
    if array.dtype.kind == "i":
        low = np.iinfo(array.dtype).min
        for c, s in zip(rows, signs):
            if s < 0 and np.any(np.take(array, c, axis=axis) == low):
                raise VectorFrameError(f"component {c} holds {low}, whose negation does not fit {array.dtype.name}")
    parts = []
    for c, s in zip(rows, signs):
        part = np.take(array, [c], axis=axis)
        parts.append(-part if s < 0 else part)
    return np.ascontiguousarray(np.concatenate(parts, axis=axis))
