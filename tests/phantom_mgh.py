"""MGH/MGZ files written byte by byte (independent of MedVol and nibabel's header objects).

Needed for header variants nibabel cannot write faithfully: ``goodRASFlag = 0`` with arbitrary
stored geometry, a missing footer, scan parameters, tags and non-zero bytes in the unused area.
"""

from __future__ import annotations

import gzip
import struct
from typing import Optional, Sequence

import numpy as np

from phantom import FLIP4, Truth

TYPES = {"uint8": 0, "int32": 1, "float32": 3, "int16": 4, "uint16": 10}


def fields_from_truth(truth: Truth):
    """``delta``, ``Mdc`` (rows = axes) and ``Pxyz_c`` (world at index N/2) of a truth grid."""
    ras = FLIP4 @ truth.affine_lps
    delta = np.linalg.norm(ras[:3, :3], axis=0)
    mdc = (ras[:3, :3] / delta).T
    c_ras = (ras @ np.r_[np.asarray(truth.shape[:3], float) / 2.0, 1.0])[:3]
    return delta, mdc, c_ras


def write_raw(path, data: np.ndarray, delta, mdc, c_ras, good_ras: int = 1, dof: int = 0,
              footer: Optional[Sequence[float]] = (0.0, 0.0, 0.0, 0.0, 0.0), tags: bytes = b"", unused: bytes = b""):
    """An MGH (``.mgz``: gzip) file from its fields; ``data`` in MGH order ``[i, j, k(, frame)]``."""
    dims = list(data.shape) + [1] * (4 - data.ndim)
    head = struct.pack(">i4iii", 1, *dims, TYPES[data.dtype.name], dof)
    head += struct.pack(">h", good_ras)
    head += struct.pack(">3f", *delta) + struct.pack(">9f", *np.asarray(mdc).reshape(-1)) + struct.pack(">3f", *c_ras)
    head += unused + b"\x00" * (284 - len(head) - len(unused))
    body = np.asarray(data).astype(data.dtype.newbyteorder(">")).tobytes(order="F")
    tail = (struct.pack(">5f", *footer) if footer is not None else b"") + tags
    blob = head + body + tail
    opener = gzip.open if str(path).endswith(".mgz") else open
    with opener(str(path), "wb") as handle:
        handle.write(blob)
    return path


def write_truth(path, truth: Truth = None, dtype: str = "float32", **kwargs):
    """The phantom ``truth`` with its own geometry fields."""
    truth = truth or Truth()
    delta, mdc, c_ras = fields_from_truth(truth)
    return write_raw(path, truth.volume().astype(dtype), delta, mdc, c_ras, **kwargs)
