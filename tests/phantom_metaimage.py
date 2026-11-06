"""MetaImage conformance phantoms (spec section 9.1), written and decoded by hand, never with MedVol.

The conventions follow ITK as verified in the orientation KB (ch04 section 3, experiment
``experiments/04_formats/e02_metaimage.py``):

- world frame LPS; ``Offset`` (or ``Origin``/``Position``) is the centre of voxel 0;
- ``TransformMatrix`` lists one axis direction per group of NDims numbers (``D`` column-major);
- ``ElementSpacing`` is the spacing per axis; DimSize and the data run i fastest;
- ``ElementNumberOfChannels`` values are interleaved per voxel; ``CompressedData`` is one zlib stream.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from phantom import Truth

DTYPES = {"uint8": "MET_UCHAR", "int16": "MET_SHORT", "uint16": "MET_USHORT", "int32": "MET_INT",
          "float32": "MET_FLOAT", "float64": "MET_DOUBLE", "int64": "MET_LONG_LONG"}
ELEMENT_TYPES = {v: k for k, v in DTYPES.items()}


def _numbers(values) -> str:
    return " ".join(f"{float(v):.17g}" for v in np.ravel(values))


def geometry_lines(truth: Truth, ndims: int = 3) -> Dict[str, str]:
    """Offset, TransformMatrix and ElementSpacing of the truth (non-space dimensions: identity, 1)."""
    linear = truth.affine_lps[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    direction = np.eye(ndims)
    direction[:3, :3] = linear / spacing
    origin = np.zeros(ndims)
    origin[:3] = truth.affine_lps[:3, 3]
    return {
        "TransformMatrix": _numbers(direction.T),       # rows of D^T = one axis per group
        "Offset": _numbers(origin),
        "ElementSpacing": _numbers(np.concatenate([spacing, np.ones(ndims - 3)])),
    }


def write_metaimage(path, truth: Truth = None, channels: bool = False, compressed: bool = False, msb: bool = False,
                    origins: Optional[Sequence[Tuple[str, Sequence[float]]]] = None,
                    direction_key: str = "TransformMatrix", row_major: bool = False,
                    extra: Sequence[Tuple[str, str]] = (), data_file: Optional[str] = None):
    """A MetaImage file of ``truth`` (``.mha`` inline, ``.mhd`` with a separate data file).

    - ``channels``: the last dimension of ``truth.shape`` is written as interleaved channels;
    - ``origins``: the origin field(s) as ``[(key, xyz), ...]`` in line order (default ``Offset``);
    - ``direction_key``: ``TransformMatrix``, ``Rotation`` or ``Orientation``;
    - ``row_major``: write ``D`` row-major (the KB pitfall P13; ITK then reads ``D^T``);
    - ``extra``: further ``(key, text)`` lines, inserted before ``ElementSpacing``.
    """
    truth = truth or Truth()
    path = Path(path)
    volume = truth.volume()                           # file order [i, j, k(, t | c)]
    ndims = volume.ndim - 1 if channels else volume.ndim
    geometry = geometry_lines(truth, ndims)
    if row_major:
        linear = truth.affine_lps[:3, :3]
        geometry["TransformMatrix"] = _numbers(linear / np.linalg.norm(linear, axis=0))
    if channels:
        payload_array = np.moveaxis(volume, -1, 0)    # channel fastest: Fortran order [c, i, j, k]
    else:
        payload_array = volume
    dtype = np.dtype(truth.dtype).newbyteorder(">" if msb else "<")
    payload = np.asarray(payload_array, dtype=dtype).tobytes(order="F")
    if compressed:
        payload = zlib.compress(payload)

    lines: List[Tuple[str, str]] = [
        ("ObjectType", "Image"), ("NDims", str(ndims)), ("BinaryData", "True"),
        ("BinaryDataByteOrderMSB", str(bool(msb))), ("CompressedData", str(bool(compressed))),
    ]
    if compressed:
        lines.append(("CompressedDataSize", str(len(payload))))
    lines.append((direction_key, geometry["TransformMatrix"]))
    for key, xyz in origins or [("Offset", None)]:
        lines.append((key, geometry["Offset"] if xyz is None else _numbers(xyz)))
    lines.append(("CenterOfRotation", "0 0 0" if ndims == 3 else " ".join(["0"] * ndims)))
    lines += list(extra)
    lines.append(("ElementSpacing", geometry["ElementSpacing"]))
    lines.append(("DimSize", " ".join(str(n) for n in volume.shape[:ndims])))
    if channels:
        lines.append(("ElementNumberOfChannels", str(volume.shape[-1])))
    lines.append(("ElementType", DTYPES[np.dtype(truth.dtype).name]))
    if path.suffix.lower() == ".mhd":
        name = data_file or path.stem + (".zraw" if compressed else ".raw")
        lines.append(("ElementDataFile", name))
        path.write_text("".join(f"{k} = {v}\n" for k, v in lines))
        (path.parent / name).write_bytes(payload)
    else:
        lines.append(("ElementDataFile", "LOCAL"))
        path.write_bytes("".join(f"{k} = {v}\n" for k, v in lines).encode() + payload)
    return path


def read_metaimage(path) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Header lines and the array in file order ([i, j, k(, ...)][, c] last), decoded independently."""
    path = Path(path)
    raw = path.read_bytes()
    lines: List[Tuple[str, str]] = []
    position = 0
    while True:
        end = raw.index(b"\n", position)
        key, _, value = raw[position:end].decode().partition("=")
        lines.append((key.strip(), value.strip()))
        position = end + 1
        if key.strip() == "ElementDataFile":
            break
    header = dict(lines)
    dims = [int(v) for v in header["DimSize"].split()]
    channels = int(header.get("ElementNumberOfChannels", "1"))
    payload = raw[position:] if header["ElementDataFile"] == "LOCAL" else (path.parent / header["ElementDataFile"]).read_bytes()
    if header.get("CompressedData") == "True":
        payload = zlib.decompress(payload)
    dtype = np.dtype(ELEMENT_TYPES[header["ElementType"]]).newbyteorder(">" if header.get("BinaryDataByteOrderMSB") == "True" else "<")
    flat = np.frombuffer(payload, dtype=dtype, count=int(np.prod(dims)) * channels)
    if channels > 1:
        return lines, np.moveaxis(flat.reshape([channels] + dims, order="F"), 0, -1)
    return lines, flat.reshape(dims, order="F")


def index_to_lps(lines: Sequence[Tuple[str, str]]) -> np.ndarray:
    """4 x 4 file index (i, j, k) -> LPS matrix from a 3-D header, using ITK's rules."""
    header = dict(lines)
    origin_key = [k for k in ("Position", "Offset", "Origin") if k in header][-1]
    direction_key = [k for k in ("Orientation", "Rotation", "TransformMatrix") if k in header][-1]
    origin = np.array(header[origin_key].split(), dtype=float)[:3]
    ndims = int(header["NDims"])
    direction = np.array(header[direction_key].split(), dtype=float).reshape(ndims, ndims).T[:3, :3]
    spacing = np.array(header["ElementSpacing"].split(), dtype=float)[:3]
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    return affine


def file_world_error(path, truth: Truth) -> float:
    """Largest distance (mm) between where the file (read by ITK's rules) puts each voxel and the truth."""
    lines, values = read_metaimage(path)
    affine = index_to_lps(lines)
    index = np.indices(values.shape).reshape(values.ndim, -1)
    world = (affine @ np.vstack([index[:3], np.ones(index.shape[1])]))[:3].T
    return float(np.max(np.linalg.norm(world - truth.world_of_values(values.reshape(-1)), axis=1)))
