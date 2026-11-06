"""MetaImage plugin (spec section 7.3): ``.mha`` (header + data) and ``.mhd`` (header) + raw data.

MetaImage is ITK's own text-header format. MedVol parses and writes it itself; there is no backend.

- World frame: ITK's LPS. The geometry lives only in the origin (``Origin``/``Offset``/``Position``),
  the direction (``TransformMatrix``/``Rotation``/``Orientation``) and ``ElementSpacing``.
- ``TransformMatrix`` holds one axis direction per group of NDims numbers, i.e. the direction matrix
  ``D`` (columns = axes) in column-major order (KB ch04 section 3.2, e02 section A).
- Synonyms are read in ITK's fixed order, the last one present wins: Position -> Offset -> Origin
  and Orientation -> Rotation -> TransformMatrix (KB ch04 section 3.3, e02 sections B/C).
  ``ElementSize`` is the spacing only when ``ElementSpacing`` is absent (e02 section E).
- ``AnatomicalOrientation`` and ``CenterOfRotation`` are kept verbatim and never used for geometry: ITK
  image IO never reads them (KB ch04 section 3.4, e02 sections D/F).

Array order: the MLArray array is the file array with its axes reversed (C order, last axis = the
fastest on disk). With ``ElementNumberOfChannels > 1`` the interleaved channel is the very last axis.
"""

from __future__ import annotations

import os
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mlarray import Meta, fingerprint
from mlarray.metadata import thawed

from medvol.geometry import GeometryError, has_shear, lps_matrix, patient_space, world_corners
from medvol.image import Image
from medvol.plugins.base import Capabilities, FieldTable, FormatPlugin, PathLike
from medvol.policy import Policy
from medvol.report import Report

#: ElementType -> numpy dtype. MetaIO's MET_LONG/MET_ULONG are 4 bytes (ITK writes 64-bit as MET_LONG_LONG).
ELEMENT_TYPES = {
    "MET_CHAR": "int8", "MET_UCHAR": "uint8", "MET_SHORT": "int16", "MET_USHORT": "uint16",
    "MET_INT": "int32", "MET_UINT": "uint32", "MET_LONG": "int32", "MET_ULONG": "uint32",
    "MET_LONG_LONG": "int64", "MET_ULONG_LONG": "uint64", "MET_FLOAT": "float32", "MET_DOUBLE": "float64",
}
#: numpy dtype -> the ElementType ITK writes
DTYPE_ELEMENT_TYPES = {
    "int8": "MET_CHAR", "uint8": "MET_UCHAR", "int16": "MET_SHORT", "uint16": "MET_USHORT",
    "int32": "MET_INT", "uint32": "MET_UINT", "int64": "MET_LONG_LONG", "uint64": "MET_ULONG_LONG",
    "float32": "MET_FLOAT", "float64": "MET_DOUBLE",
}

#: Synonyms in ITK's (MetaIO's) reading order: the last one present wins.
ORIGIN_KEYS = ("Position", "Offset", "Origin")
DIRECTION_KEYS = ("Orientation", "Rotation", "TransformMatrix")
BYTE_ORDER_KEYS = ("BinaryDataByteOrderMSB", "ElementByteOrderMSB")

#: Value types of the standard MetaIO fields; any other key is kept as text.
FIELD_TYPES = {
    "ObjectType": "str", "ObjectSubType": "str", "Comment": "str", "Name": "str", "TransformType": "str",
    "AnatomicalOrientation": "str", "DistanceUnits": "str", "Modality": "str", "ElementType": "str",
    "ElementDataFile": "str",
    "NDims": "int", "ID": "int", "ParentID": "int", "CompressedDataSize": "int", "HeaderSize": "int",
    "ElementNumberOfChannels": "int", "DimSize": "ints",
    "Position": "floats", "Offset": "floats", "Origin": "floats", "Orientation": "floats", "Rotation": "floats",
    "TransformMatrix": "floats", "CenterOfRotation": "floats", "ElementSpacing": "floats",
    "ElementSize": "floats", "Color": "floats", "SequenceID": "floats",
    "ElementMin": "float", "ElementMax": "float",
    "BinaryData": "bool", "BinaryDataByteOrderMSB": "bool", "ElementByteOrderMSB": "bool", "CompressedData": "bool",
}

#: Header fields that encode the geometry; reused verbatim while the geometry is unchanged.
GEOMETRY_FIELDS = ORIGIN_KEYS + DIRECTION_KEYS + ("ElementSpacing", "ElementSize")
#: Fields derived from the array and the file layout; always regenerated.
DERIVED_FIELDS = ("NDims", "DimSize", "ElementType", "ElementNumberOfChannels", "ElementDataFile",
                  "CompressedDataSize", "HeaderSize")
#: Encoding fields: reused under policy ``layout='source'``, else the defaults.
LAYOUT_FIELDS = ("BinaryData",) + BYTE_ORDER_KEYS + ("CompressedData",)

#: Order in which ITK writes the fields; used to place fields the source header does not have.
CANONICAL_ORDER = ("ObjectType", "NDims", "BinaryData", "BinaryDataByteOrderMSB", "CompressedData",
                   "CompressedDataSize", "TransformMatrix", "Offset", "CenterOfRotation", "AnatomicalOrientation",
                   "ElementSpacing", "DimSize", "ElementNumberOfChannels", "ElementType", "ElementDataFile")

KNOWN_FIRST_KEYS = frozenset(FIELD_TYPES)


# ------------------------------------------------------------------------------------------------
# header text <-> readable header (MLArray spec 6.4)
# ------------------------------------------------------------------------------------------------

def _parse_value(key: str, text: str) -> Any:
    """A header value as a native value: numbers for the standard numeric fields, text otherwise."""
    kind = FIELD_TYPES.get(key, "str")
    if kind == "str":
        return text
    if kind == "bool":
        return text[:1] in ("T", "t", "1")        # as MetaIO reads booleans
    tokens = text.split()
    if kind == "int":
        return int(tokens[0])
    if kind == "ints":
        return [int(token) for token in tokens]
    if kind == "float":
        return float(tokens[0])
    return [float(token) for token in tokens]


def format_number(value: Any) -> str:
    """Shortest text that reads back as the same number (``1.0`` is written ``1``, as ITK does)."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(format_number(item) for item in value)
    return format_number(value)


def parse_header(raw: bytes) -> Tuple[List[Tuple[str, str]], int, List[str]]:
    """``Key = value`` pairs up to ``ElementDataFile``, the byte offset after that line, and (for
    ``ElementDataFile = LIST``) the data file names listed after it."""
    pairs: List[Tuple[str, str]] = []
    position = 0
    while position < len(raw):
        end = raw.find(b"\n", position)
        end = len(raw) if end < 0 else end
        line = raw[position:end].decode("latin-1").rstrip("\r")
        position = end + 1
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Not a MetaImage header line: {line[:80]!r}")
        key, value = key.strip(), value.strip()
        pairs.append((key, value))
        if key == "ElementDataFile":
            listed: List[str] = []
            if value.split()[:1] == ["LIST"]:
                listed = [name.strip() for name in raw[position:].decode("latin-1").splitlines() if name.strip()]
            return pairs, min(position, len(raw)), listed
    raise ValueError("MetaImage header has no ElementDataFile field.")


def header_to_dict(pairs: Sequence[Tuple[str, str]], report: Report) -> Dict[str, Any]:
    header: Dict[str, Any] = {}
    for key, text in pairs:
        try:
            value = _parse_value(key, text)
        except (ValueError, IndexError):
            report.warn(f"MetaImage field {key} = {text!r} is not of the expected type; kept as text")
            value = text
        if key in header:
            report.warn(f"MetaImage field {key} appears twice; the last value is used")
        header[key] = value
    return header


def header_to_text(header: Dict[str, Any]) -> bytes:
    return "".join(f"{key} = {format_value(value)}\n" for key, value in header.items()).encode("latin-1")


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------

def _inflate(payload: bytes) -> bytes:
    """zlib (as MetaIO writes) or gzip stream."""
    return zlib.decompressobj(wbits=47).decompress(payload)


def _data_bytes(blob: bytes, header: Dict[str, Any], expected: int, local: bool) -> bytes:
    """The element bytes of one data file (or of the data after an .mha header)."""
    if header.get("CompressedData", False):
        size = header.get("CompressedDataSize")
        if local and isinstance(size, int) and size > 0:
            blob = blob[:size]
        return _inflate(blob)
    skip = header.get("HeaderSize", 0) if not local else 0
    if skip == -1:                       # "the last prod(DimSize) bytes of the file"
        return blob[len(blob) - expected:]
    return blob[skip:]


def _decode(blob: bytes, dtype: np.dtype, count: int, binary: bool, msb: bool) -> np.ndarray:
    if not binary:
        tokens = blob.decode("latin-1").split()
        if len(tokens) < count:
            raise ValueError(f"MetaImage ASCII data holds {len(tokens)} values, expected {count}.")
        return np.array(tokens[:count]).astype(dtype)
    stored = dtype.newbyteorder(">" if msb else "<")
    if len(blob) < count * dtype.itemsize:
        raise ValueError(f"MetaImage data holds {len(blob)} bytes, expected {count * dtype.itemsize}.")
    return np.frombuffer(blob, dtype=stored, count=count).astype(dtype)


def _encode(data: np.ndarray, binary: bool, msb: bool) -> bytes:
    """Element bytes in C order of ``data`` (file order, fastest last)."""
    if not binary:
        row = data.shape[-1] if data.ndim else 1
        flat = [str(value) for value in data.reshape(-1)]
        return "".join(" ".join(flat[n:n + row]) + "\n" for n in range(0, len(flat), row)).encode("latin-1")
    return np.ascontiguousarray(data, dtype=data.dtype.newbyteorder(">" if msb else "<")).tobytes()


# ------------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------------

def _winner(header: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    present = [key for key in keys if key in header]
    return present[-1] if present else None


def _vector(header: Dict[str, Any], key: str, length: int) -> np.ndarray:
    value = header[key]
    values = np.asarray(value if isinstance(value, (list, tuple)) else [value], dtype=np.float64)
    if values.size != length:
        raise GeometryError(f"MetaImage {key} has {values.size} values; NDims requires {length}.")
    return values


def geometry_from_header(header: Dict[str, Any], ndims: int, report: Report) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction ``D`` (columns = axes), spacing and origin in MetaImage world (LPS), as ITK reads them."""
    for keys, what in ((ORIGIN_KEYS, "origin"), (DIRECTION_KEYS, "direction")):
        present = [key for key in keys if key in header]
        if len(present) > 1:
            values = [np.asarray(header[key], dtype=np.float64).ravel() for key in present]
            if any(v.shape != values[-1].shape or not np.array_equal(v, values[-1]) for v in values):
                report.warn(f"MetaImage {what} fields {present} disagree; {present[-1]} is used, as in ITK")

    origin_key = _winner(header, ORIGIN_KEYS)
    origin = _vector(header, origin_key, ndims) if origin_key else np.zeros(ndims)
    direction_key = _winner(header, DIRECTION_KEYS)
    # one axis direction per group of ndims numbers: the column-major direction matrix
    direction = _vector(header, direction_key, ndims * ndims).reshape(ndims, ndims).T if direction_key else np.eye(ndims)
    spacing_key = "ElementSpacing" if "ElementSpacing" in header else ("ElementSize" if "ElementSize" in header else None)
    spacing = _vector(header, spacing_key, ndims) if spacing_key else np.ones(ndims)
    report.choose("metaimage_geometry", {"origin": origin_key or "none (0)", "direction": direction_key or "none (identity)",
                                         "spacing": spacing_key or "none (1)"})
    return direction, spacing, origin


def _comes_from_code(direction: np.ndarray) -> str:
    """The AnatomicalOrientation ITK writes for a 3-D direction: closest axes, comes-from letters."""
    letters = {0: ("R", "L"), 1: ("A", "P"), 2: ("I", "S")}   # (comes-from of +axis, of -axis)
    code = ""
    for column in direction.T:
        axis = int(np.argmax(np.abs(column)))
        code += letters[axis][0 if column[axis] > 0 else 1]
    return code


# ------------------------------------------------------------------------------------------------
# plugin
# ------------------------------------------------------------------------------------------------

class MetaImagePlugin(FormatPlugin):
    name = "metaimage"
    extensions = (".mha", ".mhd")
    capabilities = Capabilities(
        max_space_axes=3, min_space_axes=1, max_ndim=11,
        non_space_axes=frozenset({"time", "channel", "component", "other"}),
        shear=False, oblique=True, label_table=False, key_value=False,
        dtypes=frozenset(DTYPE_ELEMENT_TYPES),
    )
    fields = FieldTable(
        core_owned=frozenset(GEOMETRY_FIELDS + DERIVED_FIELDS),
        data_dependent=frozenset({"ElementMin", "ElementMax"}),
    )

    # -- sniff -------------------------------------------------------------------------------

    def sniff(self, path: PathLike) -> float:
        if not self.matches_extension(path) or not os.path.isfile(str(path)):
            return 0.0
        try:
            with open(path, "rb") as handle:
                head = handle.read(4096)
        except OSError:
            return 0.0
        for line in head.split(b"\n"):
            if line.strip():
                key = line.decode("latin-1").partition("=")[0].strip()
                return 1.0 if b"=" in line and key in KNOWN_FIRST_KEYS else 0.0
        return 0.0

    # -- read --------------------------------------------------------------------------------

    def read(self, path: PathLike, policy: Policy) -> Tuple[List[Image], Report]:
        report = Report("read", self.name, str(path))
        path = str(path)
        with open(path, "rb") as handle:
            raw = handle.read()
        pairs, offset, listed = parse_header(raw)
        header = header_to_dict(pairs, report)

        ndims = header.get("NDims")
        dims = header.get("DimSize")
        element_type = header.get("ElementType")
        if not isinstance(ndims, int) or not isinstance(dims, list) or len(dims) != ndims:
            raise ValueError(f"{path}: NDims and DimSize are missing or inconsistent.")
        element_type = str(element_type).replace("_ARRAY", "")
        if element_type not in ELEMENT_TYPES:
            raise ValueError(f"{path}: unsupported ElementType {header.get('ElementType')!r}.")
        dtype = np.dtype(ELEMENT_TYPES[element_type])
        channels = int(header.get("ElementNumberOfChannels", 1))
        units = str(header.get("DistanceUnits", "mm"))
        if units.lower() != "mm":
            report.warn(f"MetaImage DistanceUnits is {units!r}; geometry is stored in millimeter as written")

        array, files = self._read_array(path, raw, offset, listed, header, dims, channels, dtype)
        direction, spacing, origin = geometry_from_header(header, ndims, report)
        axes, space = self._axes_and_space(direction, spacing, origin, ndims, channels, policy, report)

        values = {"kind": "measurement"}
        n_axes = ndims + (channels > 1)
        meta = Meta(axes=axes, spaces=[space], values=values, shape=array.shape)
        source = {
            "format": self.name,
            "files": [os.path.basename(path)] + files,
            "reader": {"name": "medvol"},
            # source index order: DimSize order (i, j, k, ...), then the interleaved channel
            "index_map": {"axis_order": list(range(ndims))[::-1] + ([ndims] if channels > 1 else []),
                          "flipped": [False] * n_axes, "offset": [0] * n_axes},
            "fingerprint": fingerprint.compute(array, meta),
            "header": header,
        }
        meta = Meta(axes=axes, spaces=[space], values=values, source=[source], shape=array.shape)
        return [Image(array, meta)], report

    @staticmethod
    def _read_array(path: str, raw: bytes, offset: int, listed: List[str], header: Dict[str, Any], dims: List[int],
                    channels: int, dtype: np.dtype) -> Tuple[np.ndarray, List[str]]:
        """The array in MLArray order (``[..., k, j, i(, c)]``) and the names of separate data files."""
        count = int(np.prod(dims)) * channels
        binary = bool(header.get("BinaryData", True))
        msb = bool(header.get(_winner(header, BYTE_ORDER_KEYS) or "", False))
        data_file = str(header["ElementDataFile"]).strip()
        directory = os.path.dirname(path)
        expected = count * dtype.itemsize
        if data_file == "LOCAL":
            blob, files = _data_bytes(raw[offset:], header, expected, local=True), []
        elif data_file.split()[0] == "LIST":
            tokens = data_file.split()
            file_ndims = int(tokens[1].rstrip("Dd")) if len(tokens) > 1 else len(dims) - 1
            per_file = int(np.prod(dims[:file_ndims])) * channels * dtype.itemsize
            if len(listed) * per_file < expected:
                raise ValueError(f"{path}: ElementDataFile LIST names {len(listed)} files, too few for DimSize {dims}.")
            parts = []
            for name in listed:
                with open(os.path.join(directory, name), "rb") as handle:
                    parts.append(_data_bytes(handle.read(), header, per_file, local=False))
            blob, files = b"".join(parts), list(listed)
        elif "%" in data_file:
            raise ValueError(f"{path}: ElementDataFile file-name patterns ({data_file!r}) are not supported.")
        else:
            with open(os.path.join(directory, data_file), "rb") as handle:
                blob = _data_bytes(handle.read(), header, expected, local=False)
            files = [data_file]
        flat = _decode(blob, dtype, count, binary, msb)
        shape = tuple(dims[::-1]) + ((channels,) if channels > 1 else ())
        return flat.reshape(shape), files

    @staticmethod
    def _axes_and_space(direction: np.ndarray, spacing: np.ndarray, origin: np.ndarray, ndims: int, channels: int,
                        policy: Policy, report: Report) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Axes in MLArray order and the patient space; non-space axes never enter the space matrix."""
        n_space = min(ndims, 3)
        if ndims > n_space:
            coupled = np.any(direction[:n_space, n_space:] != 0) or np.any(direction[n_space:, :n_space] != 0)
            extra = direction[n_space:, n_space:]
            if coupled or np.any(extra != np.diag(np.diag(extra))):
                raise GeometryError("MetaImage TransformMatrix couples the space axes with axes beyond the third; "
                                    "this geometry has no separate space and non-space axes.")
        file_axes: List[Dict[str, Any]] = [{"name": name, "type": "space"} for name in ("x", "y", "z")[:n_space]]
        for d in range(n_space, ndims):
            axis = {"name": "t" if d == 3 else f"u{d - 3}", "type": "time" if d == 3 else "other",
                    "start": float(origin[d]), "step": float(spacing[d] * direction[d, d])}
            file_axes.append(axis)
        if ndims > 3:
            report.choose("metaimage_axis_4", "time (MetaImage has no axis kinds; dimension 4 is read as time)")
        axes = file_axes[::-1]
        if channels > 1:
            axes.append({"name": "c", "type": "channel"})
            report.choose("metaimage_channels", f"{channels} interleaved channels -> last axis 'c' (channel)")

        matrix_lps = np.zeros((4, n_space + 1))
        matrix_lps[:n_space, :n_space] = direction[:n_space, :n_space] * spacing[:n_space]
        matrix_lps[:n_space, n_space] = origin[:n_space]
        matrix_lps[3, n_space] = 1.0
        space = patient_space(matrix_lps, [a["name"] for a in file_axes[:n_space]], policy.world_frame)
        return axes, space

    # -- write -------------------------------------------------------------------------------

    def write(self, image: Image, path: PathLike, policy: Policy, **options) -> Report:
        report = Report("write", self.name, str(path))
        meta = image.meta
        path = str(path)
        source = next((entry for entry in meta.source if entry.format == self.name), None)
        original: Dict[str, Any] = thawed(source.header) if source is not None else {}

        order, file_axes, channel_axis = self._layout(meta, report)
        data = np.transpose(image.array, order)               # C order [..., k, j, i(, c)]
        dims = list(data.shape[:-1] if channel_axis is not None else data.shape)[::-1]
        channels = data.shape[-1] if channel_axis is not None else 1

        values: Dict[str, Any] = dict(original)
        if source is None:
            values["ObjectType"] = "Image"
        values["NDims"] = len(dims)
        values["DimSize"] = dims
        if channels > 1 or "ElementNumberOfChannels" in original:
            values["ElementNumberOfChannels"] = channels
        values["ElementType"] = self._element_type(data.dtype, original.get("ElementType"))

        if source is not None and self._geometry_unchanged(source, image):
            report.choose("geometry", "reused from source (unchanged)")
        else:
            self._set_geometry(values, original, meta, file_axes, report, policy)
        self._check_axis_fields(values, len(dims), report)
        self._set_values(meta, report, policy)
        self._data_dependent(values, source, image, report)

        binary, msb, compressed = self._encoding(original, values, policy, report)
        self._write_files(values, original, data, path, binary, msb, compressed, report)
        return report

    @staticmethod
    def _layout(meta: Meta, report: Report) -> Tuple[List[int], List[Any], Optional[int]]:
        """C-order axis permutation for the file, the file axes (fastest first) and the channel axis.

        File dimensions: the space axes (storage order kept: MLArray order reversed), then time, then
        the rest. One channel or component axis becomes the interleaved, fastest-varying channel.
        """
        axes = list(meta.axes)
        space = meta.spaces[0] if meta.spaces else None
        mapped = list(space.from_array.array_axes) if space is not None else [a.name for a in axes if a.type == "space"]
        channel_like = [i for i, a in enumerate(axes) if a.type in ("channel", "component") and a.name not in mapped]
        channel = channel_like[-1] if channel_like else None
        if channel is not None and channel != len(axes) - 1:
            report.warn(f"axis {axes[channel].name!r} is written as the interleaved channel, the fastest axis on disk")
        if len(channel_like) > 1:
            report.warn(f"MetaImage interleaves one channel axis; axes {[axes[i].name for i in channel_like[:-1]]} "
                        "are written as image dimensions")
        space_idx = [i for i, a in enumerate(axes) if a.name in mapped][::-1]
        time_idx = [i for i, a in enumerate(axes) if a.type == "time" and i != channel]
        rest = [i for i in range(len(axes)) if i not in space_idx and i not in time_idx and i != channel]
        file_dims = space_idx + time_idx + rest[::-1]
        order = file_dims[::-1] + ([channel] if channel is not None else [])
        if channel is not None and meta.shape is not None and meta.shape[channel] == 1:
            report.warn(f"channel axis {axes[channel].name!r} has length 1; MetaImage stores it as a scalar image")
        return order, [axes[i] for i in file_dims], channel

    @staticmethod
    def _element_type(dtype: np.dtype, original: Optional[str]) -> str:
        if original is not None and ELEMENT_TYPES.get(str(original).replace("_ARRAY", "")) == dtype.name:
            return original                       # e.g. MET_LONG for int32 stays MET_LONG
        return DTYPE_ELEMENT_TYPES[dtype.name]

    @staticmethod
    def _geometry_unchanged(source, image: Image) -> bool:
        fp = source.fingerprint or {}
        return (fp.get("geometry") == fingerprint.geometry_hash(image.meta)
                and list(fp.get("shape", [])) == list(image.array.shape))

    @staticmethod
    def _set_geometry(values: Dict[str, Any], original: Dict[str, Any], meta: Meta, file_axes: List[Any],
                      report: Report, policy: Policy) -> None:
        """Regenerate origin, direction and spacing from the core into every synonym the source used."""
        n = len(file_axes)
        space = meta.spaces[0] if meta.spaces else None
        mapped = list(space.from_array.array_axes) if space is not None else []
        n_space = sum(1 for a in file_axes if a.name in mapped)
        direction, spacing, origin = np.eye(n), np.ones(n), np.zeros(n)
        if space is None:
            report.warn("no space: written with identity direction, unit spacing and zero origin (index space)")
        else:
            lps = lps_matrix(space)
            columns = [mapped.index(a.name) for a in file_axes[:n_space]]
            linear, translation = lps[:3, columns], lps[:3, -1]
            if n_space < 3:
                if np.any(linear[n_space:] != 0) or np.any(translation[n_space:] != 0):
                    report.drop("geometry", f"a {n_space}-D MetaImage has {n_space} world coordinates; the "
                                f"remaining world coordinates of the grid are lost", policy.on_loss)
                linear, translation = linear[:n_space], translation[:n_space]
            norms = np.linalg.norm(linear, axis=0)
            unit = linear / norms
            if n_space > 1 and has_shear(unit):
                if n_space < 3:
                    report.drop("geometry", "sheared grid; the format stores only orthogonal axes", policy.on_loss)
                u, _, vt = np.linalg.svd(unit)
                rigid = u @ vt                      # closest orthonormal direction
                names = [axis.name for axis in meta.axes]
                sizes = [meta.shape[names.index(a.name)] for a in file_axes[:n_space]]
                exact = np.vstack([np.column_stack([linear, translation]), np.eye(1, n_space + 1, n_space)])
                approx = np.vstack([np.column_stack([rigid * norms, translation]), np.eye(1, n_space + 1, n_space)])
                moved = float(np.max(np.linalg.norm(world_corners(exact, sizes) - world_corners(approx, sizes), axis=1)))
                report.warn(f"sheared grid: written with the closest orthonormal direction; voxels move by up to {moved:.4g} mm")
                unit = rigid
            direction[:n_space, :n_space] = unit
            spacing[:n_space] = norms
            origin[:n_space] = translation
        for d in range(n_space, n):
            axis = file_axes[d]
            if axis.coordinates is not None or axis.labels is not None:
                report.drop(f"axis {axis.name!r} coordinates/labels", "MetaImage axes are uniformly sampled", policy.on_loss)
            if axis.unit is not None:
                report.drop(f"axis {axis.name!r} unit", "MetaImage has no units for non-space axes", policy.on_loss)
            if axis.step:
                spacing[d] = abs(axis.step)
                direction[d, d] = -1.0 if axis.step < 0 else 1.0
            origin[d] = axis.start or 0.0

        flat_direction = [float(v) for v in direction.T.reshape(-1)]   # one axis per group: column-major
        for keys, default, value in ((ORIGIN_KEYS, "Offset", [float(v) for v in origin]),
                                     (DIRECTION_KEYS, "TransformMatrix", flat_direction)):
            for key in [key for key in keys if key in original] or [default]:          # every synonym gets the same value, so no reader can pick a stale one
                values[key] = value
        spacing_list = [float(v) for v in spacing]
        if "ElementSize" in original and "ElementSpacing" not in original:
            values["ElementSize"] = spacing_list          # the source's spacing field
        else:
            values["ElementSpacing"] = spacing_list
            if "ElementSize" in values:
                del values["ElementSize"]
                report.warn("ElementSize (voxel extent) dropped: the geometry changed since it was read")

    @staticmethod
    def _check_axis_fields(values: Dict[str, Any], ndims: int, report: Report) -> None:
        """Verbatim per-axis fields must still fit the dimensionality, and AnatomicalOrientation is checked."""
        center = values.get("CenterOfRotation")
        if center is not None and (not isinstance(center, list) or len(center) != ndims):
            del values["CenterOfRotation"]
            report.warn("CenterOfRotation dropped: it does not match the number of dimensions any more")
        code = values.get("AnatomicalOrientation")
        if code is None:
            return
        if len(str(code)) != ndims:
            del values["AnatomicalOrientation"]
            report.warn("AnatomicalOrientation dropped: it does not match the number of dimensions any more")
            return
        key = _winner(values, DIRECTION_KEYS)
        if ndims == 3 and key is not None:
            expected = _comes_from_code(np.asarray(values[key], dtype=np.float64).reshape(3, 3).T)
            if str(code) != expected and "?" not in str(code):
                report.warn(f"AnatomicalOrientation {code} is kept verbatim, but the direction gives {expected} "
                            "(ITK never reads this field)")

    @staticmethod
    def _set_values(meta: Meta, report: Report, policy: Policy) -> None:
        if meta.values.scale is not None and (meta.values.scale != 1 or (meta.values.offset or 0) != 0):
            report.drop("values.scale/offset", "MetaImage has no value scaling; stored values are written", policy.on_loss)
        if meta.values.unit is not None:
            report.drop("values.unit", "MetaImage has no value unit", policy.on_loss)

    @staticmethod
    def _data_dependent(values: Dict[str, Any], source, image: Image, report: Report) -> None:
        if source is None:
            return
        if fingerprint.matches(source.fingerprint or {}, image.array, image.meta):
            return
        for key in ("ElementMin", "ElementMax"):
            if key in values:
                del values[key]
                report.warn(f"{key} dropped: the data changed since it was read")

    @staticmethod
    def _encoding(original: Dict[str, Any], values: Dict[str, Any], policy: Policy, report: Report) -> Tuple[bool, bool, bool]:
        """Binary/ASCII, byte order and compression: the source's under ``layout='source'``."""
        reuse = policy.layout == "source" and bool(original)
        binary = bool(original.get("BinaryData", True)) if reuse else True
        msb = bool(original.get(_winner(original, BYTE_ORDER_KEYS) or "", False)) if reuse else False
        compressed = bool(original.get("CompressedData", False)) if reuse else False
        if not binary and compressed:
            compressed = False
        # a new file gets ITK's full set; a source header gets no field it did not have unless needed
        new = not original
        if new or "BinaryData" in original or not binary:
            values["BinaryData"] = binary
        for key in [key for key in BYTE_ORDER_KEYS if key in original] or (["BinaryDataByteOrderMSB"] if new or msb else []):
            values[key] = msb
        if new or "CompressedData" in original or compressed:
            values["CompressedData"] = compressed
        report.choose("layout", {"binary": binary, "msb": msb, "compressed": compressed})
        return binary, msb, compressed

    def _write_files(self, values: Dict[str, Any], original: Dict[str, Any], data: np.ndarray, path: str,
                     binary: bool, msb: bool, compressed: bool, report: Report) -> None:
        directory, base = os.path.split(path)
        stem = base[:-4]
        detached = path.lower().endswith(".mhd")
        source_file = str(original.get("ElementDataFile", "LOCAL")).strip()
        values.pop("CompressedDataSize", None)
        if values.get("HeaderSize", 0) != 0:
            del values["HeaderSize"]
            report.warn("HeaderSize dropped: the data file is written without a leading header")

        listed: List[Tuple[str, bytes]] = []
        if detached and source_file.split()[:1] == ["LIST"]:
            tokens = source_file.split()
            file_ndims = int(tokens[1].rstrip("Dd")) if len(tokens) > 1 else values["NDims"] - 1
            slow = int(np.prod(data.shape[:values["NDims"] - file_ndims]))
            ext = ".zraw" if compressed else ".raw"
            for n, chunk in enumerate(data.reshape((slow,) + data.shape[values["NDims"] - file_ndims:])):
                payload = _encode(chunk, binary, msb)
                listed.append((f"{stem}_{n:04d}{ext}", zlib.compress(payload) if compressed else payload))
            values["ElementDataFile"] = source_file
            if compressed:
                report.warn("CompressedDataSize is not written for ElementDataFile LIST (one stream per file)")
        else:
            payload = _encode(data, binary, msb)
            if compressed:
                payload = zlib.compress(payload)
                values["CompressedDataSize"] = len(payload)
            if detached:
                ext = os.path.splitext(source_file)[1] if source_file not in ("LOCAL",) and "LIST" not in source_file else ""
                ext = ext or (".zraw" if compressed else ".raw")
                values["ElementDataFile"] = stem + ext
            else:
                values["ElementDataFile"] = "LOCAL"

        header = self._assemble(list(original), values)
        text = header_to_text(header)
        if listed:
            text += "".join(f"{name}\n" for name, _ in listed).encode("latin-1")
            with open(path, "wb") as handle:
                handle.write(text)
            for name, payload in listed:
                with open(os.path.join(directory, name), "wb") as handle:
                    handle.write(payload)
        elif detached:
            with open(path, "wb") as handle:
                handle.write(text)
            with open(os.path.join(directory, values["ElementDataFile"]), "wb") as handle:
                handle.write(payload)
        else:
            with open(path, "wb") as handle:
                handle.write(text + payload)

    @staticmethod
    def _assemble(original_order: List[str], values: Dict[str, Any]) -> Dict[str, Any]:
        """Fields in the source's order; new fields where ITK puts them; ElementDataFile last."""
        keys = [key for key in original_order if key in values]
        rank = {key: n for n, key in enumerate(CANONICAL_ORDER)}
        for key in [key for key in CANONICAL_ORDER if key in values and key not in keys] + \
                   [key for key in values if key not in keys and key not in rank]:
            position = 0
            if key in rank:
                for n, existing in enumerate(keys):
                    if existing in rank and rank[existing] < rank[key]:
                        position = n + 1
            else:
                position = len(keys)
            keys.insert(position, key)
        keys.remove("ElementDataFile")
        keys.append("ElementDataFile")
        return {key: values[key] for key in keys}
