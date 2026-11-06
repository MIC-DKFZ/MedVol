"""NRRD plugin (spec section 7.2).

pynrrd parses the header values and does the byte-level data IO, always with ``index_order='F'``.
MedVol interprets ``space``, ``space directions`` and the per-axis fields itself (pynrrd converts
nothing, KB ch04 §2.8) and writes the header text itself, so field order, comments and key-value
pairs survive a round trip (pynrrd's writer reorders fields and adds its own comments).

Array order: the MLArray array is the NRRD array with its axes reversed (C order, last axis = NRRD
axis 0, the fastest). Space axes are named ``x, y, z`` in NRRD axis order, as names only.

Readable header (``source.header``, MLArray spec 6.4)::

    {"magic": "NRRD0005",
     "comments": [" text after '#'", ...],                  # verbatim, in file order
     "fields": {"type": "int", "dimension": 3, ...},        # NRRD fields, file order, native types;
                                                            # ``none`` vectors are None
     "key_values": {"key": "value", ...}}                   # ``key:=value`` pairs, verbatim strings

``space directions`` and ``measurement frame`` are stored as written: one vector per entry, in file
order. For ``measurement frame`` this means entry ``i`` is column ``i`` of the frame matrix (pynrrd's
matrix is the transpose of the spec's, KB ch04 §2.6).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mlarray import Meta, fingerprint

from medvol.geometry import GeometryError, homogeneous, lps_matrix, patient_space
from medvol.image import Image
from medvol.plugins.base import Capabilities, FieldTable, FormatPlugin, PathLike
from medvol.policy import Policy
from medvol.report import Report


def _nrrd():
    import nrrd

    return nrrd


# ------------------------------------------------------------------------------------------------
# NRRD vocabulary
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """A named NRRD space: ``signs`` maps LPS coordinates to the space's coordinates (and back)."""

    name: str              # canonical long name
    signs: np.ndarray = field(compare=False)
    time: bool             # 4th world coordinate is time
    anatomical: bool       # the name defines an anatomical frame


_LPS, _RAS, _LAS = np.eye(3), np.diag([-1.0, -1.0, 1.0]), np.diag([1.0, -1.0, 1.0])

#: NRRD ``space`` (lower case, long and short names) -> Frame.
#: ``scanner-xyz`` and ``3D-*-handed`` have no defined anatomical meaning; ITK and Slicer take their
#: numbers as LPS (KB ch04 §2.2, e01 §A), and so does MedVol (reported on read).
FRAMES: Dict[str, Frame] = {}
for _long, _short, _signs in (("left-posterior-superior", "LPS", _LPS), ("right-anterior-superior", "RAS", _RAS),
                              ("left-anterior-superior", "LAS", _LAS)):
    FRAMES[_long] = FRAMES[_short.lower()] = Frame(_long, _signs, False, True)
    FRAMES[_long + "-time"] = FRAMES[_short.lower() + "t"] = Frame(_long + "-time", _signs, True, True)
for _long in ("scanner-xyz", "3D-right-handed", "3D-left-handed"):
    FRAMES[_long.lower()] = Frame(_long, _LPS, False, False)
    FRAMES[_long.lower() + "-time"] = Frame(_long + "-time", _LPS, True, False)

#: increases_toward words of a stored patient space -> NRRD space that holds its numbers unchanged
WORDS_TO_SPACE = {
    ("left", "posterior", "superior"): "left-posterior-superior",
    ("right", "anterior", "superior"): "right-anterior-superior",
    ("left", "anterior", "superior"): "left-anterior-superior",
}

SPACE_KINDS = {"domain", "space"}
COMPONENT_KINDS = {"vector", "covariant-vector", "normal", "point", "2-vector", "3-vector", "4-vector", "3-gradient",
                   "3-normal", "quaternion", "complex", "2d-symmetric-matrix", "2d-masked-symmetric-matrix",
                   "2d-matrix", "2d-masked-matrix", "3d-symmetric-matrix", "3d-masked-symmetric-matrix", "3d-matrix",
                   "3d-masked-matrix"}
CHANNEL_KINDS = {"rgb-color", "rgba-color", "hsv-color", "xyz-color", "3-color", "4-color"}
OTHER_KINDS = {"list", "stub", "scalar", "???", "none"}
#: kinds with a fixed axis length
KIND_SIZES = {"stub": 1, "scalar": 1, "complex": 2, "2-vector": 2, "3-vector": 3, "4-vector": 4, "3-gradient": 3,
              "3-normal": 3, "quaternion": 4, "rgb-color": 3, "rgba-color": 4, "hsv-color": 3, "xyz-color": 3,
              "3-color": 3, "4-color": 4, "2d-symmetric-matrix": 3, "2d-masked-symmetric-matrix": 4, "2d-matrix": 4,
              "2d-masked-matrix": 5, "3d-symmetric-matrix": 6, "3d-masked-symmetric-matrix": 7, "3d-matrix": 9,
              "3d-masked-matrix": 10}
DEFAULT_KINDS = {"space": "domain", "time": "time", "component": "vector", "channel": "list", "other": "list"}
AXIS_NAME_BASES = {"time": "t", "component": "c", "channel": "c", "other": "u"}

#: NRRD unit strings -> MLArray unit names (free text otherwise) and back
UNIT_NAMES = {"nm": "nanometer", "um": "micrometer", "micron": "micrometer", "microns": "micrometer",
              "mm": "millimeter", "millimeter": "millimeter", "millimeters": "millimeter", "cm": "centimeter",
              "m": "meter", "s": "second", "sec": "second", "second": "second", "seconds": "second",
              "ms": "millisecond", "msec": "millisecond", "us": "microsecond", "usec": "microsecond",
              "ns": "nanosecond", "min": "minute", "h": "hour", "hz": "hertz", "deg": "degree", "rad": "radian"}
UNIT_SPELLINGS = {"nanometer": "nm", "micrometer": "um", "millimeter": "mm", "centimeter": "cm", "meter": "m",
                  "second": "s", "millisecond": "ms", "microsecond": "us", "nanosecond": "ns", "minute": "min",
                  "hour": "h", "hertz": "Hz", "degree": "deg", "radian": "rad"}
UNKNOWN_UNIT = "unknown"

#: NRRD type names -> numpy dtype names (pynrrd's table), and the names MedVol writes
NRRD_TYPES = {
    "signed char": "int8", "int8": "int8", "int8_t": "int8", "uchar": "uint8", "unsigned char": "uint8",
    "uint8": "uint8", "uint8_t": "uint8", "short": "int16", "short int": "int16", "signed short": "int16",
    "signed short int": "int16", "int16": "int16", "int16_t": "int16", "ushort": "uint16", "unsigned short": "uint16",
    "unsigned short int": "uint16", "uint16": "uint16", "uint16_t": "uint16", "int": "int32", "signed int": "int32",
    "int32": "int32", "int32_t": "int32", "uint": "uint32", "unsigned int": "uint32", "uint32": "uint32",
    "uint32_t": "uint32", "longlong": "int64", "long long": "int64", "long long int": "int64",
    "signed long long": "int64", "signed long long int": "int64", "int64": "int64", "int64_t": "int64",
    "ulonglong": "uint64", "unsigned long long": "uint64", "unsigned long long int": "uint64", "uint64": "uint64",
    "uint64_t": "uint64", "float": "float32", "double": "float64",
}
TYPE_NAMES = {"int8": "int8", "uint8": "uint8", "int16": "int16", "uint16": "uint16", "int32": "int32",
              "uint32": "uint32", "int64": "int64", "uint64": "uint64", "float32": "float", "float64": "double"}

#: Alternative field spellings allowed by the NRRD spec -> canonical spelling
ALIASES = {"axismins": "axis mins", "axismaxs": "axis maxs", "oldmin": "old min", "oldmax": "old max",
           "datafile": "data file", "lineskip": "line skip", "byteskip": "byte skip"}

#: How each field is written (everything else is a plain string)
FIELD_FORMATS = {
    "dimension": "int", "space dimension": "int", "line skip": "int", "byte skip": "int",
    "min": "double", "max": "double", "old min": "double", "old max": "double",
    "sizes": "int list", "spacings": "double list", "thicknesses": "double list", "axis mins": "double list",
    "axis maxs": "double list", "kinds": "string list", "centerings": "string list", "labels": "quoted list",
    "units": "quoted list", "space units": "quoted list", "space origin": "vector",
    "space directions": "vector list", "measurement frame": "vector list",
}

#: Fields that describe the grid (all regenerated from the core unless the geometry is unchanged)
GEOMETRY_FIELDS = ("space", "space dimension", "space directions", "space origin", "space units", "kinds",
                   "spacings", "axis mins", "axis maxs", "centerings")
DERIVED_FIELDS = ("type", "dimension", "sizes")
LAYOUT_FIELDS = ("encoding", "endian", "data file", "line skip", "byte skip")
#: Per-axis fields that are not core-owned: they follow their axis (by name) when axes are reordered
PER_AXIS_VERBATIM = ("labels", "units", "thicknesses")
#: Order for fields MedVol adds that the source did not have (pynrrd's and teem's order)
FIELD_ORDER = ("type", "dimension", "space dimension", "space", "sizes", "space directions", "kinds", "endian",
               "encoding", "min", "max", "old min", "old max", "content", "sample units", "spacings", "thicknesses",
               "axis mins", "axis maxs", "centerings", "labels", "units", "space units", "space origin",
               "measurement frame", "data file")

ENCODINGS = {"raw": "raw", "gzip": "gzip", "gz": "gzip", "bzip2": "bzip2", "bz2": "bzip2", "ascii": "ascii",
             "text": "ascii", "txt": "ascii"}
DATA_SUFFIXES = {"raw": ".raw", "gzip": ".raw.gz", "bzip2": ".raw.bz2", "ascii": ".txt"}
DEFAULT_MAGIC = "NRRD0004"


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def _get(fields: Dict[str, Any], name: str, default: Any = None) -> Any:
    """A field by canonical name, whatever spelling the file used."""
    for key, value in fields.items():
        if canonical(key) == name:
            return value
    return default


def _spelling(fields: Dict[str, Any], name: str) -> str:
    """The spelling the file used for a field (canonical if absent)."""
    return next((key for key in fields if canonical(key) == name), name)


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and np.isfinite(value)


def _entry(fields: Dict[str, Any], name: str, axis: Optional[int]) -> Any:
    values = _get(fields, name)
    if values is None or axis is None or axis >= len(values):
        return None
    return values[axis]


def _unit_name(text: Optional[str]) -> Optional[str]:
    if text is None or not str(text).strip():
        return None
    return UNIT_NAMES.get(str(text).strip().lower(), str(text))


def _unit_spelling(unit: str, source_text: Optional[str]) -> str:
    """How to write ``unit``: the source's spelling when it means the same, else a short code."""
    if source_text is not None and _unit_name(source_text) == unit:
        return source_text
    return UNIT_SPELLINGS.get(unit, unit)


def _kind_type(kind: Optional[str]) -> Optional[str]:
    """MLArray axis type of a kind (None: decided by the axis' space direction)."""
    kind = (kind or "").lower()
    if kind in COMPONENT_KINDS:
        return "component"
    if kind in CHANNEL_KINDS:
        return "channel"
    if kind == "time":
        return "time"
    if kind in SPACE_KINDS or kind in ("", "???", "none"):
        return None
    return "other"


# ------------------------------------------------------------------------------------------------
# readable header representation (MLArray spec 6.4)
# ------------------------------------------------------------------------------------------------

def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _native(value: Any) -> Any:
    """A pynrrd header value as plain lists and numbers; ``none`` rows (NaN rows in pynrrd) become None."""
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return [None if np.all(np.isnan(row)) else [float(v) for v in row] for row in value]
        return [_scalar(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [None if v is None else _native(v) for v in value]
    return _scalar(value)


@dataclass
class RawHeader:
    magic: str
    comments: List[str]
    fields: "OrderedDict[str, Any]"
    key_values: "OrderedDict[str, str]"

    def to_dict(self) -> Dict[str, Any]:
        return {"magic": self.magic, "comments": list(self.comments), "fields": dict(self.fields),
                "key_values": dict(self.key_values)}

    @classmethod
    def from_source(cls, header) -> "RawHeader":
        def thaw(value):
            if isinstance(value, (list, tuple)):
                return [thaw(v) for v in value]
            return value
        return cls(str(header.get("magic", DEFAULT_MAGIC)), list(header.get("comments", ())),
                   OrderedDict((k, thaw(v)) for k, v in header.get("fields", {}).items()),
                   OrderedDict(header.get("key_values", {}).items()))


def _header_lines(path: str) -> List[str]:
    """The header lines (magic included, up to the blank line), undecoded except for ASCII."""
    lines = []
    with open(path, "rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", "ignore").rstrip("\r\n")
            if not line.strip() and lines:
                break
            lines.append(line)
    return lines


def read_header(path: str) -> Tuple[RawHeader, Dict[str, Any]]:
    """The readable header and pynrrd's parsed header (needed by pynrrd's data reader)."""
    nrrd = _nrrd()
    parsed = nrrd.read_header(path)
    lines = _header_lines(path)
    comments, key_values = [], OrderedDict()
    for line in lines[1:]:
        if line.startswith("#"):
            comments.append(line[1:])
            continue
        colon = line.find(":")
        if line[colon:colon + 2] == ":=":
            key_values[line[:colon]] = line[colon + 2:]
    fields = OrderedDict((key, _native(value)) for key, value in parsed.items() if key not in key_values)
    return RawHeader(lines[0].strip(), comments, fields, key_values), parsed


def _format_value(name: str, value: Any) -> str:
    from nrrd import format_number, format_vector

    style = FIELD_FORMATS.get(canonical(name), "string")
    if style == "int":
        return str(int(value))
    if style == "double":
        return format_number(float(value))
    if style == "int list":
        return " ".join(str(int(v)) for v in value)
    if style == "double list":
        return " ".join(format_number(float(v)) for v in value)
    if style == "string list":
        return " ".join(str(v) for v in value)
    if style == "quoted list":
        return " ".join(f'"{v}"' for v in value)
    if style == "vector":
        return format_vector(np.asarray(value, dtype=float))
    if style == "vector list":
        return " ".join("none" if v is None else format_vector(np.asarray(v, dtype=float)) for v in value)
    return str(value)


def header_text(header: RawHeader) -> str:
    """Header text: magic, comments, fields, key-value pairs and the closing blank line."""
    lines = [header.magic] + [f"#{text}" for text in header.comments]
    lines += [f"{name}: {_format_value(name, value)}" for name, value in header.fields.items()]
    lines += [f"{key}:={value}" for key, value in header.key_values.items()]
    return "\n".join(lines) + "\n\n"


# ------------------------------------------------------------------------------------------------
# axes and geometry
# ------------------------------------------------------------------------------------------------

@dataclass
class FileAxis:
    """One NRRD axis (file order) as MedVol interprets it."""

    index: int
    size: int
    kind: Optional[str]
    type: str                          # MLArray axis type
    direction: Optional[List[float]]   # space direction (None: not a world axis)
    name: str = ""
    time_in_space: bool = False        # time axis of a ``*-time`` space (its direction holds the step)
    notes: List[str] = field(default_factory=list)


def _frame_of(fields: Dict[str, Any]) -> Optional[Frame]:
    space = _get(fields, "space")
    if space is None:
        return None
    frame = FRAMES.get(str(space).lower())
    if frame is None:
        raise GeometryError(f"Unknown NRRD space {space!r}; MedVol does not guess its frame.")
    return frame


def classify_axes(fields: Dict[str, Any]) -> Tuple[List[FileAxis], Optional[Frame], Optional[int]]:
    """Axis types and names in file order, the frame and the world dimension (None: no space).

    Pure function of the header, so writers can recompute the names the reader gave each axis.
    """
    sizes = [int(v) for v in _get(fields, "sizes")]
    ndim = len(sizes)
    kinds = list(_get(fields, "kinds") or [None] * ndim)
    frame = _frame_of(fields)
    space_dim = _get(fields, "space dimension")
    world = (4 if frame.time else 3) if frame is not None else (int(space_dim) if space_dim is not None else None)
    directions = list(_get(fields, "space directions") or [None] * ndim) if world is not None else [None] * ndim
    if len(directions) != ndim or len(kinds) != ndim:
        raise GeometryError(f"NRRD per-axis fields do not match dimension {ndim}.")
    axes = []
    for a in range(ndim):
        kind, direction = kinds[a], directions[a]
        by_kind = _kind_type(kind)
        axis = FileAxis(a, sizes[a], kind, by_kind or "space", None)
        if direction is not None:
            direction = [float(v) for v in direction]
            if len(direction) != world:
                raise GeometryError(f"NRRD axis {a}: space direction has {len(direction)} components, the space {world}.")
            if frame is not None and frame.time:
                spatial, step = direction[:3], direction[3]
                if any(spatial) and step != 0:
                    raise GeometryError(f"NRRD axis {a} couples space and time in {frame.name!r} "
                                        f"(direction {direction}); MedVol does not separate them.")
                if step != 0:
                    axis.type, axis.time_in_space = "time", True
                else:
                    axis.type = "space"
            else:
                axis.type = "space"
            axis.direction = direction
            if by_kind is not None and by_kind != axis.type:
                axis.notes.append(f"NRRD axis {a} has kind {kind!r} but a space direction; used as a {axis.type} axis")
        elif by_kind is None and world is not None:
            axis.type = "other"
            if (kind or "").lower() in SPACE_KINDS:
                axis.notes.append(f"NRRD axis {a} has kind {kind!r} but no space direction; used as an 'other' axis")
        axes.append(axis)
    counters: Dict[str, int] = {}
    for axis in axes:
        if axis.type == "space":
            n = counters.get("space", 0)
            axis.name = "xyz"[n] if n < 3 else f"s{n}"
            counters["space"] = n + 1
        else:
            base = AXIS_NAME_BASES[axis.type]
            n = counters.get(base, 0)
            axis.name = base if n == 0 else f"{base}{n + 1}"
            counters[base] = n + 1
    return axes, frame, world


def _sampling(fields: Dict[str, Any], axis: FileAxis, report: Report) -> Tuple[Optional[float], Optional[float]]:
    """(start, step) of a non-world axis from ``spacings``/``axis mins``/``axis maxs``, at voxel centres.

    Cell centring (NRRD's default when ``centerings`` is absent, and ITK's, KB ch04 §2.5 / P7) puts sample i
    at ``min + (i + 0.5) * step``; node centring at ``min + i * step``.
    """
    a, n = axis.index, axis.size
    spacing, low, high = (_entry(fields, name, a) for name in ("spacings", "axis mins", "axis maxs"))
    spacing, low, high = (v if _finite(v) else None for v in (spacing, low, high))
    if spacing is None and low is None and high is None:
        return None, None
    centering = str(_entry(fields, "centerings", a) or "").lower()
    if centering not in ("cell", "node"):
        if low is None and high is None:
            return None, spacing
        centering = "cell"
        report.choose(f"centering axis {a}", "cell (not given: the NRRD default, as in ITK)")
    step = spacing
    if step is None and low is not None and high is not None:
        cells = n if centering == "cell" else n - 1
        step = (high - low) / cells if cells > 0 else None
    half = 0.5 if centering == "cell" else 0.0
    if low is not None:
        start = low + half * step if step is not None else (low if half == 0 else None)
    elif high is not None and step is not None:
        start = high - (n - 1 + 2 * half) * step + half * step
    else:
        start = None
    if start is None and (low is not None or high is not None):
        report.warn(f"NRRD axis {a}: cell-centred axis min/max without a spacing; start kept in source only")
    return start, step


# ------------------------------------------------------------------------------------------------
# plugin
# ------------------------------------------------------------------------------------------------

class NrrdPlugin(FormatPlugin):
    name = "nrrd"
    extensions = (".nrrd", ".nhdr")
    capabilities = Capabilities(
        max_space_axes=3, min_space_axes=0, max_ndim=16,
        non_space_axes=frozenset({"time", "component", "channel", "other"}),
        shear=True, oblique=True, label_table=False, key_value=False,
        dtypes=frozenset(TYPE_NAMES),
    )
    fields = FieldTable(
        core_owned=frozenset(GEOMETRY_FIELDS + DERIVED_FIELDS + ("sample units",)),
        data_dependent=frozenset({"min", "max"}),
    )

    # -- sniff -------------------------------------------------------------------------------

    def sniff(self, path: PathLike) -> float:
        if not self.matches_extension(path):
            return 0.0
        try:
            with open(path, "rb") as handle:
                head = handle.read(8)
        except OSError:
            return 0.0
        return 1.0 if head.startswith(b"NRRD000") else 0.0

    # -- read --------------------------------------------------------------------------------

    def read(self, path: PathLike, policy: Policy) -> Tuple[List[Image], Report]:
        nrrd = _nrrd()
        path = str(path)
        report = Report("read", self.name, path)
        raw, parsed = read_header(path)
        with open(path, "rb") as handle:
            nrrd.read_header(handle)
            data = nrrd.read_data(parsed, handle, path, index_order="F")
        array = np.ascontiguousarray(data.T)                # [.., axis 1, axis 0]
        if not array.dtype.isnative:
            array = array.astype(array.dtype.newbyteorder("="))
            report.choose("byte_order", f"{raw.fields.get('endian', 'big')}-endian data stored in native order")
        fields = raw.fields
        file_axes, frame, world = classify_axes(fields)
        for axis in file_axes:
            for note in axis.notes:
                report.warn(note)

        space = self._space(fields, file_axes, frame, world, policy, report)
        axes = self._axes(fields, file_axes, report)[::-1]
        values: Dict[str, Any] = {"kind": "measurement"}
        sample_units = _unit_name(_get(fields, "sample units"))
        if sample_units is not None:
            values["unit"] = sample_units
        if _get(fields, "measurement frame") is not None:
            report.choose("measurement_frame", "kept in source as written (entry i = column i of the frame); "
                                               "vector values are not rotated")
        spaces = [space] if space is not None else []
        meta = Meta(axes=axes, spaces=spaces, values=values, shape=array.shape)
        ndim = array.ndim
        files = [os.path.basename(path)]
        data_file = _get(fields, "data file")
        if data_file is not None:
            files.append(str(data_file))
        source = {
            "format": "nrrd",
            "format_version": raw.magic[4:].lstrip("0") or "0",
            "files": files,
            "reader": {"name": "medvol"},
            "index_map": {"axis_order": list(range(ndim))[::-1], "flipped": [False] * ndim, "offset": [0] * ndim},
            "fingerprint": fingerprint.compute(array, meta),
            "header": raw.to_dict(),
        }
        meta = Meta(axes=axes, spaces=spaces, values=values, source=[source], shape=array.shape)
        return [Image(array, meta)], report

    @staticmethod
    def _space(fields, file_axes: List[FileAxis], frame: Optional[Frame], world: Optional[int], policy: Policy,
               report: Report) -> Optional[Dict[str, Any]]:
        """The default space, built directly from ``space directions`` and ``space origin``."""
        if world is None:
            if any(_finite(v) for name in ("spacings", "axis mins", "axis maxs") for v in (_get(fields, name) or [])):
                report.warn("NRRD has no 'space': the image is in index space only; "
                            "spacings/axis mins/axis maxs are kept in source")
            else:
                report.warn("NRRD has no 'space': the image is in index space only")
            return None
        spatial = [axis for axis in file_axes if axis.type == "space"]
        if not spatial:
            report.warn("NRRD 'space' given but no axis has a space direction: index space only")
            return None
        for axis in spatial:
            if _finite(_entry(fields, "spacings", axis.index)):
                report.warn(f"NRRD axis {axis.index} has both a space direction and 'spacings' "
                            "(invalid per the NRRD spec, ITK refuses it); the space direction is used")
        origin = _get(fields, "space origin")
        if origin is None:
            report.warn("NRRD has 'space directions' but no 'space origin'; origin taken as 0")
            origin = [0.0] * world
        origin = [float(v) for v in origin]
        linear = np.array([axis.direction for axis in spatial], dtype=np.float64).T     # columns = axes
        names = [axis.name for axis in spatial]
        units = _get(fields, "space units")
        if frame is None:
            report.choose("nrrd_space", f"unnamed {world}-D space (space dimension): not a patient space")
            unit_names = [(_unit_name(u) if units else None) or UNKNOWN_UNIT for u in (units or [None] * world)]
            space_axes = ["x", "y", "z"][:world] if world <= 3 else [f"s{r}" for r in range(world)]
            return {
                "name": "nrrd",
                "axes": [{"name": n, "unit": u} for n, u in zip(space_axes, unit_names)],
                "from_array": {"array_axes": names, "matrix": homogeneous(linear, origin).tolist()},
            }
        if units and any(_unit_name(u) not in ("millimeter", None) for u in units[:3]):
            report.warn(f"NRRD space units are {list(units[:3])}; geometry is stored in millimeter as written")
        matrix_lps = homogeneous(frame.signs @ linear[:3], frame.signs @ np.asarray(origin[:3]))
        if not frame.anatomical:
            report.warn(f"NRRD space {_get(fields, 'space')!r} has no defined anatomical frame; its coordinates are "
                        "taken as LPS, as ITK and Slicer do")
        report.choose("nrrd_space", f"{_get(fields, 'space')} -> {policy.world_frame}")
        return patient_space(matrix_lps, names, policy.world_frame)

    @staticmethod
    def _axes(fields, file_axes: List[FileAxis], report: Report) -> List[Dict[str, Any]]:
        """MLArray axes in NRRD file order (the caller reverses them)."""
        origin = _get(fields, "space origin")
        space_units = _get(fields, "space units")
        out = []
        for axis in file_axes:
            entry: Dict[str, Any] = {"name": axis.name, "type": axis.type}
            if axis.type == "space":
                out.append(entry)
                continue
            unit_text = _entry(fields, "units", axis.index)
            if axis.time_in_space:
                start = float(origin[3]) if origin is not None and _finite(origin[3]) else None
                step = axis.direction[3]
                unit_text = space_units[3] if space_units is not None and len(space_units) > 3 else unit_text
            else:
                start, step = _sampling(fields, axis, report)
            if start is not None:
                entry["start"] = start
            if step is not None:
                entry["step"] = step
            unit = _unit_name(unit_text)
            if unit is not None and (axis.type == "time" or start is not None or step is not None):
                entry["unit"] = unit
            out.append(entry)
        return out

    # -- write -------------------------------------------------------------------------------

    def write(self, image: Image, path: PathLike, policy: Policy) -> Report:
        path = str(path)
        report = Report("write", self.name, path)
        meta = image.meta
        source = next((entry for entry in meta.source if entry.format == "nrrd"), None)
        original = RawHeader.from_source(source.header) if source is not None else None
        unchanged = source is not None and self._geometry_unchanged(source, image)
        nrrd_axes = list(meta.axes)[::-1]                     # NRRD axis order: fastest first
        if len(meta.spaces) > 1:
            for extra in meta.spaces[1:]:
                report.drop(f"space {extra.name!r}", "NRRD holds one space", policy.on_loss)

        if unchanged:
            report.choose("geometry", "reused from source (unchanged)")
            fields = OrderedDict((k, v) for k, v in original.fields.items() if canonical(k) not in LAYOUT_FIELDS)
            fields[_spelling(fields, "type")] = self._type_name(image.array.dtype, original)
            # value units are not part of the geometry fingerprint: always from the core
            unit, key = meta.values.unit, _spelling(fields, "sample units")
            if unit is not None:
                fields[key] = _unit_spelling(unit, fields.get(key))
            else:
                fields.pop(key, None)
        else:
            fields = self._regenerate(image, nrrd_axes, original, report, policy)
        self._data_dependent(fields, source, image, report)

        encoding, endian, detached = self._layout(image.array.dtype, path, original, policy, report)
        layout = OrderedDict()
        if endian is not None:
            layout["endian"] = endian
        layout["encoding"] = encoding
        data_path = None
        if detached:
            base = path[:-5] if path.lower().endswith(".nhdr") else path
            data_path = base + DATA_SUFFIXES[ENCODINGS[encoding.lower()]]
            layout["data file"] = os.path.basename(data_path)
        source_fields = original.fields if original is not None else {}
        for name, value in layout.items():
            fields[_spelling(source_fields, name)] = value
        fields = self._ordered(fields, source_fields)

        magic = self._magic(fields, original, policy, report)
        header = RawHeader(magic, list(original.comments) if original is not None else [],
                           fields, OrderedDict(original.key_values) if original is not None else OrderedDict())
        self._write_file(header, image.array, encoding, endian, path, data_path)
        return report

    @staticmethod
    def _geometry_unchanged(source, image: Image) -> bool:
        fp = source.fingerprint or {}
        return (fp.get("geometry") == fingerprint.geometry_hash(image.meta)
                and list(fp.get("shape", [])) == list(image.array.shape))

    @staticmethod
    def _type_name(dtype: np.dtype, original: Optional[RawHeader]) -> str:
        name = np.dtype(dtype).name
        if name not in TYPE_NAMES:
            raise GeometryError(f"NRRD cannot store dtype {name}.")
        written = _get(original.fields, "type") if original is not None else None
        if written is not None and NRRD_TYPES.get(str(written)) == name:
            return written
        return TYPE_NAMES[name]

    # -- regeneration of the core-owned fields ------------------------------------------------

    def _regenerate(self, image: Image, nrrd_axes, original: Optional[RawHeader], report: Report,
                    policy: Policy) -> "OrderedDict[str, Any]":
        meta = image.meta
        src_fields = original.fields if original is not None else OrderedDict()
        source_axes: Dict[str, FileAxis] = {}
        if original is not None:
            try:
                source_axes = {axis.name: axis for axis in classify_axes(src_fields)[0]}
            except GeometryError:
                source_axes = {}
        # the source axis each output axis came from (matched by name: names follow their axis)
        match = [source_axes.get(axis.name) for axis in nrrd_axes]
        shape = image.array.shape[::-1]
        lost = [a.name for a in source_axes.values() if a.name not in {axis.name for axis in nrrd_axes}]
        if lost:
            report.warn(f"NRRD per-axis fields of source axes {lost} dropped: no axis of that name any more")

        new: Dict[str, Any] = OrderedDict()
        new["type"] = self._type_name(image.array.dtype, original)
        new["dimension"] = len(nrrd_axes)
        new["sizes"] = [int(v) for v in shape]
        geometry = self._space_fields(meta, nrrd_axes, original, report, policy)
        new.update(geometry["fields"])
        time_in_space = geometry["time_axis"]

        # kinds: the source kind of the same axis while it still fits, else the default for the axis type
        kinds = []
        for axis, src, size in zip(nrrd_axes, match, shape):
            kind = src.kind if src is not None else None
            if kind is None or not self._kind_fits(kind, axis.type, size, axis.name == time_in_space):
                kind = DEFAULT_KINDS[axis.type]
            kinds.append(kind)
        if original is None or _get(src_fields, "kinds") is not None or any(a.type != "space" for a in nrrd_axes):
            new["kinds"] = kinds

        new.update(self._sampling_fields(meta, nrrd_axes, match, shape, src_fields, geometry["world_axes"],
                                         time_in_space, report))
        new.update(self._verbatim_per_axis(nrrd_axes, match, src_fields, time_in_space))
        unit = meta.values.unit
        if unit is not None:
            new["sample units"] = _unit_spelling(unit, _get(src_fields, "sample units"))

        # everything else in the source is verbatim (comments and key-value pairs are kept separately)
        handled = set(GEOMETRY_FIELDS + DERIVED_FIELDS + LAYOUT_FIELDS + PER_AXIS_VERBATIM
                      + ("sample units", "measurement frame"))
        fields: Dict[str, Any] = OrderedDict()
        for key, value in src_fields.items():
            name = canonical(key)
            if name in new:
                fields[key] = new.pop(name)
            elif name not in handled:
                fields[key] = value
        fields.update(new)
        return fields

    @staticmethod
    def _kind_fits(kind: str, axis_type: str, size: int, time_in_space: bool) -> bool:
        lower = kind.lower()
        if KIND_SIZES.get(lower, size) != size:
            return False
        if axis_type == "space":
            return lower in SPACE_KINDS
        if axis_type == "time":
            return lower == "time" or (time_in_space and lower in SPACE_KINDS)
        if axis_type == "other":      # 'domain' without a space direction is read as 'other', too
            return _kind_type(kind) in ("other", None)
        return _kind_type(kind) == axis_type

    def _space_fields(self, meta: Meta, nrrd_axes, original: Optional[RawHeader], report: Report,
                      policy: Policy) -> Dict[str, Any]:
        """``space``/``space dimension``, ``space directions``, ``space origin``, ``space units`` and
        ``measurement frame`` from the core's default space."""
        out: Dict[str, Any] = {"fields": OrderedDict(), "world_axes": set(), "time_axis": None}
        space = meta.spaces[0] if meta.spaces else None
        src_fields = original.fields if original is not None else {}
        if space is None:
            report.warn("no space: NRRD written without 'space' (index space only)")
            if _get(src_fields, "measurement frame") is not None:
                report.drop("measurement frame", "no space to refer it to", policy.on_loss)
            return out
        mapped = list(space.from_array.array_axes)
        matrix = space.matrix()
        fields = out["fields"]
        try:
            lps = lps_matrix(space)
        except GeometryError as error:
            lps = None
            if not (original is not None and _get(src_fields, "space dimension") is not None
                    and _get(src_fields, "space") is None):
                report.drop(f"space {space.name!r}", f"not a patient space ({error}); written as an unnamed NRRD "
                            "space ('space dimension'), which ITK reads as LPS", policy.on_loss)
        names = [axis.name for axis in nrrd_axes]
        if lps is None:
            world = len(space.axes)
            linear, origin = matrix[:world, :-1], matrix[:world, -1]
            fields["space dimension"] = world
            columns = {name: [float(v) for v in linear[:, mapped.index(name)]] for name in mapped}
            fields["space directions"] = [columns.get(name) for name in names]
            fields["space origin"] = [float(v) for v in origin]
            units = [axis.unit for axis in space.axes]
            if any(u not in (None, UNKNOWN_UNIT) for u in units):
                src_units = _get(src_fields, "space units") or [None] * world
                fields["space units"] = [_unit_spelling(u, s) if u not in (None, UNKNOWN_UNIT) else ""
                                         for u, s in zip(units, src_units)]
            out["world_axes"] = set(mapped)
            if _get(src_fields, "measurement frame") is not None:
                fields["measurement frame"] = _get(src_fields, "measurement frame")
            return out

        # patient space: the source's space name when it can hold the geometry, else the core's own frame
        src_space = _get(src_fields, "space")
        src_frame = FRAMES.get(str(src_space).lower()) if src_space is not None else None
        words = tuple(axis.increases_toward for axis in space.axes)
        time_axes = [axis for axis in nrrd_axes if axis.type == "time"]
        sampled_time = len(time_axes) == 1 and time_axes[0].step is not None
        if src_frame is not None:
            space_name, frame = src_space, src_frame
        elif sampled_time:
            # ITK reads a 4-D space only as a '*-time' space and takes its numbers as LPS without
            # conversion (KB ch04 P4), so LPS-time is the one 4-D space it reads correctly
            space_name = "left-posterior-superior-time"
            frame = FRAMES[space_name]
            report.choose("nrrd_time", "time axis written as the 4th coordinate of 'left-posterior-superior-time'")
        else:
            space_name = WORDS_TO_SPACE.get(words, "left-posterior-superior")
            frame = FRAMES[space_name]
        if frame.time and not sampled_time:
            plain_name = frame.name[: -len("-time")]
            report.warn(f"NRRD space {space_name!r} needs one sampled time axis; written as {plain_name!r}")
            space_name, frame = plain_name, FRAMES[plain_name.lower()]
        report.choose("nrrd_space", space_name)
        linear = frame.signs @ lps[:3, :-1]
        origin = frame.signs @ lps[:3, -1]
        directions = {name: [float(v) for v in linear[:, mapped.index(name)]] for name in mapped}
        origin = [float(v) for v in origin]
        if frame.time:
            time = time_axes[0]
            directions = {name: vector + [0.0] for name, vector in directions.items()}
            directions[time.name] = [0.0, 0.0, 0.0, float(time.step)]
            origin.append(float(time.start) if time.start is not None else 0.0)
            out["time_axis"] = time.name
        fields["space"] = space_name
        fields["space directions"] = [directions.get(name) for name in names]
        fields["space origin"] = origin
        out["world_axes"] = set(directions)
        src_units = _get(src_fields, "space units")
        time_unit = time_axes[0].unit if frame.time else None
        # patient spaces are in millimeter; units are written if the source had them or time needs one
        if src_units is not None or time_unit is not None:
            src_units = list(src_units or []) + [None] * (4 - len(src_units or []))
            units = [_unit_spelling("millimeter", s) for s in src_units[:3]]
            if frame.time:
                units.append(_unit_spelling(time_unit, src_units[3]) if time_unit else "")
            fields["space units"] = units
        if _get(src_fields, "measurement frame") is not None:
            # the written space has the source frame's axis signs whenever the source named one, so the
            # frame (which maps measurement coordinates to that space) stays valid as written
            fields["measurement frame"] = _get(src_fields, "measurement frame")
        return out

    def _sampling_fields(self, meta: Meta, nrrd_axes, match, shape, src_fields, world_axes, time_axis,
                         report: Report) -> Dict[str, Any]:
        """``spacings``/``axis mins``/``axis maxs``/``centerings``.

        Axes outside the space: from the core's start/step (voxel centres), using the centring the source
        used for that axis (cell: ``min = start - step/2``), or node for new axes. Space axes of an image
        without a space: the source values of the same axis (the core has no geometry for them); axis
        mins/maxs only while the axis length is unchanged.
        """
        has_space = bool(meta.spaces)
        src_has = {name: _get(src_fields, name) is not None for name in ("spacings", "axis mins", "axis maxs",
                                                                         "centerings")}
        spacings, mins, maxs, centerings = [], [], [], []
        need_centerings = False
        for axis, src, n in zip(nrrd_axes, match, shape):
            s_index = src.index if src is not None else None
            src_center = str(_entry(src_fields, "centerings", s_index) or "???")
            src_vals = {name: _entry(src_fields, name, s_index) for name in ("spacings", "axis mins", "axis maxs")}
            src_vals = {k: (v if _finite(v) else None) for k, v in src_vals.items()}
            spacing = low = high = None
            center = src_center if src_has["centerings"] and src is not None else "???"
            if axis.name in world_axes or axis.name == time_axis:
                pass
            elif axis.type == "space" and not has_space:
                if src is not None:
                    spacing = src_vals["spacings"]
                    if src.size == n:
                        low, high = src_vals["axis mins"], src_vals["axis maxs"]
                    elif src_vals["axis mins"] is not None or src_vals["axis maxs"] is not None:
                        report.warn(f"axis {axis.name!r}: NRRD axis mins/maxs dropped, the axis length changed")
            elif axis.type != "space" and (axis.start is not None or axis.step is not None):
                step, start = axis.step, axis.start
                if src is not None:
                    centering = src_center.lower() if src_center.lower() in ("cell", "node") else (
                        "cell" if src_vals["axis mins"] is not None or src_vals["axis maxs"] is not None else None)
                    wants = [k for k, v in src_vals.items() if v is not None] or ["spacings", "axis mins"]
                else:
                    centering, wants = None, ["spacings", "axis mins"]
                if centering is None or (centering == "cell" and step is None):
                    centering = "node"
                    need_centerings = True
                    center = "node"
                half = 0.5 if centering == "cell" else 0.0
                if "spacings" in wants and step is not None:
                    spacing = step
                if start is not None and ("axis mins" in wants or "axis maxs" not in wants):
                    low = start - half * step if half else start
                if "axis maxs" in wants and start is not None and step is not None:
                    high = start + (n - 1 + half) * step
            spacings.append(spacing)
            mins.append(low)
            maxs.append(high)
            centerings.append(center)
        out: Dict[str, Any] = OrderedDict()
        for name, values in (("spacings", spacings), ("axis mins", mins), ("axis maxs", maxs)):
            if any(v is not None for v in values):
                out[name] = [float("nan") if v is None else float(v) for v in values]
        if src_has["centerings"] or need_centerings:
            if any(c not in ("???", "none") for c in centerings):
                out["centerings"] = centerings
        return out

    @staticmethod
    def _verbatim_per_axis(nrrd_axes, match, src_fields, time_axis) -> Dict[str, Any]:
        """``labels``, ``units`` and ``thicknesses`` follow their axis; ``units`` of non-space axes with a
        unit come from the core (except the time axis of a ``*-time`` space: its unit is in ``space units``)."""
        out: Dict[str, Any] = OrderedDict()
        for name, empty in (("labels", ""), ("units", ""), ("thicknesses", float("nan"))):
            values = []
            for axis, src in zip(nrrd_axes, match):
                value = _entry(src_fields, name, src.index if src is not None else None)
                if name == "units" and axis.type != "space" and axis.unit is not None and axis.name != time_axis:
                    value = _unit_spelling(axis.unit, value)
                values.append(empty if value is None else value)
            if any((_finite(v) if name == "thicknesses" else v != "") for v in values):
                out[name] = values
        return out

    # -- data-dependent and layout fields ----------------------------------------------------

    @staticmethod
    def _data_dependent(fields, source, image: Image, report: Report) -> None:
        if source is None:
            return
        fp = source.fingerprint or {}
        if fp and fingerprint.matches(fp, image.array, image.meta):
            return
        for key in list(fields):
            if canonical(key) in ("min", "max"):
                report.warn(f"{key} dropped: the data changed since it was read")
                del fields[key]

    @staticmethod
    def _layout(dtype, path: str, original: Optional[RawHeader], policy: Policy,
                report: Report) -> Tuple[str, Optional[str], bool]:
        """Encoding, endianness (None: not written) and whether the header is detached."""
        src = original.fields if original is not None else {}
        encoding = str(_get(src, "encoding", "gzip")) if policy.layout == "source" else "gzip"
        if encoding.lower() not in ENCODINGS:
            report.warn(f"NRRD encoding {encoding!r} cannot be written; using gzip")
            encoding = "gzip"
        endian = None
        if np.dtype(dtype).itemsize > 1 and ENCODINGS[encoding.lower()] != "ascii":
            endian = str(_get(src, "endian", "little")) if policy.layout == "source" else "little"
        elif policy.layout == "source" and _get(src, "endian") is not None:
            endian = str(_get(src, "endian"))     # harmless for 1-byte or text data; kept as stored
        detached = path.lower().endswith(".nhdr")
        report.choose("layout", f"encoding {encoding}" + (f", {endian} endian" if endian else "")
                      + (", detached header" if detached else ""))
        return encoding, endian, detached

    @staticmethod
    def _ordered(fields: Dict[str, Any], source_fields: Dict[str, Any]) -> "OrderedDict[str, Any]":
        """Source field order first; new fields at their usual place in teem/pynrrd order."""
        def rank(key: str) -> int:
            name = canonical(key)
            return FIELD_ORDER.index(name) if name in FIELD_ORDER else len(FIELD_ORDER)

        order = [k for k in source_fields if k in fields]
        for key in sorted((k for k in fields if k not in order), key=rank):
            later = [i for i, existing in enumerate(order) if rank(existing) > rank(key)]
            order.insert(later[0] if later else len(order), key)
        data_file = [k for k in order if canonical(k) == "data file"]
        order = [k for k in order if canonical(k) != "data file"] + data_file   # always last
        return OrderedDict((k, fields[k]) for k in order)

    @staticmethod
    def _magic(fields, original: Optional[RawHeader], policy: Policy, report: Report) -> str:
        """The source's magic under ``layout='source'``, raised when the fields need a newer version."""
        need = 1
        names = {canonical(k) for k in fields}
        if original is not None and original.key_values:
            need = 2
        if names & {"kinds", "centerings"}:
            need = max(need, 3)
        if names & {"space", "space dimension", "space directions", "space origin", "space units",
                    "measurement frame"}:
            need = max(need, 4)
        magic = original.magic if original is not None and policy.layout == "source" else DEFAULT_MAGIC
        version = int(magic[4:]) if magic[4:].isdigit() else 5
        if version < need:
            report.choose("nrrd_version", f"NRRD000{need} (the fields need it; the source was {magic})")
            version = need
        return f"NRRD{version:04d}"

    @staticmethod
    def _write_file(header: RawHeader, array: np.ndarray, encoding: str, endian: Optional[str], path: str,
                    data_path: Optional[str]) -> None:
        from nrrd.writer import _write_data

        data = np.asarray(array)
        if endian is not None and data.dtype.itemsize > 1:
            data = data.astype(data.dtype.newbyteorder("<" if endian == "little" else ">"), copy=False)
        data = data.T                                        # NRRD order [axis 0, axis 1, ...]; F bytes = C bytes
        kind = ENCODINGS[encoding.lower()]
        level = 6 if kind == "gzip" else 9
        with open(path, "wb") as handle:
            handle.write(header_text(header).encode("ascii"))
            if data_path is None:
                _write_data(data, handle, {"encoding": kind}, compression_level=level, index_order="F")
        if data_path is not None:
            with open(data_path, "wb") as handle:
                _write_data(data, handle, {"encoding": kind}, compression_level=level, index_order="F")
