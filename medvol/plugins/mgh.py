"""FreeSurfer MGH / MGZ plugin (spec section 7.4).

nibabel provides the header layout (``header_dtype``, ``footer_dtype``), the data type codes, gzip
handling (``ImageOpener``) and the array writer. MedVol parses and writes the header itself:
nibabel's ``MGHHeader`` rewrites ``goodRASFlag = 0`` headers on construction, and ``MGHImage``
recomputes ``delta``, ``Mdc`` and ``Pxyz_c`` from its affine on save.

File layout (big-endian; KB ``docs/04-nrrd-mha-other-formats.md`` section 4):

- bytes 0-89: ``version``, ``dims`` (i, j, k, frames), ``type``, ``dof``, ``goodRASFlag``, ``delta``,
  ``Mdc`` (unit vectors of i, j, k, one row per axis), ``Pxyz_c`` (``c_ras``); bytes 90-283 unused;
- data from byte 284, ``i`` fastest;
- optional footer: ``tr`` (ms), ``flip_angle``, ``te``, ``ti``, ``fov``, then FreeSurfer tags.

Geometry (FreeSurfer ``MRIxfmCRS2XYZ``): ``vox2ras = [M | c_ras - M . N/2]`` with
``M = Mdc^T . diag(delta)`` and ``N/2`` the exact (non-integer for odd N) half size. With
``goodRASFlag <= 0`` FreeSurfer ignores ``delta``, ``Mdc`` and ``Pxyz_c`` and uses coronal LIA,
1 mm, ``c_ras`` = 0; nibabel 5.4.2 reads such files as LSP (KB 04 P22b), so its affine is not used.

Array order: the MLArray array is the MGH array with its axes reversed (C order, last axis = i).
A frame axis (``dims[3] > 1``) is a ``time`` axis with step ``tr`` (ms) when ``tr > 0``, else ``other``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from mlarray import Meta, fingerprint

from medvol.geometry import GeometryError, has_shear, homogeneous, lps_matrix, patient_space
from medvol.image import Image
from medvol.plugins.base import Capabilities, FieldTable, FormatPlugin, PathLike
from medvol.policy import Policy
from medvol.report import Report

FLIP = np.diag([-1.0, -1.0, 1.0])
FLIP4 = np.diag([-1.0, -1.0, 1.0, 1.0])

DATA_OFFSET = 284
HEADER_FIELDS = ("version", "dims", "type", "dof", "goodRASFlag", "delta", "Mdc", "Pxyz_c")
FOOTER_FIELDS = ("tr", "flip_angle", "te", "ti", "fov")

#: Header fields that encode the geometry; reused verbatim while the geometry is unchanged.
GEOMETRY_FIELDS = ("goodRASFlag", "delta", "Mdc", "Pxyz_c")
DERIVED_FIELDS = ("dims", "type")

#: FreeSurfer's default for goodRASFlag <= 0: i -> left, j -> inferior, k -> anterior (RAS columns), 1 mm
LIA_RAS = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

#: time units -> factor to milliseconds (the unit of ``tr``)
TO_MILLISECOND = {"millisecond": 1.0, "second": 1000.0, "microsecond": 1e-3}


def _mgh():
    from nibabel.freesurfer import mghformat

    return mghformat


def _opener(path: str, mode: str):
    from nibabel.openers import ImageOpener

    # .mgz is gzip; nibabel registers the extension when mghformat is imported
    _mgh()
    return ImageOpener(path, mode)


# ------------------------------------------------------------------------------------------------
# readable header representation (MLArray spec 6.4)
# ------------------------------------------------------------------------------------------------

def _native(value: np.ndarray) -> Any:
    """A header field as a plain value: shortest float that maps back to the stored float32.

    2-D fields (``Mdc``) stay nested, one row per axis as stored.
    """
    value = np.asarray(value)
    if value.ndim == 2:
        return [_native(row) for row in value]
    if value.dtype.kind == "f":
        items = [float(str(v)) if np.isfinite(v) else float(v) for v in value.reshape(-1)]
    else:
        items = [int(v) for v in value.reshape(-1)]
    return items if value.ndim else items[0]


def parse(raw: bytes) -> Tuple[Dict[str, Any], np.ndarray]:
    """The readable header (field order as stored) and the data in MGH order ``[i, j, k(, frame)]``."""
    mgh = _mgh()
    if len(raw) < DATA_OFFSET:
        raise ValueError("file too short for an MGH header")
    fixed = np.frombuffer(raw, mgh.header_dtype, count=1)[0]
    header: Dict[str, Any] = {name: _native(fixed[name]) for name in HEADER_FIELDS}
    if header["version"] != 1:
        raise ValueError(f"MGH version {header['version']} is not 1")
    dims = [int(v) for v in header["dims"]]
    if min(dims) < 1:
        raise ValueError(f"MGH dims {dims} must be positive")
    unused = raw[mgh.header_dtype.itemsize:DATA_OFFSET]
    if any(unused):
        header["unused"] = bytes(unused)
    try:
        dtype = mgh.data_type_codes.numpy_dtype[header["type"]]
    except KeyError:
        raise ValueError(f"MGH data type {header['type']} is not supported") from None
    count = int(np.prod(dims))
    end = DATA_OFFSET + count * dtype.itemsize
    if len(raw) < end:
        raise ValueError(f"MGH data truncated: {len(raw) - DATA_OFFSET} of {end - DATA_OFFSET} bytes")
    shape = tuple(dims) if dims[3] > 1 else tuple(dims[:3])
    data = np.frombuffer(raw, dtype, count=count, offset=DATA_OFFSET).reshape(shape, order="F")
    trailer = raw[end:]
    if len(trailer) >= mgh.footer_dtype.itemsize:
        footer = np.frombuffer(trailer, mgh.footer_dtype, count=1)[0]
        header.update({name: _native(footer[name]) for name in FOOTER_FIELDS})
        trailer = trailer[mgh.footer_dtype.itemsize:]
    if trailer:
        header["tags"] = bytes(trailer)   # FreeSurfer tags, kept as stored
    return header, data


# ------------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------------

def vox2ras(header: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    """FreeSurfer's vox2ras (RAS, voxel centres) and whether the stored geometry was valid."""
    dims = np.asarray(header["dims"][:3], dtype=np.float64)
    if int(header["goodRASFlag"]) > 0:
        delta = np.asarray(header["delta"], dtype=np.float32).astype(np.float64)
        mdc = np.asarray(header["Mdc"], dtype=np.float32).astype(np.float64)   # rows = axes i, j, k
        linear = mdc.T * delta
        c_ras = np.asarray(header["Pxyz_c"], dtype=np.float32).astype(np.float64)
        good = True
    else:
        linear, c_ras, good = LIA_RAS.copy(), np.zeros(3), False
    return homogeneous(linear, c_ras - linear @ (dims / 2.0)), good


def geometry_fields(linear_ras: np.ndarray, translation_ras: np.ndarray, dims) -> Dict[str, Any]:
    """``delta``, ``Mdc`` and ``Pxyz_c`` for a RAS grid (columns = step vectors of i, j, k)."""
    delta = np.linalg.norm(linear_ras, axis=0)
    mdc = (linear_ras / delta).T
    c_ras = linear_ras @ (np.asarray(dims[:3], dtype=np.float64) / 2.0) + translation_ras
    return {"goodRASFlag": 1, "delta": delta, "Mdc": mdc, "Pxyz_c": c_ras}


# ------------------------------------------------------------------------------------------------
# plugin
# ------------------------------------------------------------------------------------------------

class MghPlugin(FormatPlugin):
    name = "mgh"
    extensions = (".mgh", ".mgz")
    capabilities = Capabilities(
        max_space_axes=3, min_space_axes=3, max_ndim=4,
        non_space_axes=frozenset({"time", "other", "channel", "component"}),
        shear=True, oblique=True, label_table=False, key_value=False,   # Mdc holds sheared columns exactly (KB 04 §4.4)
        dtypes=frozenset({"uint8", "int16", "uint16", "int32", "float32"}),
    )
    fields = FieldTable(
        # tr is the frame step of a time axis; for other files it is a scan parameter and stays verbatim
        core_owned=frozenset(GEOMETRY_FIELDS + DERIVED_FIELDS + ("tr",)),
        data_dependent=frozenset(),
    )

    # -- sniff -------------------------------------------------------------------------------

    def sniff(self, path: PathLike) -> float:
        path = str(path)
        if not self.matches_extension(path):
            return 0.0
        try:
            with _opener(path, "rb") as handle:
                head = handle.read(_mgh().header_dtype.itemsize)
            fixed = np.frombuffer(head, _mgh().header_dtype, count=1)[0]
        except Exception:
            return 0.0
        return 1.0 if int(fixed["version"]) == 1 and int(fixed["type"]) in _mgh().data_type_codes.code else 0.0

    # -- read --------------------------------------------------------------------------------

    def read(self, path: PathLike, policy: Policy) -> Tuple[List[Image], Report]:
        report = Report("read", self.name, str(path))
        with _opener(str(path), "rb") as handle:
            raw = handle.read()
        header, stored = parse(raw)
        ndim = stored.ndim
        array = np.ascontiguousarray(stored.transpose(tuple(range(ndim))[::-1]))
        array = array.astype(array.dtype.newbyteorder("="), copy=False)   # same values, native byte order

        affine_ras, good = vox2ras(header)
        if good:
            report.choose("mgh_geometry", "stored (goodRASFlag 1)")
        else:
            report.choose("mgh_geometry", f"FreeSurfer default (goodRASFlag {header['goodRASFlag']})")
            report.warn(f"goodRASFlag = {header['goodRASFlag']}: the stored delta, Mdc and Pxyz_c are not valid; "
                        "geometry is FreeSurfer's default (coronal LIA, 1 mm, c_ras = 0). nibabel 5.4.2 reads "
                        "this file differently (LSP)")
        space = patient_space(FLIP4 @ affine_ras, ["x", "y", "z"], policy.world_frame)
        axes = self._axes(header, ndim)
        values = {"kind": "measurement"}
        meta = Meta(axes=axes, spaces=[space], values=values, shape=array.shape)
        source = {
            "format": "mgh",
            "format_version": str(header["version"]),
            "files": [os.path.basename(str(path))],
            "reader": {"name": "medvol"},
            "index_map": {"axis_order": list(range(ndim))[::-1], "flipped": [False] * ndim, "offset": [0] * ndim},
            "fingerprint": fingerprint.compute(array, meta),
            "header": header,
        }
        meta = Meta(axes=axes, spaces=[space], values=values, source=[source], shape=array.shape)
        return [Image(array, meta)], report

    @staticmethod
    def _axes(header: Dict[str, Any], ndim: int) -> List[Dict[str, Any]]:
        """Axes in MLArray (reversed MGH) order."""
        mgh_axes: List[Dict[str, Any]] = [{"name": name, "type": "space"} for name in ("x", "y", "z")]
        if ndim == 4:
            tr = header.get("tr", 0.0)
            if np.isfinite(tr) and tr > 0:
                mgh_axes.append({"name": "t", "type": "time", "step": tr, "unit": "millisecond"})
            else:
                mgh_axes.append({"name": "f", "type": "other"})
        return mgh_axes[::-1]

    # -- write -------------------------------------------------------------------------------

    def write(self, image: Image, path: PathLike, policy: Policy) -> Report:
        report = Report("write", self.name, str(path))
        meta = image.meta
        path = str(path)
        source = next((entry for entry in meta.source if entry.format == "mgh"), None)   # encoding comes from the extension

        order, mgh_axes = self._layout(meta)
        data = np.transpose(image.array, order)              # MGH order [i, j, k(, frame)]
        header = self._start_header(source)
        dims = list(data.shape) + [1] * (4 - data.ndim)
        header["dims"] = dims
        try:
            header["type"] = int(_mgh().data_type_codes.code[data.dtype.newbyteorder("=")])
        except KeyError:
            raise ValueError(f"MGH cannot store {data.dtype.name} data") from None

        if source is not None and self._geometry_unchanged(source, image):
            report.choose("geometry", "reused from source (unchanged)")
        else:
            self._set_geometry(header, meta, mgh_axes, dims, source, report)
            self._set_frames(header, mgh_axes, report, policy)
        self._check_values(meta, report, policy)
        self._write_file(header, data, path)
        return report

    @staticmethod
    def _layout(meta: Meta) -> Tuple[List[int], List[Any]]:
        """Array axis order for MGH: space axes reversed (i fastest), then the frame axis."""
        axes = list(meta.axes)
        space = meta.spaces[0] if meta.spaces else None
        mapped = list(space.from_array.array_axes) if space is not None else [a.name for a in axes if a.type == "space"]
        space_idx = [i for i, a in enumerate(axes) if a.name in mapped][::-1]
        rest = [i for i in range(len(axes)) if i not in space_idx]
        if len(space_idx) != 3 or len(rest) > 1:
            raise GeometryError(f"MGH holds 3 space axes and at most one frame axis; got axes {[a.name for a in axes]}.")
        order = space_idx + rest
        return order, [axes[i] for i in order]

    @staticmethod
    def _start_header(source) -> Dict[str, Any]:
        """The source header (same format only) or FreeSurfer's defaults, in file field order."""
        if source is not None:
            return {key: value for key, value in source.header.items()}
        header: Dict[str, Any] = {"version": 1, "dims": [1, 1, 1, 1], "type": 3, "dof": 0}
        header.update({"goodRASFlag": 1, "delta": [1.0] * 3, "Mdc": LIA_RAS.T.tolist(), "Pxyz_c": [0.0] * 3})
        header.update({name: 0.0 for name in FOOTER_FIELDS})
        return header

    @staticmethod
    def _geometry_unchanged(source, image: Image) -> bool:
        fp = source.fingerprint or {}
        return (fp.get("geometry") == fingerprint.geometry_hash(image.meta)
                and list(fp.get("shape", [])) == list(image.array.shape))

    @staticmethod
    def _set_geometry(header: Dict[str, Any], meta: Meta, mgh_axes, dims, source, report: Report) -> None:
        space = meta.spaces[0] if meta.spaces else None
        if space is None:
            report.warn("no space: written with goodRASFlag = 0 (FreeSurfer then assumes coronal LIA, 1 mm, c_ras = 0)")
            header.update({"goodRASFlag": 0, "delta": [1.0] * 3, "Mdc": LIA_RAS.T.tolist(), "Pxyz_c": [0.0] * 3})
            return
        lps = lps_matrix(space)
        array_axes = list(space.from_array.array_axes)
        columns = [array_axes.index(axis.name) for axis in mgh_axes[:3]]
        linear = FLIP @ lps[:3, columns]
        if has_shear(linear):
            report.warn("sheared grid: written as non-orthonormal Mdc columns. nibabel and FreeSurfer mri_info read "
                        "them exactly, but the format defines Mdc as direction cosines and other readers may not")
        if source is not None and int(source.header.get("goodRASFlag", 1)) <= 0:
            report.warn(f"source goodRASFlag = {source.header['goodRASFlag']}: written with goodRASFlag = 1 and the "
                        "explicit geometry FreeSurfer assumed for it")
        header.update(geometry_fields(linear, FLIP @ lps[:3, -1], dims))

    @staticmethod
    def _set_frames(header: Dict[str, Any], mgh_axes, report: Report, policy: Policy) -> None:
        """``tr`` from a time axis with a step (in ms); otherwise the scan parameter stays as it is."""
        if len(mgh_axes) < 4:
            return
        frame = mgh_axes[3]
        if frame.type not in ("time", "other"):
            report.warn(f"axis {frame.name!r} ({frame.type}) is written as MGH frames and reads back as type 'other'")
        if frame.type != "time" or frame.step is None:
            return
        factor = TO_MILLISECOND.get(frame.unit) if frame.unit is not None else 1.0
        if factor is None:
            report.warn(f"time axis unit {frame.unit!r} is not a time; tr not set from its step")
            return
        if frame.unit is None:
            report.warn("time axis without unit: its step is written as tr in milliseconds")
        if frame.start:
            report.drop(f"axis {frame.name!r} start", "MGH has no time offset", policy.on_loss)
        header["tr"] = frame.step * factor
        for name in FOOTER_FIELDS:             # a source without footer gets one to hold tr
            header.setdefault(name, 0.0)

    @staticmethod
    def _check_values(meta: Meta, report: Report, policy: Policy) -> None:
        values = meta.values
        if values.scale is not None and (values.scale != 1 or (values.offset or 0) != 0):
            report.drop("values.scale/offset", "MGH has no value scaling; stored values are written as they are",
                        policy.on_loss)

    @staticmethod
    def _write_file(header: Dict[str, Any], data: np.ndarray, path: str) -> None:
        """Header, data, footer and tags, written without nibabel's header objects (see module docstring)."""
        from nibabel.volumeutils import array_to_file

        mgh = _mgh()
        fixed = np.zeros((), dtype=mgh.header_dtype)
        for name in HEADER_FIELDS:
            fixed[name] = np.asarray(header[name])
        unused = bytes(header.get("unused", b""))
        unused = unused + b"\x00" * (DATA_OFFSET - mgh.header_dtype.itemsize - len(unused))
        out_dtype = mgh.data_type_codes.numpy_dtype[int(header["type"])]
        with _opener(path, "wb") as handle:
            handle.write(fixed.tobytes())
            handle.write(unused)
            array_to_file(np.asfortranarray(data), handle, out_dtype, offset=None, order="F")
            if all(name in header for name in FOOTER_FIELDS):
                footer = np.zeros((), dtype=mgh.footer_dtype)
                for name in FOOTER_FIELDS:
                    footer[name] = header[name]
                handle.write(footer.tobytes())
            handle.write(bytes(header.get("tags", b"")))
