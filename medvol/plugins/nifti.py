"""NIfTI-1 / NIfTI-2 plugin (spec section 7.1).

nibabel parses headers and does the byte-level data IO; MedVol interprets the geometry itself and
writes header, extensions and data itself, so nibabel never rewrites qform/sform or the scaling.

Array order: the MLArray array is the NIfTI array with its axes reversed (C order, last axis = i).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from mlarray import Meta, fingerprint
from mlarray.metadata import Axis

from medvol.geometry import FRAME_WORDS, GeometryError, has_shear, homogeneous, lps_matrix, patient_space, world_corners
from medvol.image import Image
from medvol.plugins.base import Capabilities, FieldTable, FormatPlugin, PathLike
from medvol.policy import Policy
from medvol.report import Report
from medvol import vectors

FLIP4 = np.diag([-1.0, -1.0, 1.0, 1.0])

#: NIfTI xform codes and the spaces they name
CODE_SPACES = {1: ("patient", None), 2: ("aligned", None), 3: ("talairach", {"scheme": "atlas", "id": "Talairach"}),
               4: ("mni", {"scheme": "atlas", "id": "MNI152"}), 5: ("template", None)}
SPACE_CODES = {name: code for code, (name, _) in CODE_SPACES.items()}

SPACE_UNITS = {1: "meter", 2: "millimeter", 3: "micrometer"}
TIME_UNITS = {8: "second", 16: "millisecond", 24: "microsecond", 32: "hertz", 40: "ppm", 48: "rad/s"}
UNIT_SPACE_CODES = {v: k for k, v in SPACE_UNITS.items()}
UNIT_TIME_CODES = {v: k for k, v in TIME_UNITS.items()}
#: symmetric matrix, displacement vector, vector, point set, triangle, quaternion, dimensionless
VECTOR_INTENTS = {1005, 1006, 1007, 1008, 1009, 1010, 1011}
#: intents whose 3 components are a world vector, in a frame NIfTI does not declare (KB ch03 §8.2)
FRAME_INTENTS = {1006: "displacement vector", 1007: "vector"}

#: Header fields that encode the geometry; reused verbatim while the geometry is unchanged.
GEOMETRY_FIELDS = ("dim_info", "pixdim", "slice_start", "slice_end", "slice_code", "slice_duration", "xyzt_units", "toffset",
                   "qform_code", "sform_code", "quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z",
                   "srow_x", "srow_y", "srow_z")
DERIVED_FIELDS = ("sizeof_hdr", "vox_offset", "magic", "dim", "datatype", "bitpix")


def _nib():
    import nibabel as nib

    return nib


# ------------------------------------------------------------------------------------------------
# readable header representation (MLArray spec 6.4)
# ------------------------------------------------------------------------------------------------

def _native(value: np.ndarray) -> Any:
    """A header field as a plain value: shortest float that maps back to the stored binary float."""
    value = np.asarray(value)
    if value.dtype.kind == "S":
        return value.tobytes().split(b"\x00", 1)[0].decode("latin-1")
    if value.dtype.kind == "f":
        items = [float(str(v)) if np.isfinite(v) else float(v) for v in value.reshape(-1)]
    else:
        items = [int(v) for v in value.reshape(-1)]
    return items if value.ndim else items[0]


def _raw_extension(extension) -> bytes:
    """The stored bytes of an extension (typed extensions decode on get_content())."""
    raw = getattr(extension, "content", None)
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return bytes(extension._mangle(extension.get_content()))


def header_to_dict(header) -> Dict[str, Any]:
    out = {key: _native(header[key]) for key in header.keys()}
    extensions = [{"code": int(ext.get_code()), "content": _raw_extension(ext)} for ext in header.extensions]
    if extensions:
        out["extensions"] = extensions
    return out


# ------------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------------

def _form_corner_distance(a: np.ndarray, b: np.ndarray, shape) -> float:
    return float(np.max(np.linalg.norm(world_corners(a, shape[:3]) - world_corners(b, shape[:3]), axis=1)))


def _choose_form(header, shape, policy: Policy, report: Report) -> Tuple[np.ndarray, int, str]:
    """The RAS affine (file index -> world) the policy selects, its code and form name."""
    qcode, scode = int(header["qform_code"]), int(header["sform_code"])
    qform, sform = header.get_qform(), header.get_sform()
    if qcode > 0 and scode > 0:
        distance = _form_corner_distance(qform, sform, shape)
        if distance > policy.nifti_form_tolerance_mm and policy.nifti_form_disagreement != "ignore":
            message = f"qform and sform disagree by {distance:.4g} mm"
            if policy.nifti_form_disagreement == "error":
                raise GeometryError(message + " (policy nifti_form_disagreement='error')")
            report.warn(message)
    rule = policy.nifti_form
    if rule == "itk":
        # ITK (NiftiImageIO): sform when sform_code > 0 and it is rigid+scale (no shear), else qform.
        use_sform = scode > 0 and (qcode == 0 or not has_shear(sform[:3, :3]))
        rule = "sform" if use_sform else "qform"
    order = {"sform": ("sform", "qform"), "qform": ("qform", "sform"), "nibabel": ("sform", "qform")}[rule]
    for form in order:
        code = scode if form == "sform" else qcode
        if code > 0:
            report.choose("nifti_form", f"{form} (code {code})")
            return (sform if form == "sform" else qform), code, form
    return _no_codes_affine(header, shape, policy, report), 0, "none"


def _no_codes_affine(header, shape, policy: Policy, report: Report) -> np.ndarray:
    zooms = [float(z) for z in header["pixdim"][1:4]]
    rule = policy.nifti_no_codes
    if rule == "error":
        raise GeometryError("NIfTI has qform_code = sform_code = 0; choose policy nifti_no_codes "
                            "('spec', 'nibabel' or 'itk'): tools disagree on this geometry.")
    report.choose("nifti_no_codes", rule)
    if rule == "spec":       # NIfTI method 1: x = pixdim[1] * i, ... in RAS, origin 0
        return np.diag([*zooms, 1.0])
    if rule == "nibabel":    # nibabel's base affine: x flipped, centred
        return header.get_base_affine()
    # ITK: identity direction in LPS, origin 0  ->  RAS: diag(-z1, -z2, z3)
    return np.diag([-zooms[0], -zooms[1], zooms[2], 1.0])


# ------------------------------------------------------------------------------------------------
# plugin
# ------------------------------------------------------------------------------------------------

class NiftiPlugin(FormatPlugin):
    name = "nifti"
    extensions = (".nii", ".nii.gz", ".hdr", ".img", ".hdr.gz", ".img.gz")
    capabilities = Capabilities(
        max_space_axes=3, min_space_axes=1, max_ndim=7,
        non_space_axes=frozenset({"time", "component", "channel", "other"}),
        shear=True, oblique=True, label_table=False, key_value=False,
        dtypes=frozenset({"uint8", "int8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                          "float32", "float64", "complex64", "complex128"}),
    )
    fields = FieldTable(
        core_owned=frozenset(GEOMETRY_FIELDS + DERIVED_FIELDS + ("scl_slope", "scl_inter", "intent_code")),
        data_dependent=frozenset({"cal_min", "cal_max", "glmin", "glmax"}),
    )

    # -- sniff -------------------------------------------------------------------------------

    def sniff(self, path: PathLike) -> float:
        path = str(path)
        if not self.matches_extension(path):
            return 0.0
        try:
            header = _nib().load(path).header
        except Exception:
            return 0.0
        return 1.0 if hasattr(header, "get_qform") else 0.0

    # -- read --------------------------------------------------------------------------------

    def read(self, path: PathLike, policy: Policy) -> Tuple[List[Image], Report]:
        nib = _nib()
        report = Report("read", self.name, str(path))
        image = nib.load(str(path))
        # the header as stored: nibabel moves scl_slope/scl_inter into the data proxy on load
        holder = image.file_map.get("header", image.file_map["image"])   # single files only have "image"
        with holder.get_prepare_fileobj("rb") as handle:
            header = image.header_class.from_fileobj(handle)
        if not hasattr(header, "get_qform"):
            raise ValueError(f"{path} is an Analyze file, not NIfTI.")
        version = "2" if isinstance(header, nib.Nifti2Header) else "1"
        stored = np.asarray(image.dataobj.get_unscaled())
        shape = stored.shape
        axis_order = list(range(stored.ndim))[::-1]
        axes = self._axes(header, stored.ndim, report)
        if stored.ndim == 5 and shape[3] == 1 and int(header["intent_code"]) in VECTOR_INTENTS:
            # the NIfTI layout of a vector image (dim 4 = 1, components in dim 5): the singleton time axis
            # exists only because of the format, and the writer puts it back
            stored = stored[:, :, :, 0, :]
            axes = [axis for axis in axes if axis["name"] != "t"]
            axis_order = [3, 2, 1, 0]                 # NIfTI axes i, j, k, c (dim 4 left out)
            report.choose("nifti_vector_layout", "singleton dim 4 of the vector image not kept as an axis")
        ndim = stored.ndim
        array = np.ascontiguousarray(stored.transpose(tuple(range(ndim))[::-1]))

        affine_ras, code, form = _choose_form(header, shape + (1,) * max(0, 3 - len(shape)), policy, report)
        n_space = min(ndim, 3)
        space_names = ["x", "y", "z"][:n_space]
        matrix_lps = FLIP4 @ affine_ras
        matrix_lps = np.hstack([matrix_lps[:, :n_space], matrix_lps[:, 3:4]])
        space_name, reference = CODE_SPACES.get(code, ("patient", None))
        if code not in CODE_SPACES:
            if code:
                report.warn(f"unknown NIfTI xform code {code}; space named 'patient'")
            space_name = "patient"
        space = patient_space(matrix_lps, space_names, policy.world_frame, name=space_name, reference=reference)
        values = self._values(header)
        axes = self._vector_frame(header, axes, policy, report, values)
        meta = Meta(axes=axes, spaces=[space], values=values, shape=array.shape)
        source = {
            "format": "nifti",
            "format_version": version,
            "files": [os.path.basename(str(path))],
            "reader": {"name": "medvol"},
            "index_map": {"axis_order": axis_order, "flipped": [False] * ndim, "offset": [0] * ndim},
            "fingerprint": fingerprint.compute(array, meta),
            "header": header_to_dict(header),
        }
        meta = Meta(axes=axes, spaces=[space], values=values, source=[source], shape=array.shape)
        return [Image(array, meta)], report

    @staticmethod
    def _axes(header, ndim: int, report: Report) -> List[Dict[str, Any]]:
        """Axes in MLArray (reversed NIfTI) order."""
        space_unit_code = int(header["xyzt_units"]) & 0x07
        time_unit_code = int(header["xyzt_units"]) & 0x38
        if space_unit_code not in (0, 2):
            report.warn(f"NIfTI spatial unit is {SPACE_UNITS.get(space_unit_code, space_unit_code)}; geometry is stored in millimeter as written")
        nifti_axes: List[Dict[str, Any]] = [{"name": name, "type": "space"} for name in ("x", "y", "z")[:min(ndim, 3)]]
        if ndim >= 4:
            time = {"name": "t", "type": "time"}
            step = float(str(np.float32(header["pixdim"][4])))
            unit = TIME_UNITS.get(time_unit_code)
            if unit in ("hertz", "ppm", "rad/s"):
                time["type"] = "other"
            if step:
                time.update(start=float(str(np.float32(header["toffset"]))), step=step)
            if unit:
                time["unit"] = unit
            nifti_axes.append(time)
        if ndim >= 5:
            intent = int(header["intent_code"])
            nifti_axes.append({"name": "c", "type": "component" if intent in VECTOR_INTENTS else "other"})
        for extra in range(5, ndim):
            nifti_axes.append({"name": f"u{extra - 4}", "type": "other"})
        return nifti_axes[::-1]

    @staticmethod
    def _vector_frame(header, axes: List[Dict[str, Any]], policy: Policy, report: Report,
                      values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Labels of a 3-component vector axis (NIfTI dim 5).

        NIfTI does not say in which frame vector components are: ITK and ANTs write LPS components, NiftyReg
        RAS ones, under the same intent (KB ch03 §8.2). The frame must therefore be declared.
        """
        intent = int(header["intent_code"])
        index = next((i for i, axis in enumerate(axes) if axis["name"] == "c"), None)   # NIfTI dim 5
        if index is None or axes[index].get("type") != "component" or intent not in FRAME_INTENTS:
            return axes
        size = int(header["dim"][5])
        if size != 3:
            report.choose("vector_frame", f"not applied: intent {intent} with {size} components, not 3")
            return axes
        what = f"intent {intent} ({FRAME_INTENTS[intent]})"
        rule = policy.vector_frame
        if rule == "error":
            raise vectors.VectorFrameError(f"NIfTI {what}: the frame of the vector components is not declared by "
                                           "the format (policy vector_frame='error').")
        if rule == "from_file":
            raise vectors.undeclared("NIfTI", f"{what} does not declare the frame of its vector components "
                                     "(ITK/ANTs write LPS components, NiftyReg RAS ones)")
        words = vectors.POLICY_WORDS[rule]
        report.choose("vector_frame", f"{rule}: components {vectors.describe(words)} as declared by the policy; "
                                      "NIfTI does not store the frame")
        values["kind"] = "vector"
        return vectors.with_labels(axes, index, words)

    @staticmethod
    def _values(header) -> Dict[str, Any]:
        values: Dict[str, Any] = {"kind": "measurement"}
        slope, inter = float(header["scl_slope"]), float(header["scl_inter"])
        if np.isfinite(slope) and slope != 0:
            values["scale"] = float(str(np.float32(slope)))
            values["offset"] = float(str(np.float32(inter))) if np.isfinite(inter) else 0.0
        return values

    # -- write -------------------------------------------------------------------------------

    def write(self, image: Image, path: PathLike, policy: Policy, **options) -> Report:
        nib = _nib()
        report = Report("write", self.name, str(path))
        meta = image.meta
        path = str(path)
        source = next((entry for entry in meta.source if entry.format == "nifti"), None)
        # layout is the output encoding only: NIfTI-1 or -2 as in the source, or NIfTI-1 with layout='default'
        version = ((source.format_version if source is not None else None) or "1") if policy.layout == "source" else "1"

        image = Image(self._vector_components(image, policy, report), meta)
        order, nifti_axes = self._layout(meta, report)
        data = np.transpose(image.array, order)            # NIfTI order [i, j, k, t, c, ...] (C memory)
        if any(axis is None for axis in nifti_axes):        # vector image without time: dim 4 = 1, components in dim 5
            slot = nifti_axes.index(None)
            data = np.expand_dims(data, slot)
            nifti_axes[slot] = Axis(name="_", type="other")
        pair = path.lower().endswith((".hdr", ".img", ".hdr.gz", ".img.gz"))
        header_class = {("1", False): nib.nifti1.Nifti1Header, ("2", False): nib.nifti2.Nifti2Header,
                        ("1", True): nib.nifti1.Nifti1PairHeader, ("2", True): nib.nifti2.Nifti2PairHeader}[(version, pair)]
        header = header_class()
        if source is not None:
            self._apply_source(header, source, report)

        unchanged = source is not None and self._geometry_unchanged(source, image)
        self._set_data_fields(header, data)
        if unchanged:
            report.choose("geometry", "reused from source (unchanged)")
        else:
            self._set_geometry(header, meta, nifti_axes, data.shape, source, report, policy)
        self._set_values(header, meta, source)
        self._data_dependent(header, source, image, report)
        self._write_file(header, data, path)
        return report

    @staticmethod
    def _layout(meta: Meta, report: Report) -> Tuple[List[int], List[Any]]:
        """Array axis order for NIfTI: space axes (reversed: i fastest), then time, then the rest."""
        axes = list(meta.axes)
        space = meta.spaces[0] if meta.spaces else None
        mapped = list(space.from_array.array_axes) if space is not None else [a.name for a in axes if a.type == "space"]
        space_idx = [i for i, a in enumerate(axes) if a.name in mapped][::-1]
        time_idx = [i for i, a in enumerate(axes) if a.type == "time"]
        rest = [i for i in range(len(axes)) if i not in space_idx and i not in time_idx]
        if len(time_idx) > 1:
            raise GeometryError("NIfTI holds one time axis.")
        components = [i for i in rest if axes[i].type == "component"]
        if components and not time_idx and len(space_idx) == 3:
            # NIfTI's vector layout: dim 4 (time) = 1, the components in dim 5
            rest = components[:1] + [i for i in rest if i != components[0]][::-1]
            order = space_idx + rest
            return order, [axes[i] for i in space_idx] + [None] + [axes[i] for i in rest]
        if rest and not time_idx:
            # NIfTI dim 4 is time; a component axis without time gets a singleton time axis? Not representable
            # without changing the shape, so the first non-space axis takes dim 4.
            time_idx, rest = rest[:1], rest[1:]
            report.warn(f"axis {axes[time_idx[0]].name!r} ({axes[time_idx[0]].type}) is written as NIfTI dim 4")
        order = space_idx + time_idx + rest[::-1]
        return order, [axes[i] for i in order]

    @staticmethod
    def _vector_components(image: Image, policy: Policy, report: Report) -> np.ndarray:
        """The array to write: components as stored when their frame is LPS or RAS (the two conventions a NIfTI
        reader can assume, reported, since NIfTI cannot store the frame); other frames are re-expressed in the
        ``world_frame`` (a signed permutation, reported)."""
        index, words, unknown = vectors.frame_axis(image.meta)
        vectors.report_unknown(unknown, report)
        if index is None:
            return image.array
        name = vectors.NAMED_FRAMES.get(words)
        if name is not None:
            others = "ITK and ANTs assume LPS" if name == "RAS" else "NiftyReg and nibabel-based tools assume RAS"
            report.choose("vector_frame", f"components written as stored, in {vectors.describe(words)}")
            report.warn(f"NIfTI cannot store the frame of vector components: readers must assume {name} "
                        f"(MedVol: vector_frame='{name.lower()}'; {others})")
            return image.array
        target = FRAME_WORDS[policy.world_frame]
        array = vectors.permute(image.array, index, words, target)
        name = vectors.NAMED_FRAMES[tuple(target)]
        report.choose("vector_frame", f"components converted from {vectors.describe(words)} to "
                                      f"{vectors.describe(target)} (world_frame)")
        report.warn(f"vector component values changed: re-expressed from {vectors.letters(words)} to {name}, "
                    f"which NIfTI readers can assume; readers must assume {name} (MedVol: vector_frame='{name.lower()}')")
        return array

    @staticmethod
    def _apply_source(header, source, report: Report) -> None:
        values = source.header
        for key in header.keys():
            if key in values and key not in DERIVED_FIELDS:
                try:
                    header[key] = values[key]
                except (ValueError, TypeError) as error:
                    report.warn(f"source field {key} could not be restored: {error}")
        for extension in values.get("extensions", ()):
            header.extensions.append(_nib().nifti1.Nifti1Extension(int(extension["code"]), bytes(extension["content"])))

    @staticmethod
    def _geometry_unchanged(source, image: Image) -> bool:
        fp = source.fingerprint or {}
        return (fp.get("geometry") == fingerprint.geometry_hash(image.meta)
                and list(fp.get("shape", [])) == list(image.array.shape))

    @staticmethod
    def _set_data_fields(header, data: np.ndarray) -> None:
        header.set_data_dtype(data.dtype)
        header.set_data_shape(data.shape)

    def _set_geometry(self, header, meta: Meta, nifti_axes, shape, source, report: Report, policy: Policy) -> None:
        space = meta.spaces[0] if meta.spaces else None
        n_space = sum(1 for a in nifti_axes if a.type == "space")
        if space is None:
            report.warn("no space: written with qform_code = sform_code = 0 (index space)")
            header["qform_code"], header["sform_code"] = 0, 0
            header["pixdim"][1:4] = 1.0
            return
        lps = lps_matrix(space)
        names = [a.name for a in nifti_axes[:n_space]]
        columns = [list(space.from_array.array_axes).index(name) for name in names]
        linear = lps[:3, columns]
        if n_space < 3:
            # complete a 2-D (or 1-D) grid with orthogonal unit columns, right-handed
            basis = [linear[:, c] / np.linalg.norm(linear[:, c]) for c in range(n_space)]
            while len(basis) < 3:
                candidate = np.cross(basis[0], basis[1]) if len(basis) == 2 else np.linalg.svd(np.array(basis))[2][-1]
                basis.append(candidate / np.linalg.norm(candidate))
            linear = np.column_stack([linear] + [b[:, None] for b in basis[n_space:]])
        affine_ras = FLIP4 @ homogeneous(linear, lps[:3, -1])
        code = SPACE_CODES.get(space.name, 2)
        sform_code = int(source.header.get("sform_code", 0)) if source is not None else code
        qform_code = int(source.header.get("qform_code", 0)) if source is not None else code
        sform_code = sform_code or code
        header.set_sform(affine_ras, code=sform_code)
        if has_shear(linear):
            qform_code = int(source.header.get("qform_code", 0)) if source is not None else 0
            header.set_qform(affine_ras, code=qform_code, strip_shears=True)
            report.warn("sheared grid: the qform holds the closest rigid approximation"
                        f" (qform_code {qform_code}); the sform is exact")
        else:
            header.set_qform(affine_ras, code=qform_code or code)
        zooms = np.linalg.norm(linear, axis=0)
        pixdim = header["pixdim"]
        pixdim[1:4] = zooms
        space_unit = UNIT_SPACE_CODES.get(space.axes[0].unit, 2)
        # without a time axis the time-unit bits carry no geometry; keep the source's
        time_unit = int(source.header.get("xyzt_units", 0)) & 0x38 if source is not None else 0
        if len(nifti_axes) > 3 or (len(nifti_axes) > n_space and n_space < 3):
            extra = nifti_axes[n_space] if len(nifti_axes) > n_space else None
            if extra is not None and extra.type == "time":
                pixdim[4] = extra.step if extra.step is not None else 0.0
                header["toffset"] = extra.start if extra.start is not None else 0.0
                time_unit = UNIT_TIME_CODES.get(extra.unit, 0)
        header["pixdim"] = pixdim
        header["xyzt_units"] = space_unit | time_unit

    @staticmethod
    def _set_values(header, meta: Meta, source) -> None:
        values = meta.values
        if values.scale is not None:
            header["scl_slope"] = values.scale
            header["scl_inter"] = values.offset or 0.0
        else:
            original = source.header.get("scl_slope") if source is not None else None
            no_scaling = original is not None and (original == 0 or original != original)
            header["scl_slope"] = original if no_scaling else np.nan
            header["scl_inter"] = source.header.get("scl_inter", np.nan) if no_scaling else np.nan
        if source is None and any(a.type == "component" for a in meta.axes):
            header["intent_code"] = 1007

    @staticmethod
    def _data_dependent(header, source, image: Image, report: Report) -> None:
        if source is None:
            return
        fp = source.fingerprint or {}
        if fp and fingerprint.matches(fp, image.array, image.meta):
            return
        for key in ("cal_min", "cal_max", "glmin", "glmax"):
            if float(source.header.get(key, 0) or 0) != 0:
                report.warn(f"{key} dropped: the data changed since it was read")
            header[key] = 0

    @staticmethod
    def _write_file(header, data: np.ndarray, path: str) -> None:
        """Header (+ extensions) and data, written without letting nibabel touch the header."""
        from nibabel.openers import ImageOpener
        from nibabel.volumeutils import array_to_file

        lower = path.lower()
        data = np.asfortranarray(data)
        if lower.endswith((".hdr", ".img", ".hdr.gz", ".img.gz")):
            base = path[:-7] if lower.endswith(".gz") else path[:-4]
            gz = ".gz" if lower.endswith(".gz") else ""
            header["magic"] = header.pair_magic
            header["vox_offset"] = 0
            with ImageOpener(f"{base}.hdr{gz}", "wb") as handle:
                header.write_to(handle)
            with ImageOpener(f"{base}.img{gz}", "wb") as handle:
                array_to_file(data, handle, data.dtype, offset=0, order="F")
            return
        header["magic"] = header.single_magic
        minimum = int(header.single_vox_offset + header.extensions.get_sizeondisk())
        header["vox_offset"] = int(np.ceil(minimum / 16.0) * 16)
        with ImageOpener(path, "wb") as handle:
            header.write_to(handle)
            handle.write(b"\x00" * (int(header["vox_offset"]) - minimum))
            array_to_file(data, handle, data.dtype, offset=None, order="F")
