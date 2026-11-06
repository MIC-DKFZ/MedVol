"""DICOM plugin (spec sections 7.5 and 7.6).

Reading, phase 1: single-frame series (CT, MR, PET, ...), regular or gantry-tilted. Phase 2
(enhanced multi-frame, ``dicom_enhanced.py``), phase 3 (DICOM SEG, ``dicom_seg.py``) and phase 4
(RTSTRUCT, ``dicom_rtstruct.py``) are dispatched from ``DicomPlugin.read`` and ``DicomPlugin.write``.

- pydicom parses files and decodes pixels; MedVol assembles the series and builds the geometry itself.
- Files are grouped by SeriesInstanceUID, then by in-plane geometry (``dicom_stacks``), sorted by the
  projection of ImagePositionPatient on the slice normal (never by InstanceNumber), and checked for
  regular spacing (``dicom_irregular_spacing``) and tilt (``dicom_tilt``).
- ``dicom_tilt='resample'`` and ``dicom_irregular_spacing='resample'`` resample the stack onto a
  regular grid without shear (``dicom_resample.py``); ``dicom_rescale='apply'`` stores physical
  values when the rescale differs between slices (``dicom_rescale.py``). Both are opt-in and reported.
- The array is [slice, row, column] in C order; the patient space is built in DICOM's own LPS frame.
- ``source.header`` holds ``shared`` (identical in every file), ``per_file`` (only the fields that
  differ, in slice order) and ``file_names``; public elements by keyword, private ones by tag with
  ``{vr, value}``; the file meta information under ``file_meta``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from mlarray import Meta, fingerprint

from medvol.geometry import GeometryError, homogeneous, patient_space
from medvol.image import Image
from medvol.plugins.base import Capabilities, FieldTable, FormatPlugin, PathLike
from medvol.policy import Policy
from medvol.report import Report

ORIENTATION_TOLERANCE = 1e-4      # direction cosines
SPACING_TOLERANCE = 1e-3          # relative slice-spacing deviation
TILT_TOLERANCE = 1e-4             # mm of in-plane drift per mm of slice step

#: Elements regenerated from the core (spec 4); PixelData always comes from the array.
CORE_OWNED = frozenset({
    "Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient", "RescaleSlope",
    "RescaleIntercept", "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "SamplesPerPixel",
    "NumberOfFrames", "PixelData",
})
DATA_DEPENDENT = frozenset({
    "SmallestImagePixelValue", "LargestImagePixelValue", "SmallestPixelValueInSeries", "LargestPixelValueInSeries",
    "SliceLocation", "InstanceNumber", "SOPInstanceUID",
})


def _pydicom():
    import pydicom

    return pydicom


# ------------------------------------------------------------------------------------------------
# readable header representation (MLArray spec 6.4)
# ------------------------------------------------------------------------------------------------

def _value(element) -> Any:
    """An element's value as plain msgpack data."""
    pydicom = _pydicom()
    vr, value = element.VR, element.value
    if vr == "SQ":
        return [dataset_to_dict(item) for item in value]
    if value is None or value == "":
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, pydicom.multival.MultiValue) or isinstance(value, (list, tuple)):
        return [_scalar(v, vr) for v in value]
    return _scalar(value, vr)


def _scalar(value, vr: str) -> Any:
    pydicom = _pydicom()
    if vr == "DS":
        return float(value)
    if vr == "IS":
        return int(value)
    if vr == "AT":
        return f"({value.group:04X},{value.element:04X})"
    if isinstance(value, pydicom.valuerep.PersonName):
        return str(value)
    if isinstance(value, (int, float, str, bytes)):
        return value
    return str(value)


def dataset_to_dict(dataset) -> Dict[str, Any]:
    """Public elements by keyword, private (or unknown) ones by tag with ``{vr, value}``; no pixel data."""
    out: Dict[str, Any] = {}
    for element in dataset:
        if element.tag == 0x7FE00010:
            continue
        keyword = element.keyword
        if keyword and not element.tag.is_private:
            out[keyword] = _value(element)
        else:
            out[f"({element.tag.group:04X},{element.tag.element:04X})"] = {"vr": element.VR, "value": _value(element)}
    return out


def _file_dict(dataset) -> Dict[str, Any]:
    out = dataset_to_dict(dataset)
    if getattr(dataset, "file_meta", None) is not None:
        out = {"file_meta": dataset_to_dict(dataset.file_meta), **out}
    return out


def split_shared(headers: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """(fields equal in every header, per-header remainders), keeping first-file field order."""
    keys = list(headers[0])
    shared = {key: headers[0][key] for key in keys if all(key in h and h[key] == headers[0][key] for h in headers)}
    per_file = [{key: value for key, value in h.items() if key not in shared} for h in headers]
    return shared, per_file


# ------------------------------------------------------------------------------------------------
# series assembly
# ------------------------------------------------------------------------------------------------

def _is_dicom_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _files(path: PathLike) -> List[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and _is_dicom_file(p))
    return [path]


def _geometry_key(ds) -> Tuple:
    iop = tuple(float(v) for v in ds.ImageOrientationPatient)
    spacing = tuple(float(v) for v in ds.PixelSpacing)
    return (int(ds.Rows), int(ds.Columns), spacing, iop)


def _same(a: Tuple, b: Tuple) -> bool:
    if a[:2] != b[:2]:
        return False
    return np.allclose(a[2], b[2], rtol=1e-6) and np.allclose(a[3], b[3], atol=ORIENTATION_TOLERANCE)


def _stacks(datasets: List, policy: Policy, report: Report) -> List[List]:
    """Group the datasets of one series into geometric stacks per ``dicom_stacks``."""
    groups: List[Tuple[Tuple, List]] = []
    for ds in datasets:
        key = _geometry_key(ds)
        for group_key, members in groups:
            if _same(group_key, key):
                members.append(ds)
                break
        else:
            groups.append((key, [ds]))
    stacks = [members for _, members in groups]
    if len(stacks) > 1:
        sizes = sorted((len(s) for s in stacks), reverse=True)
        if policy.dicom_stacks == "error":
            raise GeometryError(f"series {datasets[0].SeriesInstanceUID} holds {len(stacks)} geometric stacks {sizes}"
                                " (policy dicom_stacks='error')")
        if policy.dicom_stacks == "largest":
            largest = max(stacks, key=len)
            report.warn(f"series has {len(stacks)} geometric stacks {sizes}; kept the largest ({len(largest)} files)")
            report.choose("dicom_stacks", "largest")
            return [largest]
        report.warn(f"series has {len(stacks)} geometric stacks {sizes}; read as separate images")
        report.choose("dicom_stacks", "split")
    return stacks


def _sorted_stack(stack: List, policy: Policy, report: Report) -> List[Tuple[List, np.ndarray]]:
    """Slices sorted along the normal; split into regular runs if the policy says so.

    Returns [(datasets, step vector), ...].
    """
    iop = np.array(stack[0].ImageOrientationPatient, dtype=np.float64)
    normal = np.cross(iop[:3], iop[3:])
    positions = np.array([np.array(ds.ImagePositionPatient, dtype=np.float64) for ds in stack])
    order = np.argsort(positions @ normal, kind="stable")
    stack = [stack[i] for i in order]
    positions = positions[order]
    if len(stack) == 1:
        thickness = float(getattr(stack[0], "SpacingBetweenSlices", 0) or getattr(stack[0], "SliceThickness", 0) or 1.0)
        report.warn(f"single slice: slice spacing taken from {'SpacingBetweenSlices' if getattr(stack[0], 'SpacingBetweenSlices', None) else 'SliceThickness'} ({thickness} mm)")
        return [(stack, normal * thickness)]
    steps = np.diff(positions, axis=0)
    distances = steps @ normal
    if np.any(np.abs(distances) < 1e-6):
        raise GeometryError("several files share one slice position (e.g. several time points or echoes); "
                            "this is not a single-frame volume (phase 2 handles 4-D series)")
    mean = float(np.mean(distances))
    irregular = np.abs(distances - mean) > SPACING_TOLERANCE * abs(mean)
    if np.any(irregular):
        if policy.dicom_irregular_spacing == "error":
            raise GeometryError(f"irregular slice spacing {np.round(distances, 4).tolist()} mm "
                                "(policy dicom_irregular_spacing='error')")
        if policy.dicom_irregular_spacing == "resample":
            return [(stack, _resampling(positions, normal, distances, policy, report, irregular=True))]
        runs: List[List[int]] = [[0]]
        reference = None
        for gap, distance in enumerate(distances):          # gap between slices gap and gap + 1
            if reference is None or abs(distance - reference) <= SPACING_TOLERANCE * abs(reference):
                runs[-1].append(gap + 1)
                reference = distance if reference is None else reference
            else:
                runs.append([gap + 1])
                reference = None
        report.choose("dicom_irregular_spacing", "split")
        report.warn(f"irregular slice spacing: split into {len(runs)} regular runs {[len(r) for r in runs]}")
        return [item for run in runs for item in _sorted_stack([stack[i] for i in run], policy, report)]
    step = (positions[-1] - positions[0]) / (len(stack) - 1)
    drift = np.linalg.norm(steps - step, axis=1)
    if np.any(drift > TILT_TOLERANCE * np.linalg.norm(step) + 1e-6):
        raise GeometryError("slice positions do not lie on a straight line; cannot form one grid")
    in_plane = step - (step @ normal) * normal
    if np.linalg.norm(in_plane) > TILT_TOLERANCE * abs(step @ normal):
        angle = float(np.degrees(np.arctan2(np.linalg.norm(in_plane), abs(step @ normal))))
        if policy.dicom_tilt == "error":
            raise GeometryError(f"gantry tilt of {angle:.3f} degrees (policy dicom_tilt='error')")
        if policy.dicom_tilt == "resample":
            return [(stack, _resampling(positions, normal, distances, policy, report, irregular=False, source_step=step))]
        report.choose("dicom_tilt", "exact")
        report.warn(f"gantry tilt of {angle:.3f} degrees: stored as an exact sheared grid")
    return [(stack, step)]


def _resampling(positions: np.ndarray, normal: np.ndarray, gaps: np.ndarray, policy: Policy, report: Report,
                irregular: bool, source_step: np.ndarray = None):
    """The plan for ``dicom_irregular_spacing='resample'`` or ``dicom_tilt='resample'`` (``dicom_resample``).

    Resampling always yields a grid without shear, so a tilted irregular stack needs
    ``dicom_tilt='resample'`` as well; ``exact`` (keep the shear) and ``error`` refuse it.
    """
    from medvol.plugins import dicom_resample

    reasons = []
    overall = positions[-1] - positions[0]
    in_plane = overall - (overall @ normal) * normal
    tilted = np.linalg.norm(in_plane) > TILT_TOLERANCE * abs(overall @ normal)
    if irregular:
        reasons.append("irregular slice spacing")
        report.warn(f"irregular slice spacing {np.round(gaps, 4).tolist()} mm")
    if tilted:
        angle = float(np.degrees(np.arctan2(np.linalg.norm(in_plane), abs(overall @ normal))))
        if policy.dicom_tilt == "error":
            raise GeometryError(f"gantry tilt of {angle:.3f} degrees (policy dicom_tilt='error')")
        if policy.dicom_tilt == "exact":
            raise GeometryError(f"the stack is irregularly spaced and tilted by {angle:.3f} degrees: resampling "
                                "(dicom_irregular_spacing='resample') gives a grid without shear, which drops the tilt "
                                "that dicom_tilt='exact' keeps; use dicom_tilt='resample' as well, or "
                                "dicom_irregular_spacing='split'")
        reasons.append(f"gantry tilt of {angle:.3f} degrees")
    plan = dicom_resample.plan(positions, normal, tuple(reasons), source_step=source_step)
    if irregular:
        report.choose("dicom_irregular_spacing", dicom_resample.method(plan))
    if tilted:
        report.choose("dicom_tilt", dicom_resample.method(plan))
    return plan


def _resample_volume(array: np.ndarray, resampling, orientation: np.ndarray, pixel_spacing, values: Dict[str, Any],
                     space_of, axes: List[str], report: Report):
    """Resample ``array`` (..., slices, rows, columns) per ``resampling`` (``dicom_resample``).

    Returns (array, source fingerprint, values). The fingerprint describes the source data on its
    source grid (the sheared grid of a tilted stack; no geometry hash for an irregular stack, which
    has no single grid), so it does not match the resampled image and a DICOM write regenerates
    geometry and UIDs or refuses.
    """
    from medvol.plugins import dicom_resample

    if resampling.source_step is not None:
        source_space = space_of(resampling.source_step, resampling.origin)
        fp = fingerprint.compute(array, Meta(axes=axes, spaces=[source_space], values=values, shape=array.shape))
    else:
        fp = {"shape": [int(v) for v in array.shape], "dtype": array.dtype.name,
              "geometry": "none: irregular source slice positions", "data": fingerprint.data_hash(array)}
    fill = values.get("fill")
    resampled, stats = dicom_resample.apply(array, resampling, orientation, pixel_spacing, fill)
    values = dict(values)
    if stats["outside_per_volume"] and fill is None:
        values["fill"] = float("nan")
    for line in dicom_resample.describe(resampling, stats, values.get("fill")):
        report.warn(line)
    report.warn(f"resampling changed the values: stored as {resampled.dtype.name}; the source fingerprint no longer "
                "matches, so a DICOM write regenerates geometry and UIDs or is refused")
    return resampled, fp, values


# ------------------------------------------------------------------------------------------------
# plugin
# ------------------------------------------------------------------------------------------------

class DicomPlugin(FormatPlugin):
    name = "dicom"
    extensions = (".dcm",)
    capabilities = Capabilities(
        max_space_axes=3, min_space_axes=2, max_ndim=4, shear=True, oblique=True,
        non_space_axes=frozenset({"time", "other", "channel"}),   # 4-D enhanced; SEG channel layout
        label_table=True,                                          # SEG segments
        key_value=False,
    )
    fields = FieldTable(core_owned=CORE_OWNED, data_dependent=DATA_DEPENDENT)

    def sniff(self, path: PathLike) -> float:
        path = Path(path)
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file():
                    return 1.0 if _is_dicom_file(candidate) else 0.0
            return 0.0
        return 1.0 if _is_dicom_file(path) else 0.0

    def read(self, path: PathLike, policy: Policy, reference: PathLike = None) -> Tuple[List[Image], Report]:
        """All images in ``path``.

        ``reference``: for an RTSTRUCT, the referenced image series (``dicom_rtstruct``). An
        RTSTRUCT whose referenced series is read together with it (same directory) is rasterised
        onto that series without ``reference``.
        """
        from medvol.plugins import dicom_rtstruct

        if reference is not None:
            return dicom_rtstruct.read_rtstruct(path, reference, policy)
        report = Report("read", self.name, str(path))
        images, structure_sets = self.read_images(path, policy, report)
        for ds in structure_sets:
            images.append(dicom_rtstruct.rasterise_onto(ds, images, policy, report))
        return images, report

    def read_images(self, path: PathLike, policy: Policy, report: Report) -> Tuple[List[Image], List]:
        """(images of every image and SEG series in ``path``, RTSTRUCT datasets, not yet rasterised)."""
        from medvol.plugins import dicom_rtstruct

        pydicom = _pydicom()
        files = _files(path)
        if not files:
            raise ValueError(f"No DICOM files in {path}.")
        series: Dict[str, List] = defaultdict(list)
        for file in files:
            ds = pydicom.dcmread(str(file))
            ds.filename_relative = os.path.relpath(file, path) if Path(path).is_dir() else file.name
            series[str(getattr(ds, "SeriesInstanceUID", ""))].append(ds)
        if len(series) > 1:
            report.warn(f"{len(series)} series found; read as separate images")
        images = []
        structure_sets = []
        for datasets in series.values():
            # --- phase 4: RTSTRUCT, rasterised after the image series (dicom_rtstruct.py) ---
            if dicom_rtstruct.is_rtstruct(datasets[0]):
                structure_sets.extend(datasets)
                continue
            # --- phases 2 and 3: SEG and enhanced multi-frame objects (dicom_seg.py, dicom_enhanced.py) ---
            from medvol.plugins import dicom_enhanced, dicom_seg

            if dicom_seg.is_seg_dataset(datasets[0]):
                for ds in datasets:
                    images.extend(dicom_seg.read_seg(ds, policy, report))
                continue
            if dicom_enhanced.is_enhanced(datasets[0]):
                images.extend(dicom_enhanced.read_enhanced(datasets, policy, report))
                continue
            if "NumberOfFrames" in datasets[0] and int(datasets[0].NumberOfFrames) > 1:
                raise NotImplementedError("multi-frame DICOM without functional groups (e.g. multi-frame secondary "
                                          "capture, NM) is not supported")
            # --- end of phases 2 and 3 ---
            for stack in _stacks(datasets, policy, report):
                for slices, step in _sorted_stack(stack, policy, report):
                    images.append(self._image(slices, step, policy, report))
        return images, structure_sets

    def _image(self, slices: List, step, policy: Policy, report: Report) -> Image:
        """The image of sorted slices; ``step`` is the slice step vector or a ``dicom_resample.Resampling``."""
        from medvol.plugins import dicom_resample

        first = slices[0]
        iop = np.array(first.ImageOrientationPatient, dtype=np.float64)
        row_spacing, column_spacing = (float(v) for v in first.PixelSpacing)
        resampling = step if isinstance(step, dicom_resample.Resampling) else None
        source_step = step if resampling is None else resampling.source_step
        reference = None
        if getattr(first, "FrameOfReferenceUID", None):
            reference = {"scheme": "dicom_frame_of_reference", "id": str(first.FrameOfReferenceUID)}

        def space_of(slice_step, origin):
            # columns: slice k, row j, column i (array order); IOP[:3] = direction of increasing column index
            linear = np.column_stack([slice_step, iop[3:] * row_spacing, iop[:3] * column_spacing])
            return patient_space(homogeneous(linear, origin), ["z", "y", "x"], policy.world_frame, reference=reference)

        array = np.stack([ds.pixel_array for ds in slices])
        values, pairs = self._values(slices, policy, report)
        if pairs is not None:
            from medvol.plugins.dicom_rescale import to_physical

            array, kind = to_physical(array, pairs)
            report.choose("dicom_rescale", f"apply: physical values slope * stored + intercept per slice, {kind}")
            report.warn(f"rescale slope/intercept differ between slices {sorted(set(pairs))}: stored the physical "
                        f"values as {array.dtype.name} (policy dicom_rescale='apply'); values.scale/offset not set")
        axes = ["z", "y", "x"]
        headers = [_file_dict(ds) for ds in slices]
        shared, per_file = split_shared(headers)
        header = {"shared": shared, "per_file": per_file, "file_names": [str(ds.filename_relative) for ds in slices]}
        if resampling is None:
            space = space_of(source_step, np.array(first.ImagePositionPatient, dtype=np.float64))
            fp = fingerprint.compute(array, Meta(axes=axes, spaces=[space], values=values, shape=array.shape))
        else:
            array, fp, values = _resample_volume(array, resampling, iop, (row_spacing, column_spacing), values,
                                                 space_of, axes, report)
            space = space_of(resampling.step, resampling.origin)
        source = {
            "format": "dicom",
            "files": header["file_names"],
            "reader": {"name": "medvol"},
            "index_map": {"axis_order": [2, 1, 0], "flipped": [False, False, False], "offset": [0, 0, 0]},
            "fingerprint": fp,
            "header": header,
        }
        return Image(array, Meta(axes=axes, spaces=[space], values=values, source=[source], shape=array.shape))

    @staticmethod
    def _values(slices: List, policy: Policy, report: Report) -> Tuple[Dict[str, Any], Any]:
        """(values, per-slice rescale pairs to apply or None)."""
        def rescale(ds):
            return (float(getattr(ds, "RescaleSlope", 1) or 1), float(getattr(ds, "RescaleIntercept", 0) or 0))

        pairs = [rescale(ds) for ds in slices]
        apply = None
        first = slices[0]
        values: Dict[str, Any] = {"kind": "measurement"}
        if len(set(pairs)) > 1:
            if policy.dicom_rescale == "error":
                raise GeometryError("rescale slope/intercept differ between slices; one MLArray holds one scale "
                                    "(policy dicom_rescale='error'; dicom_rescale='apply' stores the physical values)")
            apply = pairs
        elif "RescaleSlope" in first or "RescaleIntercept" in first:
            values["scale"], values["offset"] = pairs[0]
        rescale_type = str(getattr(first, "RescaleType", "") or "")
        if rescale_type.upper() == "HU" or (not rescale_type and str(getattr(first, "Modality", "")) == "CT"):
            values["unit"] = "hounsfield_unit"
        if "PixelPaddingValue" in first:
            if apply is None:
                values["fill"] = int(first.PixelPaddingValue)
            else:
                report.warn(f"PixelPaddingValue {int(first.PixelPaddingValue)} is a stored value; with per-slice "
                            "rescaling applied it maps to a different physical value per slice, so values.fill is not set")
        return values, apply

    def write(self, image: Image, path: PathLike, policy: Policy, reference=None, **options) -> Report:
        """Write ``image`` as DICOM (spec 7.6).

        - label maps and SEG sources: a DICOM SEG via highdicom (``dicom_seg.write_seg``; the
          referenced image series is given as ``reference=``);
        - enhanced multi-frame sources: the original file(s) re-emitted (``dicom_enhanced``);
        - single-frame sources: the original series re-emitted into the directory ``path``.
        """
        from medvol.plugins import dicom_seg

        if dicom_seg.is_segmentation(image):
            return dicom_seg.write_seg(image, str(path), policy, reference=reference)
        source = next((entry for entry in image.meta.source if entry.format == "dicom"), None)
        if source is not None and "frames" in source.header:
            from medvol.plugins.dicom_enhanced import write_enhanced

            return write_enhanced(image, str(path), policy)
        from medvol.plugins.dicom_write import write_series

        return write_series(image, str(path), policy)
