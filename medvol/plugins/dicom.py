"""DICOM plugin (spec sections 7.5 and 7.6).

Reading, phase 1: single-frame series (CT, MR, PET, ...), regular or gantry-tilted.

- pydicom parses files and decodes pixels; MedVol assembles the series and builds the geometry itself.
- Files are grouped by SeriesInstanceUID, then by in-plane geometry (``dicom_stacks``), sorted by the
  projection of ImagePositionPatient on the slice normal (never by InstanceNumber), and checked for
  regular spacing (``dicom_irregular_spacing``) and tilt (``dicom_tilt``).
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
            raise NotImplementedError("dicom_irregular_spacing='resample' is not implemented yet")
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
            raise NotImplementedError("dicom_tilt='resample' is not implemented yet")
        report.choose("dicom_tilt", "exact")
        report.warn(f"gantry tilt of {angle:.3f} degrees: stored as an exact sheared grid")
    return [(stack, step)]


# ------------------------------------------------------------------------------------------------
# plugin
# ------------------------------------------------------------------------------------------------

class DicomPlugin(FormatPlugin):
    name = "dicom"
    extensions = (".dcm",)
    capabilities = Capabilities(
        max_space_axes=3, min_space_axes=2, max_ndim=3, shear=True, oblique=True,
        label_table=False, key_value=False,
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

    def read(self, path: PathLike, policy: Policy) -> Tuple[List[Image], Report]:
        pydicom = _pydicom()
        report = Report("read", self.name, str(path))
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
        for datasets in series.values():
            if "NumberOfFrames" in datasets[0] and int(datasets[0].NumberOfFrames) > 1:
                raise NotImplementedError("enhanced multi-frame DICOM is phase 2 of the DICOM reader")
            for stack in _stacks(datasets, policy, report):
                for slices, step in _sorted_stack(stack, policy, report):
                    images.append(self._image(slices, step, policy, report))
        return images, report

    def _image(self, slices: List, step: np.ndarray, policy: Policy, report: Report) -> Image:
        first = slices[0]
        iop = np.array(first.ImageOrientationPatient, dtype=np.float64)
        row_spacing, column_spacing = (float(v) for v in first.PixelSpacing)
        # columns: slice k, row j, column i (array order); IOP[:3] = direction of increasing column index
        linear = np.column_stack([step, iop[3:] * row_spacing, iop[:3] * column_spacing])
        matrix = homogeneous(linear, np.array(first.ImagePositionPatient, dtype=np.float64))
        reference = None
        if getattr(first, "FrameOfReferenceUID", None):
            reference = {"scheme": "dicom_frame_of_reference", "id": str(first.FrameOfReferenceUID)}
        space = patient_space(matrix, ["z", "y", "x"], policy.world_frame, reference=reference)

        array = np.stack([ds.pixel_array for ds in slices])
        values = self._values(slices, report)
        headers = [_file_dict(ds) for ds in slices]
        shared, per_file = split_shared(headers)
        header = {"shared": shared, "per_file": per_file, "file_names": [str(ds.filename_relative) for ds in slices]}
        axes = ["z", "y", "x"]
        meta = Meta(axes=axes, spaces=[space], values=values, shape=array.shape)
        source = {
            "format": "dicom",
            "files": header["file_names"],
            "reader": {"name": "medvol"},
            "index_map": {"axis_order": [2, 1, 0], "flipped": [False, False, False], "offset": [0, 0, 0]},
            "fingerprint": fingerprint.compute(array, meta),
            "header": header,
        }
        return Image(array, Meta(axes=axes, spaces=[space], values=values, source=[source], shape=array.shape))

    @staticmethod
    def _values(slices: List, report: Report) -> Dict[str, Any]:
        def rescale(ds):
            return (float(getattr(ds, "RescaleSlope", 1) or 1), float(getattr(ds, "RescaleIntercept", 0) or 0))

        pairs = {rescale(ds) for ds in slices}
        if len(pairs) > 1:
            raise GeometryError("rescale slope/intercept differ between slices; one MLArray holds one scale "
                                "(per-slice rescaling is not supported yet)")
        first = slices[0]
        values: Dict[str, Any] = {"kind": "measurement"}
        if "RescaleSlope" in first or "RescaleIntercept" in first:
            values["scale"], values["offset"] = pairs.pop()
        rescale_type = str(getattr(first, "RescaleType", "") or "")
        if rescale_type.upper() == "HU" or (not rescale_type and str(getattr(first, "Modality", "")) == "CT"):
            values["unit"] = "hounsfield_unit"
        if "PixelPaddingValue" in first:
            values["fill"] = int(first.PixelPaddingValue)
        return values

    def write(self, image: Image, path: PathLike, policy: Policy) -> Report:
        """Re-emit the original series into the directory ``path`` (spec 7.6)."""
        from medvol.plugins.dicom_write import write_series

        return write_series(image, str(path), policy)
