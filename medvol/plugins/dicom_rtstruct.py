"""DICOM reading, phase 4: RTSTRUCT -> label map (spec 7.5).

An RT Structure Set stores each ROI as planar contours in patient coordinates (mm). It has no grid
of its own, so it is rasterised onto the grid of its referenced image series. Tools disagree on
the rasterisation rule by 2-13 % of the voxels (KB ch02 D6), so MedVol uses one explicit rule.

**Rasterisation rule.**

- A contour belongs to the slice whose plane it lies on: its points, mapped into the reference
  grid's index space, must lie on one slice plane, within ``CONTOUR_PLANE_TOLERANCE_MM`` along
  the slice normal. Contours off every slice plane, or beyond the first or last slice, are not
  rasterised and are reported (``on_loss``: ``warn`` drops them, ``error`` raises).
- Only closed planar contours are areas: ``CLOSED_PLANAR`` and ``CLOSEDPLANAR_XOR``. ``POINT``,
  ``OPEN_PLANAR`` and ``OPEN_NONPLANAR`` contours are reported and skipped. The last point
  connects back to the first.
- A voxel is inside an ROI if its **centre** lies inside the ROI's contours on its slice by the
  **even-odd rule**, taken over all closed contours of that ROI on that slice together: a ray from
  the voxel centre towards increasing column index crosses an odd number of contour edges. A hole
  given as a separate contour inside an outer contour is therefore outside, an island inside the
  hole inside again (this is also the XOR of the contours, as for ``CLOSEDPLANAR_XOR``).
- A voxel centre exactly on an edge follows the half-open crossing test, i.e. the classic PNPOLY
  test: an edge from row ``y0`` to row ``y1`` counts for the centres of row ``y`` when
  ``(y0 > y) != (y1 > y)`` (``min(y0, y1) <= y < max(y0, y1)``), and only if it crosses that row
  strictly to the right of the centre (towards increasing column index). It is deterministic but
  not symmetric: a centre on a left edge or at a lower row bound is inside, on a right edge or at
  an upper row bound outside. Rows and columns are the reference image's pixel rows and columns.
- Points are converted with the inverse of the reference grid's matrix, so oblique and sheared
  (gantry-tilted) grids are handled exactly; nothing is snapped.

This is the "own" rule of KB experiment ``verify_dicom_real/04``, which agrees voxel for voxel
with dcmrtstruct2nii on real data.

**Output.** One Image per RTSTRUCT, on the reference image's space (same matrix and Frame of
Reference), with ``values.kind = "label"``. Labels come from the Structure Set ROI Sequence: the
ROI Number is the label value, ``name`` is ROI Name, ``color`` is the ROI Display Color of the ROI
Contour Sequence, ``code`` is the first item of the RT ROI Identification Code Sequence of the
RT ROI Observations (``{scheme, value, meaning}``) where present.

- ROIs that do not overlap (and whose numbers are positive): ``label_layout: index``, array
  ``[z, y, x]`` holding ROI numbers (uint8, uint16 or uint32 by the largest number).
- Overlapping ROIs: ``label_layout: channel``, array ``[c, z, y, x]`` of 0/1 (uint8), channel
  ``c`` = the ``c``-th ROI in ascending ROI Number, labels keyed by channel index.

``source``: the RTSTRUCT header as ``shared`` / ``per_file`` / ``file_names`` (one file). The
reference series is not copied into the RTSTRUCT image's ``source``; the report names it.

**API.** ``read_rtstruct(path, reference, policy)``; ``DicomPlugin.read(path, policy,
reference=...)`` calls it. A directory holding the RTSTRUCT and its referenced series is read by
``DicomPlugin.read`` without ``reference``: the RTSTRUCT is rasterised onto the image series
read from the same directory. Writing RTSTRUCT is not supported.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from mlarray import Meta, fingerprint

from medvol.geometry import GeometryError, lps_matrix
from medvol.image import Image
from medvol.plugins.base import PathLike
from medvol.policy import Policy
from medvol.report import Report

RTSTRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
CLOSED_TYPES = frozenset({"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"})
#: mm along the slice normal: a contour farther than this from every slice plane is not rasterised.
CONTOUR_PLANE_TOLERANCE_MM = 1e-2

RULE = ("voxel centre inside: even-odd rule over all closed planar contours of the ROI on the voxel's slice "
        "(holes as separate contours), half-open crossing test for centres on an edge")


def is_rtstruct(ds) -> bool:
    return str(getattr(ds, "SOPClassUID", "")) == RTSTRUCT_STORAGE or str(getattr(ds, "Modality", "")) == "RTSTRUCT"


# ------------------------------------------------------------------------------------------------
# point in polygon
# ------------------------------------------------------------------------------------------------

def even_odd(polygons: List[np.ndarray], rows: int, columns: int) -> np.ndarray:
    """Pixel centres (j, i) inside ``polygons`` (each (N, 2) in (row, column) index coordinates).

    Even-odd rule over all edges of all polygons together; each polygon is closed (last point to
    first). Half-open crossing test: edge (p, q) counts for row j if ``(p.y > j) != (q.y > j)``, and
    toggles the centres strictly left of the crossing.
    """
    parity = np.zeros((rows, columns), dtype=bool)
    if not polygons:
        return parity
    starts = np.concatenate([np.asarray(p, dtype=np.float64) for p in polygons])
    ends = np.concatenate([np.roll(np.asarray(p, dtype=np.float64), -1, axis=0) for p in polygons])
    y0, x0, y1, x1 = starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1]
    columns_index = np.arange(columns, dtype=np.float64)
    low = max(int(np.floor(min(y0.min(), y1.min()))), 0)
    high = min(int(np.ceil(max(y0.max(), y1.max()))), rows - 1)
    for j in range(low, high + 1):
        crossing = (y0 > j) != (y1 > j)
        if not np.any(crossing):
            continue
        a0, b0, a1, b1 = y0[crossing], x0[crossing], y1[crossing], x1[crossing]
        x = b0 + (j - a0) * (b1 - b0) / (a1 - a0)
        parity[j] = (np.count_nonzero(x[:, None] > columns_index[None, :], axis=0) % 2).astype(bool)
    return parity


# ------------------------------------------------------------------------------------------------
# reading
# ------------------------------------------------------------------------------------------------

def _datasets(path: PathLike) -> List:
    from medvol.plugins.dicom import _files, _pydicom

    pydicom = _pydicom()
    out = []
    for file in _files(path):
        ds = pydicom.dcmread(str(file))
        if is_rtstruct(ds):
            ds.filename_relative = file.name if not Path(path).is_dir() else str(file.relative_to(path))
            out.append(ds)
    if not out:
        raise ValueError(f"No RTSTRUCT in {path}.")
    return out


def read_rtstruct(path: PathLike, reference: PathLike, policy: Policy) -> Tuple[List[Image], Report]:
    """The label map of every RTSTRUCT in ``path``, rasterised onto the image series ``reference``.

    ``reference`` is read with the same policy (a directory or file(s) that ``medvol`` reads as
    DICOM); RTSTRUCTs in it are ignored. Returns (images, report).
    """
    from medvol.plugins.dicom import DicomPlugin

    report = Report("read", "dicom", str(path))
    images, _ = DicomPlugin().read_images(reference, policy, report)
    report.choose("rtstruct_reference", str(reference))
    return [rasterise_onto(ds, images, policy, report) for ds in _datasets(path)], report


def _referenced_series(ds) -> List[str]:
    uids = []
    for frame in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for series in getattr(study, "RTReferencedSeriesSequence", []):
                if "SeriesInstanceUID" in series:
                    uids.append(str(series.SeriesInstanceUID))
    return uids


def _referenced_instances(ds) -> set:
    out = set()
    for frame in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for series in getattr(study, "RTReferencedSeriesSequence", []):
                for image in getattr(series, "ContourImageSequence", []):
                    out.add(str(image.ReferencedSOPInstanceUID))
    for roi in getattr(ds, "ROIContourSequence", []):
        for contour in getattr(roi, "ContourSequence", []):
            for image in getattr(contour, "ContourImageSequence", []):
                out.add(str(image.ReferencedSOPInstanceUID))
    return out


def _source_values(image: Image, key: str) -> List[str]:
    source = next((entry for entry in image.meta.source if entry.format == "dicom"), None)
    if source is None or "shared" not in source.header:
        return []
    header = source.header
    if key in header["shared"]:
        return [str(header["shared"][key])]
    return [str(fields[key]) for fields in header.get("per_file", []) if key in fields]


def _is_image(image: Image) -> bool:
    if image.meta.values.kind != "measurement":
        return False
    return bool(_source_values(image, "SeriesInstanceUID")) and not any(
        value == RTSTRUCT_STORAGE for value in _source_values(image, "SOPClassUID"))


def _choose_reference(ds, images: List[Image]) -> Image:
    series = set(_referenced_series(ds))
    candidates = [image for image in images if _is_image(image)]
    if series:
        candidates = [image for image in candidates if set(_source_values(image, "SeriesInstanceUID")) & series]
    if not candidates:
        raise ValueError(f"the RTSTRUCT references image series {sorted(series)}, which is not among the images read; "
                         "pass the series as reference (medvol.plugins.dicom_rtstruct.read_rtstruct(path, reference, policy))")
    if len(candidates) > 1:
        instances = _referenced_instances(ds)
        covering = [image for image in candidates if instances & set(_source_values(image, "SOPInstanceUID"))]
        if len(covering) == 1:
            return covering[0]
        raise GeometryError(f"the referenced series forms {len(candidates)} images (stacks); the RTSTRUCT's contour "
                            "image references do not single one out")
    return candidates[0]


def _labels(ds, numbers: List[int]) -> Dict[int, Dict[str, Any]]:
    labels: Dict[int, Dict[str, Any]] = {}
    for item in getattr(ds, "StructureSetROISequence", []):
        labels[int(item.ROINumber)] = {"name": str(getattr(item, "ROIName", "") or "")}
    for item in getattr(ds, "ROIContourSequence", []):
        number = int(item.ReferencedROINumber)
        if number in labels and "ROIDisplayColor" in item:
            labels[number]["color"] = [int(v) for v in item.ROIDisplayColor]
    for item in getattr(ds, "RTROIObservationsSequence", []):
        number = int(getattr(item, "ReferencedROINumber", -1))
        if number in labels and "RTROIIdentificationCodeSequence" in item and len(item.RTROIIdentificationCodeSequence):
            code = item.RTROIIdentificationCodeSequence[0]
            value = getattr(code, "CodeValue", None) or getattr(code, "LongCodeValue", None) or getattr(code, "URNCodeValue", None)
            labels[number]["code"] = {"scheme": str(getattr(code, "CodingSchemeDesignator", "") or ""), "value": str(value or "")}
            if getattr(code, "CodeMeaning", None):
                labels[number]["code"]["meaning"] = str(code.CodeMeaning)
    return {number: labels[number] for number in numbers}


def rasterise_onto(ds, images: List[Image], policy: Policy, report: Report) -> Image:
    """The label map of RTSTRUCT ``ds`` on the grid of its referenced image among ``images``."""
    from medvol.plugins.dicom import _file_dict, split_shared

    reference = _choose_reference(ds, images)
    if [axis.type for axis in reference.meta.axes] != ["space"] * 3:
        raise NotImplementedError("RTSTRUCT rasterisation needs a 3-D reference image (no time or other axes)")
    space = reference.space()
    frame_of_reference = dict(space.reference or {}).get("id")
    for item in getattr(ds, "StructureSetROISequence", []):
        uid = str(getattr(item, "ReferencedFrameOfReferenceUID", "") or "")
        if uid and frame_of_reference and uid != frame_of_reference:
            raise GeometryError(f"ROI {int(item.ROINumber)} ({getattr(item, 'ROIName', '')}) is in Frame of Reference "
                                f"{uid}, the reference image in {frame_of_reference}; its coordinates do not apply "
                                "(KB ch02 D7)")
    lps = lps_matrix(space)
    array_axes = list(space.from_array.array_axes)
    names = [axis.name for axis in reference.meta.axes]
    order = [array_axes.index(name) for name in names]          # matrix column of each array axis
    matrix = np.eye(4)
    matrix[:3, :3] = lps[:3, :3][:, order]
    matrix[:3, 3] = lps[:3, 3]
    inverse = np.linalg.inv(matrix)
    shape = reference.array.shape
    nz, ny, nx = shape
    column, row = matrix[:3, 2], matrix[:3, 1]
    normal = np.cross(row, column)
    normal /= np.linalg.norm(normal)
    plane_distance = abs(float(matrix[:3, 0] @ normal))       # mm along the normal per slice index

    numbers = sorted(int(item.ROINumber) for item in getattr(ds, "StructureSetROISequence", []))
    contours_of = {int(item.ReferencedROINumber): item for item in getattr(ds, "ROIContourSequence", [])}
    labels = _labels(ds, numbers)
    masks: Dict[int, np.ndarray] = {}
    for number in numbers:
        name = labels[number]["name"]
        per_slice: Dict[int, List[np.ndarray]] = defaultdict(list)
        skipped: Dict[str, int] = defaultdict(int)
        off_plane, off_grid, beyond, worst = 0, 0, 0, 0.0
        for contour in getattr(contours_of.get(number), "ContourSequence", []) or []:
            kind = str(getattr(contour, "ContourGeometricType", "") or "")
            if kind not in CLOSED_TYPES:
                skipped[kind or "unknown"] += 1
                continue
            points = np.asarray([float(v) for v in contour.ContourData], dtype=np.float64).reshape(-1, 3)
            if len(points) < 3:
                skipped["closed contour with fewer than 3 points"] += 1
                continue
            index = (inverse @ np.vstack([points.T, np.ones(len(points))]))[:3].T     # (z, y, x)
            k = int(np.round(np.mean(index[:, 0])))
            distance = float(np.max(np.abs(index[:, 0] - k))) * plane_distance
            if distance > CONTOUR_PLANE_TOLERANCE_MM:
                off_plane += 1
                worst = max(worst, distance)
                continue
            if not 0 <= k < nz:
                beyond += 1
                continue
            if (index[:, 1].min() < -0.5 or index[:, 1].max() > ny - 0.5 or index[:, 2].min() < -0.5
                    or index[:, 2].max() > nx - 0.5):
                off_grid += 1
            per_slice[k].append(index[:, 1:])
        mask = np.zeros(shape, dtype=bool)
        for k, polygons in per_slice.items():
            mask[k] = even_odd(polygons, ny, nx)
        masks[number] = mask
        for kind, count in skipped.items():
            report.warn(f"ROI {number} ({name}): {count} {kind} contour(s) are not areas and were not rasterised")
        if off_plane:
            report.drop(f"ROI {number} ({name}) contours", f"{off_plane} contour(s) lie up to {worst:.4g} mm off every slice "
                        f"plane of the reference grid (tolerance {CONTOUR_PLANE_TOLERANCE_MM} mm)", policy.on_loss)
        if beyond:
            report.drop(f"ROI {number} ({name}) contours", f"{beyond} contour(s) lie beyond the first or last slice of "
                        "the reference grid", policy.on_loss)
        if off_grid:
            report.drop(f"ROI {number} ({name}) contours", f"{off_grid} contour(s) extend beyond the in-plane field of "
                        "view of the reference grid; the part outside is not represented", policy.on_loss)

    report.choose("rtstruct_rasterisation", RULE)
    stacked = np.stack([masks[n] for n in numbers]) if numbers else np.zeros((0,) + shape, dtype=bool)
    overlap = bool(np.any(np.count_nonzero(stacked, axis=0) > 1)) if numbers else False
    axes: List[Dict[str, Any]] = [{"name": name, "type": "space"} for name in names]
    if overlap or any(n <= 0 for n in numbers):
        reason = "ROIs overlap" if overlap else "ROI numbers are not all positive"
        report.warn(f"{reason}: stored as one channel per ROI (label_layout 'channel')")
        array = stacked.astype(np.uint8)
        axes.insert(0, {"name": "c", "type": "channel"})
        values = {"kind": "label", "label_layout": "channel",
                  "labels": {str(c): labels[number] for c, number in enumerate(numbers)}}
    else:
        top = max(numbers, default=0)
        dtype = np.uint8 if top <= 0xFF else np.uint16 if top <= 0xFFFF else np.uint32
        array = np.zeros(shape, dtype=dtype)
        for number in numbers:
            array[masks[number]] = number
        values = {"kind": "label", "label_layout": "index", "labels": {str(n): labels[n] for n in numbers}}
        if not numbers:
            del values["labels"]
    report.choose("rtstruct_rois", {str(n): int(np.count_nonzero(masks[n])) for n in numbers})

    header_shared, per_file = split_shared([_file_dict(ds)])
    name = str(getattr(ds, "filename_relative", "") or "")
    header = {"shared": header_shared, "per_file": per_file, "file_names": [name]}
    space_dict = space.to_dict()
    meta = Meta(axes=axes, spaces=[space_dict], values=values, shape=array.shape)
    source = {
        "format": "dicom",
        "files": [name] if name else [],
        "reader": {"name": "medvol"},
        "fingerprint": fingerprint.compute(array, meta),
        "header": header,
    }
    return Image(array, Meta(axes=axes, spaces=[space_dict], values=values, source=[source], shape=array.shape))
