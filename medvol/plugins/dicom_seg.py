"""DICOM reading phase 3 and SEG writing: DICOM SEG <-> label map (spec 7.5 and 7.6).

Reading (pydicom, MedVol's own geometry):

- Frames are placed by their Plane Position on the slice normal, never by frame order (KB ch02
  7.2, pitfalls D2/D3): writers omit empty frames, group frames by segment, and the order is up to
  the writer. The grid spans the stored frames; slices without any stored frame (omitted because
  empty) are filled with 0. Grid spacing is the Spacing Between Slices of the Pixel Measures when
  every gap is a whole multiple of it, else the smallest gap; positions off that regular grid
  follow ``dicom_irregular_spacing``, tilt follows ``dicom_tilt``, several geometric stacks follow
  ``dicom_stacks``.
- ``values.labels`` holds every segment of the Segment Sequence: ``name`` (Segment Label),
  ``color`` (Recommended Display CIELab Value converted to RGB) and ``code`` (Segmented Property
  Type Code Sequence as ``{scheme, value, meaning}``).
- BINARY SEG with non-overlapping segments and LABELMAP SEG: ``kind: label``, ``label_layout:
  index``, array ``[z, y, x]`` holding segment numbers, labels keyed by segment number.
  BINARY SEG with overlapping segments: ``label_layout: channel``, array ``[c, z, y, x]`` of 0/1,
  channel ``c`` = the ``c``-th segment in ascending segment number, labels keyed by channel index.
  FRACTIONAL SEG: ``kind: probability``, ``label_layout: channel``, stored values kept raw (uint8)
  with ``scale = 1 / MaximumFractionalValue``.
- ``source.header``: ``shared`` / ``per_file`` / ``file_names`` and ``frames`` (one entry per
  stored frame placed in this image, in file order: ``frame_index``, ``segment``, ``slice`` and the
  frame's ``functional_groups``).

Writing (highdicom): a SEG is created from a label map and the referenced image series:

- The reference is given as ``medvol.write(..., reference=...)`` (a series directory, a list of files or of
  pydicom datasets), or with ``write_seg(image, path, policy, reference=...)``.
- The label map's grid must coincide with the reference: every label-map slice must be an image of
  the series (position, orientation, spacing and size within ``GEOMETRY_TOLERANCE`` mm at the
  slice corners) and the Frame of Reference must agree; otherwise the write is refused. Source
  images are passed to highdicom in label-map slice order (KB pitfall D4), and the frame positions
  of the result are checked against the label map before saving.
- Segment number, label, property type code and colour come from the core (``values.labels``);
  everything else about a segment (category, algorithm, tracking ids, description) is kept from the
  source SEG when there is one, otherwise defaults are used and reported.
- A SEG source keeps its top-level descriptive elements and, while the fingerprint matches, its UIDs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mlarray import Meta, fingerprint

from medvol.geometry import GeometryError, homogeneous, lps_matrix, patient_space
from medvol.image import Image
from medvol.plugins.dicom import SPACING_TOLERANCE, TILT_TOLERANCE, _files, _pydicom, _sorted_stack
from medvol.plugins.dicom_enhanced import (Frame, cluster_positions, frame_header, frame_pixels, frames_of, group_stacks,
                                           normal_of)
from medvol.policy import Policy
from medvol.report import Report

SEGMENTATION_STORAGE = "1.2.840.10008.5.1.4.1.1.66.4"
LABEL_MAP_SEGMENTATION_STORAGE = "1.2.840.10008.5.1.4.1.1.66.7"
SEG_SOP_CLASSES = frozenset({SEGMENTATION_STORAGE, LABEL_MAP_SEGMENTATION_STORAGE})

GEOMETRY_TOLERANCE = 1e-3      # mm: label-map grid vs referenced images
NOT_RESAMPLED = ("{policy}='resample' does not apply to DICOM SEG: label maps are not interpolated; read the SEG "
                 "with the default policies and resample the label map explicitly if needed")

#: Segment fields owned by the core (values.labels); the rest of a segment item is verbatim.
CORE_SEGMENT_FIELDS = frozenset({"SegmentNumber", "SegmentLabel", "SegmentedPropertyTypeCodeSequence",
                                 "RecommendedDisplayCIELabValue"})
DEFAULT_CATEGORY = ("91723000", "SCT", "Anatomical Structure")
#: Top-level elements of a source SEG that are rebuilt by highdicom, never copied from the source.
STRUCTURAL = frozenset({
    "file_meta", "SOPClassUID", "SOPInstanceUID", "SeriesInstanceUID", "SeriesNumber", "InstanceNumber",
    "SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence", "DimensionIndexSequence",
    "DimensionOrganizationSequence", "DimensionOrganizationType", "ReferencedSeriesSequence", "SourceImageSequence",
    "NumberOfFrames", "Rows", "Columns", "SamplesPerPixel", "PhotometricInterpretation", "BitsAllocated", "BitsStored",
    "HighBit", "PixelRepresentation", "PixelPaddingValue", "SegmentSequence", "SegmentationType",
    "SegmentationFractionalType", "MaximumFractionalValue", "SegmentsOverlap", "LossyImageCompression",
    "FrameOfReferenceUID", "PositionReferenceIndicator", "ContributingEquipmentSequence", "PresentationLUTShape",
    "RedPaletteColorLookupTableDescriptor", "GreenPaletteColorLookupTableDescriptor",
    "BluePaletteColorLookupTableDescriptor", "RedPaletteColorLookupTableData", "GreenPaletteColorLookupTableData",
    "BluePaletteColorLookupTableData", "PaletteColorLookupTableUID", "ICCProfile",
})
#: Descriptive elements highdicom fills with new values; a source SEG's values take precedence.
SOURCE_PRECEDENCE = ("ContentDate", "ContentTime", "InstanceCreationDate", "InstanceCreationTime", "SeriesDate",
                     "SeriesTime", "ContentLabel", "ContentDescription", "ContentCreatorName", "Manufacturer",
                     "ManufacturerModelName", "SoftwareVersions", "DeviceSerialNumber")


def _highdicom():
    import highdicom

    return highdicom


def is_seg_dataset(ds) -> bool:
    return str(getattr(ds, "SOPClassUID", "")) in SEG_SOP_CLASSES


def _dicom_source(image: Image):
    return next((entry for entry in image.meta.source if entry.format == "dicom"), None)


def _source_is_seg(source) -> bool:
    return source is not None and str(source.header.get("shared", {}).get("SOPClassUID", "")) in SEG_SOP_CLASSES


def is_segmentation(image: Image) -> bool:
    """True when ``image`` is written as a DICOM SEG: a label/probability map or a SEG source."""
    return image.meta.values.kind in ("label", "probability") or _source_is_seg(_dicom_source(image))


# ------------------------------------------------------------------------------------------------
# reading
# ------------------------------------------------------------------------------------------------

def _rgb(lab: Sequence[int], report: Report) -> List[int]:
    rgb = [int(v) for v in _highdicom().color.CIELabColor.from_dicom_value(tuple(int(v) for v in lab)).to_rgb()]
    clipped = [min(max(v, 0), 255) for v in rgb]
    if clipped != rgb:
        report.warn(f"CIELab display colour {list(lab)} lies outside RGB; clipped to {clipped}")
    return clipped


def segment_labels(ds, report: Report) -> Dict[int, Dict[str, Any]]:
    """Segment number -> ``{name, color, code}`` from the Segment Sequence.

    A label-map SEG may describe its background as segment 0.
    """
    labels: Dict[int, Dict[str, Any]] = {}
    for item in getattr(ds, "SegmentSequence", []):
        entry: Dict[str, Any] = {"name": str(getattr(item, "SegmentLabel", "") or "")}
        if "RecommendedDisplayCIELabValue" in item:
            entry["color"] = _rgb(item.RecommendedDisplayCIELabValue, report)
        if "SegmentedPropertyTypeCodeSequence" in item and len(item.SegmentedPropertyTypeCodeSequence):
            code = item.SegmentedPropertyTypeCodeSequence[0]
            value = getattr(code, "CodeValue", None) or getattr(code, "LongCodeValue", None) or getattr(code, "URNCodeValue", None)
            entry["code"] = {"scheme": str(getattr(code, "CodingSchemeDesignator", "") or ""), "value": str(value or "")}
            if getattr(code, "CodeMeaning", None):
                entry["code"]["meaning"] = str(code.CodeMeaning)
        labels[int(item.SegmentNumber)] = entry
    numbers = [number for number in sorted(labels)
               if number != 0 or str(getattr(ds, "SegmentationType", "")).upper() != "LABELMAP"]
    if numbers != list(range(1, len(numbers) + 1)):
        report.warn(f"segment numbers {numbers} do not run 1..{len(numbers)} as DICOM requires")
    return labels


def _grid(clusters: List[List[Frame]], policy: Policy, report: Report) -> List[Tuple[List[int], List[List[Frame]], np.ndarray]]:
    """[(slice index per cluster, clusters, step vector)] of the regular grid(s) the frames lie on.

    Gaps between stored positions (omitted empty frames) are allowed when they are whole multiples
    of the grid spacing.
    """
    representatives = [cluster[0] for cluster in clusters]
    normal = normal_of(representatives[0])
    if len(clusters) == 1:
        first = representatives[0]
        thickness = first.SpacingBetweenSlices or first.SliceThickness or 1.0
        report.warn(f"single slice: slice spacing taken from "
                    f"{'SpacingBetweenSlices' if first.SpacingBetweenSlices else 'SliceThickness'} ({thickness} mm)")
        return [([0], clusters, normal * thickness)]
    positions = np.array([frame.ImagePositionPatient for frame in representatives])
    gaps = np.diff(positions @ normal)
    counts = None
    for unit in (representatives[0].SpacingBetweenSlices, float(np.min(gaps))):
        if not unit:
            continue
        multiples = np.round(gaps / unit)
        if np.all(multiples >= 1) and np.all(np.abs(gaps - multiples * unit) <= SPACING_TOLERANCE * unit * multiples):
            counts = multiples.astype(int)
            break
    if counts is None:
        if policy.dicom_irregular_spacing == "error":
            raise GeometryError(f"irregular slice spacing {np.round(gaps, 4).tolist()} mm "
                                "(policy dicom_irregular_spacing='error')")
        if policy.dicom_irregular_spacing == "resample":
            raise NotImplementedError(NOT_RESAMPLED.format(policy="dicom_irregular_spacing"))
        cluster_of = {id(cluster[0]): cluster for cluster in clusters}
        runs = _sorted_stack(representatives, policy, report)
        if any(not isinstance(step, np.ndarray) for _, step in runs):
            raise NotImplementedError(NOT_RESAMPLED.format(policy="dicom_tilt"))
        return [(list(range(len(run))), [cluster_of[id(rep)] for rep in run], step) for run, step in runs]
    indices = np.concatenate([[0], np.cumsum(counts)])
    n = int(indices[-1]) + 1
    step = (positions[-1] - positions[0]) / (n - 1)
    drift = np.linalg.norm(positions - (positions[0] + np.outer(indices, step)), axis=1)
    if np.any(drift > TILT_TOLERANCE * np.linalg.norm(step) + 1e-6):
        raise GeometryError("SEG frame positions do not lie on a straight line; cannot form one grid")
    in_plane = step - (step @ normal) * normal
    if np.linalg.norm(in_plane) > TILT_TOLERANCE * abs(step @ normal):
        angle = float(np.degrees(np.arctan2(np.linalg.norm(in_plane), abs(step @ normal))))
        if policy.dicom_tilt == "error":
            raise GeometryError(f"gantry tilt of {angle:.3f} degrees (policy dicom_tilt='error')")
        if policy.dicom_tilt == "resample":
            raise NotImplementedError(NOT_RESAMPLED.format(policy="dicom_tilt"))
        report.choose("dicom_tilt", "exact")
        report.warn(f"gantry tilt of {angle:.3f} degrees: stored as an exact sheared grid")
    if n > len(clusters):
        report.warn(f"the SEG stores frames at {len(clusters)} of the {n} slice positions of its grid; "
                    "the other slices are empty")
    return [([int(i) for i in indices], clusters, step)]


def read_seg(ds, policy: Policy, report: Report) -> List[Image]:
    """The label map(s) of one SEG file."""
    seg_type = str(getattr(ds, "SegmentationType", "BINARY")).upper()
    labels = segment_labels(ds, report)
    frames = frames_of(ds, 0)
    if seg_type != "LABELMAP":
        for frame in frames:
            if frame.segment is None:
                if len(labels) != 1:
                    raise GeometryError(f"frame {frame.index} names no segment (Segment Identification Sequence)")
                frame.segment = next(iter(labels))
            if frame.segment not in labels:
                raise GeometryError(f"frame {frame.index} references segment {frame.segment}, which the Segment Sequence lacks")
    pixels = frame_pixels(ds)
    report.choose("dicom_seg", seg_type.lower())
    images = []
    for stack in group_stacks(frames, policy, report, use_stack_id=False):
        for indices, clusters, step in _grid(cluster_positions(stack), policy, report):
            images.append(_image(ds, labels, seg_type, indices, clusters, step, pixels, policy, report))
    return images


def _image(ds, labels: Dict[int, Dict[str, Any]], seg_type: str, indices: List[int], clusters: List[List[Frame]],
           step: np.ndarray, pixels: np.ndarray, policy: Policy, report: Report) -> Image:
    first = clusters[0][0]
    rows, columns = first.Rows, first.Columns
    n = indices[-1] + 1
    numbers = sorted(labels)
    placed: List[Tuple[Frame, int]] = []
    if seg_type == "LABELMAP":
        data = np.zeros((n, rows, columns), dtype=pixels.dtype)
        for z, cluster in zip(indices, clusters):
            if len(cluster) > 1:
                raise GeometryError(f"a label-map SEG holds {len(cluster)} frames at one slice position")
            data[z] = pixels[cluster[0].index]
            placed.append((cluster[0], z))
    else:
        data = np.zeros((len(numbers), n, rows, columns), dtype=pixels.dtype)
        filled = np.zeros((len(numbers), n), dtype=bool)
        for z, cluster in zip(indices, clusters):
            for frame in cluster:
                c = numbers.index(frame.segment)
                if filled[c, z]:
                    raise GeometryError(f"segment {frame.segment} has two frames at one slice position")
                filled[c, z] = True
                data[c, z] = pixels[frame.index]
                placed.append((frame, z))

    values: Dict[str, Any]
    axes: List[Dict[str, Any]] = [{"name": "z", "type": "space"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    if seg_type == "FRACTIONAL":
        maximum = int(getattr(ds, "MaximumFractionalValue", 255))
        values = {"kind": "probability", "scale": 1.0 / maximum, "label_layout": "channel",
                  "labels": {str(c): labels[number] for c, number in enumerate(numbers)}}
        array = data
    elif seg_type == "LABELMAP" or not np.any(np.count_nonzero(data, axis=0) > 1):
        if seg_type != "LABELMAP":
            dtype = np.uint8 if (numbers[-1] if numbers else 0) <= 255 else np.uint16
            array = np.zeros(data.shape[1:], dtype=dtype)
            for c, number in enumerate(numbers):
                array[data[c] > 0] = number
        else:
            array = data
        values = {"kind": "label", "label_layout": "index", "labels": {str(number): labels[number] for number in numbers}}
    else:
        report.warn("segments overlap: stored as one channel per segment (label_layout 'channel')")
        values = {"kind": "label", "label_layout": "channel",
                  "labels": {str(c): labels[number] for c, number in enumerate(numbers)}}
        array = data
    if array.ndim == 4:
        axes.insert(0, {"name": "c", "type": "channel"})
    if seg_type != "FRACTIONAL" and values["labels"] == {}:
        del values["labels"]

    iop = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    row_spacing, column_spacing = first.PixelSpacing
    origin = clusters[0][0].ImagePositionPatient
    matrix = homogeneous(np.column_stack([step, iop[3:] * row_spacing, iop[:3] * column_spacing]), origin)
    uid = getattr(ds, "FrameOfReferenceUID", None)
    reference = {"scheme": "dicom_frame_of_reference", "id": str(uid)} if uid else None
    space = patient_space(matrix, ["z", "y", "x"], policy.world_frame, reference=reference)

    placed.sort(key=lambda item: item[0].index)
    header = frame_header([ds], [frame for frame, _ in placed])
    for entry, (frame, z) in zip(header["frames"], placed):
        entry["segment"] = frame.segment
        entry["slice"] = z
    order = [3, 2, 1, 0] if array.ndim == 4 else [2, 1, 0]
    meta = Meta(axes=axes, spaces=[space], values=values, shape=array.shape)
    source = {
        "format": "dicom",
        "files": header["file_names"],
        "reader": {"name": "medvol"},
        "index_map": {"axis_order": order, "flipped": [False] * len(order), "offset": [0] * len(order)},
        "fingerprint": fingerprint.compute(array, meta),
        "header": header,
    }
    return Image(array, Meta(axes=axes, spaces=[space], values=values, source=[source], shape=array.shape))


# ------------------------------------------------------------------------------------------------
# writing
# ------------------------------------------------------------------------------------------------

def _reference_datasets(reference) -> List:
    """The image datasets of the referenced series (directory, file(s) or pydicom datasets)."""
    pydicom = _pydicom()
    if isinstance(reference, (str, os.PathLike)):
        reference = _files(reference)
    datasets = [item if isinstance(item, pydicom.dataset.Dataset) else pydicom.dcmread(str(item)) for item in reference]
    datasets = [ds for ds in datasets if not is_seg_dataset(ds)]
    if not datasets:
        raise ValueError("the referenced series holds no DICOM images")
    for ds in datasets:
        if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1 or "PerFrameFunctionalGroupsSequence" in ds:
            raise NotImplementedError("SEG writing references single-frame image series only; "
                                      "an enhanced multi-frame reference is not supported yet")
    return [ds for ds in datasets if "ImagePositionPatient" in ds]


def _match_reference(datasets: List, origin: np.ndarray, step: np.ndarray, row: np.ndarray, column: np.ndarray,
                     shape: Tuple[int, int, int], frame_of_reference: Optional[str]) -> Tuple[List, float]:
    """The referenced image of every label-map slice (in slice order) and the largest deviation in mm.

    Each slice must coincide with one image: same size, and the corner voxel centres within
    ``GEOMETRY_TOLERANCE`` mm. Raises GeometryError otherwise.
    """
    nz, ny, nx = shape
    positions = np.array([[float(v) for v in ds.ImagePositionPatient] for ds in datasets])
    matched, worst = [], 0.0
    for k in range(nz):
        expected = origin + k * step
        distances = np.linalg.norm(positions - expected, axis=1)
        n = int(np.argmin(distances))
        if distances[n] > GEOMETRY_TOLERANCE:
            raise GeometryError(f"label-map slice {k} at {np.round(expected, 4).tolist()} mm is no image of the referenced "
                                f"series (nearest image {distances[n]:.4f} mm away); the SEG must lie on the reference grid")
        ds = datasets[n]
        if (int(ds.Rows), int(ds.Columns)) != (ny, nx):
            raise GeometryError(f"label-map slices are {ny} x {nx}, the referenced images {ds.Rows} x {ds.Columns}")
        iop = np.array([float(v) for v in ds.ImageOrientationPatient])
        row_spacing, column_spacing = (float(v) for v in ds.PixelSpacing)
        reference_row, reference_column = iop[3:] * row_spacing, iop[:3] * column_spacing
        deviation = max(np.linalg.norm((ny - 1) * (row - reference_row)), np.linalg.norm((nx - 1) * (column - reference_column)),
                        float(distances[n]))
        if deviation > GEOMETRY_TOLERANCE:
            raise GeometryError(f"the in-plane grid of label-map slice {k} differs from the referenced image by "
                                f"{deviation:.4f} mm at its corners (orientation, spacing or flipped axes); "
                                "the SEG must lie on the reference grid")
        uid = str(getattr(ds, "FrameOfReferenceUID", "") or "")
        if frame_of_reference and uid and uid != frame_of_reference:
            raise GeometryError(f"the label map is in Frame of Reference {frame_of_reference}, the referenced series in {uid}")
        if any(ds is other for other in matched):
            raise GeometryError(f"two label-map slices fall on the same referenced image ({k})")
        matched.append(ds)
        worst = max(worst, deviation)
    return matched, worst


def _label_map_grid(image: Image):
    """(array as [z, y, x(, c)], origin, slice step, row vector, column vector) in LPS from the core."""
    space = image.space()
    if space is None:
        raise GeometryError("a SEG needs a label map placed in patient space")
    names = [axis.name for axis in image.meta.axes]
    space_axes = list(space.from_array.array_axes)
    spatial = [name for name in names if name in space_axes]
    if len(spatial) != 3:
        raise GeometryError(f"a SEG needs 3 space axes, the label map maps {spatial}")
    others = [axis for axis in image.meta.axes if axis.name not in space_axes]
    layout = image.meta.values.label_layout or "index"
    if layout == "channel":
        if len(others) != 1 or others[0].type != "channel":
            raise GeometryError("label_layout 'channel' needs exactly one channel axis besides the space axes")
    elif others:
        raise GeometryError(f"an index label map has only space axes; it also has {[a.name for a in others]}")
    order = [names.index(name) for name in spatial] + [names.index(axis.name) for axis in others]
    array = np.transpose(image.array, order)
    lps = lps_matrix(space)
    step, row, column = (lps[:3, space_axes.index(name)] for name in spatial)
    reference = space.reference
    frame_of_reference = None
    if reference is not None and reference.get("scheme") == "dicom_frame_of_reference":
        frame_of_reference = str(reference.get("id"))
    return array, lps[:3, -1], step, row, column, frame_of_reference


def _code(entry: Optional[Dict[str, Any]]):
    from pydicom.sr.coding import Code

    code = (entry or {}).get("code")
    if not code:
        return None
    return Code(str(code["value"]), str(code["scheme"]), str(code.get("meaning") or code["value"]))


def write_seg(image: Image, path: str, policy: Policy, reference=None) -> Report:
    """A DICOM SEG of the label map ``image`` on the referenced series (spec 7.6).

    ``reference`` (or ``medvol.write(..., reference=...)``): the referenced image series as a directory, a list
    of files or of pydicom datasets. ``path`` is the output file (``*.dcm``) or a directory.
    """
    from medvol import __version__
    from medvol.plugins.dicom_write import dict_to_dataset

    hd = _highdicom()
    report = Report("write", "dicom", str(path))
    if reference is None:
        raise ValueError("writing a DICOM SEG needs the referenced image series: pass reference= with the "
                         "series directory, a list of its files or pydicom datasets (or call "
                         "medvol.plugins.dicom_seg.write_seg(image, path, policy, reference=...))")
    values = image.meta.values
    layout = values.label_layout or "index"
    if values.kind not in ("label", "probability"):
        raise ValueError(f"a DICOM SEG holds label or probability maps, not values of kind {values.kind!r}")
    if values.kind == "probability" and layout != "channel":
        raise ValueError("a probability map needs label_layout 'channel' (one channel per segment)")

    array, origin, step, row, column, frame_of_reference = _label_map_grid(image)
    sources, deviation = _match_reference(_reference_datasets(reference), origin, step, row, column,
                                          array.shape[:3], frame_of_reference)
    sources = [_with_type2(ds, report) for ds in sources]
    report.choose("seg_reference", f"{len(sources)} images of series {sources[0].SeriesInstanceUID} "
                                   f"(largest deviation {deviation:.2e} mm)")

    source = _dicom_source(image)
    seg_source = source if _source_is_seg(source) else None
    shared = dict(seg_source.header["shared"]) if seg_source is not None else {}
    source_segments = {int(item["SegmentNumber"]): item for item in shared.get("SegmentSequence", []) or []}
    labels = {int(key): dict(entry) for key, entry in (values.labels or {}).items()}

    # segments and the pixel array highdicom expects: (frames, rows, columns[, segments])
    if layout == "index":
        if array.dtype.kind not in "ui":
            raise ValueError(f"an index label map needs integer values, not {array.dtype}")
        present = [int(v) for v in np.unique(array) if v != 0]
        if any(v < 0 for v in present) or any(key < 0 for key in labels):
            raise ValueError("negative label values cannot be stored as SEG segment numbers")
        numbers = sorted(set(present) | {key for key in labels if key > 0})
        renumber = {number: n + 1 for n, number in enumerate(numbers)}
        if any(old != new for old, new in renumber.items()):
            report.warn(f"label values {numbers} renumbered to segment numbers 1..{len(numbers)} as DICOM requires")
            report.choose("segment_numbers", "renumbered")
        lookup = np.zeros(max(numbers, default=0) + 1, dtype=np.uint16)
        for old, new in renumber.items():
            lookup[old] = new
        pixel_array = lookup[array].astype(np.uint8 if len(numbers) <= 255 else np.uint16)
        segments = [(renumber[number], number, labels.get(number)) for number in numbers]
        seg_type = "LABELMAP" if shared.get("SegmentationType") == "LABELMAP" else "BINARY"
    else:
        segments = [(c + 1, c + 1, labels.get(c)) for c in range(array.shape[3])]
        if values.kind == "probability":
            seg_type = "FRACTIONAL"
            if array.dtype.kind == "f":
                maximum, pixel_array = int(shared.get("MaximumFractionalValue", 255)), array
            else:
                if values.scale is None:
                    raise ValueError("an integer probability map needs values.scale (1 / maximum fractional value)")
                maximum = int(round(1.0 / values.scale))
                pixel_array = array.astype(np.float64) / maximum
            if pixel_array.min() < 0 or pixel_array.max() > 1:
                raise ValueError("fractional SEG values must lie in 0..1 after scaling")
        else:
            seg_type = "BINARY"
            if not np.all((array == 0) | (array == 1)):
                raise ValueError("a channel label map written as a binary SEG must hold only 0 and 1")
            pixel_array = array.astype(np.uint8)
    if not segments:
        raise ValueError("the label map has no segments (no non-zero values and no labels)")

    descriptions = []
    for number, original, entry in segments:
        if entry is None:
            report.warn(f"segment {number} has no entry in values.labels: named 'segment {number}'")
            entry = {"name": f"segment {number}"}
        code = _code(entry)
        if code is None:
            report.warn(f"segment {number} ({entry.get('name')}) has no code: property type "
                        f"{DEFAULT_CATEGORY[2]} ({DEFAULT_CATEGORY[1]} {DEFAULT_CATEGORY[0]}) used")
            code = _code({"code": {"value": DEFAULT_CATEGORY[0], "scheme": DEFAULT_CATEGORY[1], "meaning": DEFAULT_CATEGORY[2]}})
        if original not in source_segments:
            report.warn(f"segment {number}: category {DEFAULT_CATEGORY[2]} and algorithm type MANUAL assumed (no source SEG segment)")
        color = entry.get("color")
        descriptions.append(hd.seg.SegmentDescription(
            segment_number=number, segment_label=str(entry.get("name") or f"segment {number}"),
            segmented_property_category=_code({"code": {"value": DEFAULT_CATEGORY[0], "scheme": DEFAULT_CATEGORY[1],
                                                        "meaning": DEFAULT_CATEGORY[2]}}),
            segmented_property_type=code, algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL,
            display_color=hd.color.CIELabColor.from_rgb(*[int(v) for v in color[:3]]) if color else None))

    fp = seg_source.fingerprint if seg_source is not None else None
    unchanged = bool(fp) and fp.get("geometry") == fingerprint.geometry_hash(image.meta) \
        and list(fp.get("shape", [])) == list(image.array.shape) and fp.get("data") == fingerprint.data_hash(image.array)
    if seg_source is not None and not unchanged:
        report.warn("the label map changed since the SEG was read: new SOP Instance UID; the Series Instance UID is kept")
    keep_series = seg_source is not None and "SeriesInstanceUID" in shared
    kwargs = dict(
        series_instance_uid=shared["SeriesInstanceUID"] if keep_series else hd.UID(),
        series_number=int(shared.get("SeriesNumber", 1) or 1),
        sop_instance_uid=shared["SOPInstanceUID"] if unchanged and "SOPInstanceUID" in shared else hd.UID(),
        instance_number=int(shared.get("InstanceNumber", 1) or 1),
        manufacturer=str(shared.get("Manufacturer") or "MedVol"),
        manufacturer_model_name=str(shared.get("ManufacturerModelName") or "medvol"),
        software_versions=shared.get("SoftwareVersions") or f"medvol {__version__}",
        device_serial_number=str(shared.get("DeviceSerialNumber") or "0"),
        content_label=str(shared.get("ContentLabel") or "SEGMENTATION"),
    )
    if isinstance(kwargs["software_versions"], (list, tuple)):
        kwargs["software_versions"] = tuple(str(v) for v in kwargs["software_versions"])
    if seg_type == "FRACTIONAL":
        kwargs["max_fractional_value"] = maximum
        kwargs["fractional_type"] = str(shared.get("SegmentationFractionalType") or "PROBABILITY")
    seg = hd.seg.Segmentation(source_images=sources, pixel_array=pixel_array, segmentation_type=seg_type,
                              segment_descriptions=descriptions, omit_empty_frames=False, **kwargs)
    report.choose("seg_type", seg_type)

    # verbatim segment fields and top-level elements of a source SEG
    by_number = {number: (original, entry) for number, original, entry in segments}
    for item in seg.SegmentSequence:     # a label-map SEG also describes its background (segment 0)
        original, entry = by_number.get(int(item.SegmentNumber), (int(item.SegmentNumber), None))
        source_item = source_segments.get(original, {})
        verbatim = {key: value for key, value in source_item.items() if key not in CORE_SEGMENT_FIELDS}
        for element in dict_to_dataset(verbatim, report):
            item[element.tag] = element
        lab = source_item.get("RecommendedDisplayCIELabValue")
        if "RecommendedDisplayCIELabValue" in item and lab is not None \
                and _rgb(lab, Report("read", "dicom")) == [int(v) for v in (entry or {}).get("color", [])[:3]]:
            item.RecommendedDisplayCIELabValue = [int(v) for v in lab]      # the original CIELab, not a re-conversion
    if seg_source is not None:
        present = {element.keyword if element.keyword and not element.tag.is_private
                   else f"({element.tag.group:04X},{element.tag.element:04X})" for element in seg}
        rest = {key: value for key, value in shared.items()
                if key not in STRUCTURAL and (key not in present or key in SOURCE_PRECEDENCE)}
        for element in dict_to_dataset(rest, report):
            seg[element.tag] = element

    _check_frames(seg, sources, origin, step)
    target = Path(path)
    if target.suffix.lower() != ".dcm":
        name = seg_source.header["file_names"][0] if seg_source is not None and seg_source.header.get("file_names") else "seg.dcm"
        target.mkdir(parents=True, exist_ok=True)
        target = target / name
    target.parent.mkdir(parents=True, exist_ok=True)
    seg.save_as(str(target))
    return report


def _with_type2(ds, report: Report):
    """``ds``, or a copy with the empty type 2 patient/study attributes highdicom copies from it."""
    missing = [key for key in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
                               "ReferringPhysicianName", "StudyID", "AccessionNumber") if key not in ds]
    if not missing:
        return ds
    import copy

    ds = copy.deepcopy(ds)
    for key in missing:
        setattr(ds, key, None)
    report.warn(f"the referenced images lack {', '.join(missing)}: written empty (type 2) in the SEG")
    return ds


def _check_frames(seg, sources: List, origin: np.ndarray, step: np.ndarray) -> None:
    """Every SEG frame must sit where its label-map slice is (guards against KB pitfall D4)."""
    slice_of = {str(ds.SOPInstanceUID): k for k, ds in enumerate(sources)}
    for n, item in enumerate(seg.PerFrameFunctionalGroupsSequence):
        position = np.array([float(v) for v in item.PlanePositionSequence[0].ImagePositionPatient])
        uid = str(item.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
        k = slice_of[uid]
        if np.linalg.norm(position - (origin + k * step)) > GEOMETRY_TOLERANCE:
            raise GeometryError(f"SEG frame {n} would be placed {np.linalg.norm(position - (origin + k * step)):.4f} mm "
                                f"away from label-map slice {k}")
