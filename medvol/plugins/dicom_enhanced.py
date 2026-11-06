"""DICOM reading, phase 2: enhanced multi-frame images (spec 7.5), and re-emitting them (spec 7.6).

Enhanced CT/MR/PET (and Legacy Converted Enhanced) objects keep all frames in one file and the
geometry in functional groups, not at the top level (KB ch02 4.1, pitfall D1):

- Every frame's Plane Position, Plane Orientation and Pixel Measures are looked up in its
  Per-frame Functional Groups item first, then in the Shared Functional Groups. Canon puts
  orientation and pixel measures in the shared group, Philips and Siemens XA repeat them per frame
  (KB ``verify_dicom_real/02``).
- Frames are grouped into geometric stacks (in-plane geometry and Stack ID, policy ``dicom_stacks``)
  and placed by the projection of their position on the slice normal, never by frame order or by
  Dimension Index Values, which are ordinals (KB pitfall D2).
- Several frames at one position are told apart by the non-spatial dimensions of the Multi-frame
  Dimension module (Dimension Index Values, or the Temporal Position Index of Frame Content).
  One temporal dimension gives a 4-D ``[t, z, y, x]`` image with a ``time`` axis, one other
  dimension an ``other`` axis named ``u``; when several dimensions vary, the frames are split by
  the non-temporal ones and the parts are handled per ``dicom_stacks`` (as dcm2niix splits
  echoes). All frames of one image lie on one regular grid; spacing and tilt follow the phase 1
  policies (``dicom_irregular_spacing``, ``dicom_tilt``).
- A series may consist of several multi-frame files (Siemens XA: one volume per file); the frames
  of all files are assembled together.

``source.header``: ``shared`` / ``per_file`` / ``file_names`` over the files, as for single-frame
series, with the Shared Functional Groups kept as they are, and ``frames``: one entry per array
frame in array order (C order over the axes before ``y, x``) with ``file`` (index into
``file_names``), ``frame_index`` (0-based index of the frame in that file) and
``functional_groups`` (the frame's Per-frame Functional Groups item). The Per-frame Functional
Groups Sequence itself is not stored a second time.

Writing (``write_enhanced``) re-emits the original file(s): every top-level element and every
functional group comes from ``source``, pixel data from the array, put back at each frame's
original index. When the geometry changed, Plane Position, Plane Orientation and Pixel Measures
are regenerated from the core in the functional groups (shared or per frame) where the source has
them; when ``values.scale``/``offset`` changed, the Pixel Value Transformation is regenerated; and
physical values read with ``dicom_rescale='apply'`` are inverted per frame (``dicom_rescale``).
Frame order, Dimension Index Values and In-Stack Position Numbers are kept verbatim.

Policies ``dicom_rescale='apply'`` (per-frame rescale, ``dicom_rescale.py``) and
``dicom_tilt``/``dicom_irregular_spacing='resample'`` (``dicom_resample.py``) apply to the frames
as to single-frame series; a 4-D image is resampled per time point (or other index) with the same
plan, since every time point has the same slice positions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mlarray import Meta, fingerprint

from medvol.geometry import GeometryError, homogeneous, patient_space
from medvol.image import Image
from medvol.plugins.dicom import _file_dict, _pydicom, _same, _sorted_stack, dataset_to_dict, split_shared
from medvol.policy import Policy
from medvol.report import Report

PER_FRAME_GROUPS = 0x52009230
SHARED_GROUPS = 0x52009229
PIXEL_DATA = 0x7FE00010

#: Dimension Index Pointers that index space (position, stack), not a further dimension.
SPATIAL_DIMENSIONS = frozenset({0x00200032, 0x00209056, 0x00209057, 0x00201041})
#: Dimension Index Pointers that index time.
TEMPORAL_DIMENSIONS = frozenset({
    0x00209128,   # Temporal Position Index
    0x00209153,   # Nominal Cardiac Trigger Delay Time
    0x0020930D,   # Temporal Position Time Offset
    0x00189074,   # Frame Acquisition DateTime
    0x00189151,   # Frame Reference DateTime
})
TEMPORAL_POSITION_INDEX = 0x00209128

POSITION_TOLERANCE = 1e-4      # mm: frames closer than this share one slice position


# ------------------------------------------------------------------------------------------------
# frames and their geometry
# ------------------------------------------------------------------------------------------------

@dataclass
class Frame:
    """One frame of a multi-frame object with its geometry.

    The DICOM-keyword attributes let the phase 1 helpers (``_sorted_stack``) treat frames like
    single-frame datasets.
    """

    file: int                                  # index of the file in the series
    index: int                                 # 0-based frame index in that file
    ImagePositionPatient: np.ndarray
    ImageOrientationPatient: np.ndarray
    PixelSpacing: Tuple[float, float]
    Rows: int
    Columns: int
    SeriesInstanceUID: str
    SpacingBetweenSlices: Optional[float] = None
    SliceThickness: Optional[float] = None
    stack_id: Optional[str] = None
    dimensions: Dict[int, Any] = field(default_factory=dict)   # Dimension Index Pointer -> index value
    rescale: Optional[Tuple[float, float]] = None              # (slope, intercept), None if absent
    rescale_type: Optional[str] = None
    segment: Optional[int] = None                              # SEG: Referenced Segment Number


def _first(item, keyword: str):
    """The first item of sequence ``keyword`` in ``item`` (a functional groups item), or None."""
    if item is None or keyword not in item:
        return None
    sequence = item[keyword].value
    return sequence[0] if len(sequence) else None


def functional_group(shared, per_frame, keyword: str):
    """A functional group macro of one frame: its per-frame item, else the shared one (KB ch02 4.1).

    The standard forbids a group in both places; per-frame first is the defensive reading.
    """
    found = _first(per_frame, keyword)
    return found if found is not None else _first(shared, keyword)


def _float_or_none(value) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def frames_of(ds, file_index: int = 0) -> List[Frame]:
    """The frames of a multi-frame dataset with the geometry from its functional groups."""
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    shared = ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds and len(ds.SharedFunctionalGroupsSequence) else None
    per_frame = list(ds.PerFrameFunctionalGroupsSequence) if "PerFrameFunctionalGroupsSequence" in ds else [None] * n
    if len(per_frame) != n:
        raise GeometryError(f"NumberOfFrames is {n}, but the Per-frame Functional Groups Sequence has {len(per_frame)} items")
    pointers = [int(item.DimensionIndexPointer) for item in getattr(ds, "DimensionIndexSequence", [])]
    series = str(getattr(ds, "SeriesInstanceUID", ""))
    frames = []
    for index, item in enumerate(per_frame):
        position = functional_group(shared, item, "PlanePositionSequence")
        position = getattr(position, "ImagePositionPatient", None) if position is not None else getattr(ds, "ImagePositionPatient", None)
        orientation = functional_group(shared, item, "PlaneOrientationSequence")
        orientation = getattr(orientation, "ImageOrientationPatient", None) if orientation is not None else getattr(ds, "ImageOrientationPatient", None)
        measures = functional_group(shared, item, "PixelMeasuresSequence")
        measures = measures if measures is not None else ds
        spacing = getattr(measures, "PixelSpacing", None)
        if position is None or orientation is None or spacing is None:
            missing = [name for name, value in (("Plane Position", position), ("Plane Orientation", orientation),
                                                ("Pixel Spacing", spacing)) if value is None]
            raise GeometryError(f"frame {index} has no {', '.join(missing)} in its functional groups; "
                                "it cannot be placed in patient space")
        content = _first(item, "FrameContentSequence")
        dimensions: Dict[int, Any] = {}
        if content is not None and "DimensionIndexValues" in content:
            values = content.DimensionIndexValues
            values = list(values) if isinstance(values, (list, tuple, _pydicom().multival.MultiValue)) else [values]
            if len(values) == len(pointers):
                dimensions = {pointer: int(value) for pointer, value in zip(pointers, values)}
        if content is not None and "TemporalPositionIndex" in content and TEMPORAL_POSITION_INDEX not in dimensions:
            dimensions[TEMPORAL_POSITION_INDEX] = int(content.TemporalPositionIndex)
        transformation = functional_group(shared, item, "PixelValueTransformationSequence")
        source = transformation if transformation is not None else ds
        rescale = None
        if "RescaleSlope" in source or "RescaleIntercept" in source:
            rescale = (float(getattr(source, "RescaleSlope", 1) or 1), float(getattr(source, "RescaleIntercept", 0) or 0))
        segment = _first(item, "SegmentIdentificationSequence") or _first(shared, "SegmentIdentificationSequence")
        frames.append(Frame(
            file=file_index, index=index,
            ImagePositionPatient=np.array([float(v) for v in position]),
            ImageOrientationPatient=np.array([float(v) for v in orientation]),
            PixelSpacing=(float(spacing[0]), float(spacing[1])),
            Rows=int(ds.Rows), Columns=int(ds.Columns), SeriesInstanceUID=series,
            SpacingBetweenSlices=_float_or_none(getattr(measures, "SpacingBetweenSlices", None)),
            SliceThickness=_float_or_none(getattr(measures, "SliceThickness", None)),
            stack_id=str(content.StackID) if content is not None and "StackID" in content else None,
            dimensions=dimensions, rescale=rescale,
            rescale_type=str(getattr(source, "RescaleType", "") or "") or None,
            segment=int(segment.ReferencedSegmentNumber) if segment is not None and "ReferencedSegmentNumber" in segment else None,
        ))
    return frames


def frame_pixels(ds) -> np.ndarray:
    """The decoded frames of a dataset as ``(frames, rows, columns)``, stored values."""
    if int(getattr(ds, "SamplesPerPixel", 1) or 1) != 1:
        raise NotImplementedError("multi-frame DICOM with several samples per pixel (colour) is not supported yet")
    pixels = ds.pixel_array
    return pixels.reshape(int(getattr(ds, "NumberOfFrames", 1) or 1), int(ds.Rows), int(ds.Columns))


def normal_of(frame: Frame) -> np.ndarray:
    iop = np.asarray(frame.ImageOrientationPatient, dtype=np.float64)
    return np.cross(iop[:3], iop[3:])


def _geometry_key(frame: Frame) -> Tuple:
    return (frame.Rows, frame.Columns, tuple(frame.PixelSpacing), tuple(frame.ImageOrientationPatient))


def group_stacks(frames: List[Frame], policy: Policy, report: Report, use_stack_id: bool = True) -> List[List[Frame]]:
    """Frames grouped into geometric stacks (in-plane geometry, and Stack ID) per ``dicom_stacks``."""
    groups: List[Tuple[Tuple, Optional[str], List[Frame]]] = []
    for frame in frames:
        key, stack_id = _geometry_key(frame), frame.stack_id if use_stack_id else None
        for group_key, group_id, members in groups:
            if group_id == stack_id and _same(group_key, key):
                members.append(frame)
                break
        else:
            groups.append((key, stack_id, [frame]))
    return choose_groups([members for _, _, members in groups], policy, report, "geometric stacks")


def choose_groups(groups: List[List[Frame]], policy: Policy, report: Report, what: str) -> List[List[Frame]]:
    """Apply ``dicom_stacks`` to frames that form several images."""
    if len(groups) <= 1:
        return groups
    sizes = sorted((len(g) for g in groups), reverse=True)
    if policy.dicom_stacks == "error":
        raise GeometryError(f"the frames form {len(groups)} {what} {sizes} (policy dicom_stacks='error')")
    if policy.dicom_stacks == "largest":
        largest = max(groups, key=len)
        report.warn(f"the frames form {len(groups)} {what} {sizes}; kept the largest ({len(largest)} frames)")
        report.choose("dicom_stacks", "largest")
        return [largest]
    report.warn(f"the frames form {len(groups)} {what} {sizes}; read as separate images")
    report.choose("dicom_stacks", "split")
    return groups


def cluster_positions(frames: List[Frame]) -> List[List[Frame]]:
    """Frames grouped by slice position, sorted along the slice normal (never by frame order)."""
    normal = normal_of(frames[0])
    projections = np.array([frame.ImagePositionPatient @ normal for frame in frames])
    order = np.argsort(projections, kind="stable")
    clusters: List[List[Frame]] = []
    last = None
    for n in order:
        frame = frames[n]
        if last is not None and abs(projections[n] - last) <= POSITION_TOLERANCE:
            reference = clusters[-1][0].ImagePositionPatient
            if np.linalg.norm(frame.ImagePositionPatient - reference) > POSITION_TOLERANCE:
                raise GeometryError("frames at one slice position along the normal are shifted in-plane "
                                    f"({reference.tolist()} vs {frame.ImagePositionPatient.tolist()}); cannot form one grid")
            clusters[-1].append(frame)
        else:
            clusters.append([frame])
            last = projections[n]
    return clusters


# ------------------------------------------------------------------------------------------------
# non-spatial dimensions
# ------------------------------------------------------------------------------------------------

def _keyword(tag: int) -> str:
    from pydicom.datadict import keyword_for_tag

    return keyword_for_tag(tag) or f"({tag >> 16:04X},{tag & 0xFFFF:04X})"


def split_dimensions(stack: List[Frame], policy: Policy, report: Report) -> List[Tuple[List[Frame], List[int], Optional[str]]]:
    """[(frames, varying dimension pointers, axis type or None)] for the images of one stack."""
    keys = [{tag: value for tag, value in frame.dimensions.items() if tag not in SPATIAL_DIMENSIONS} for frame in stack]
    tags = sorted({tag for key in keys for tag in key})
    varying = [tag for tag in tags if len({key.get(tag) for key in keys}) > 1]
    if not varying:
        return [(stack, [], None)]
    temporal = [tag for tag in varying if tag in TEMPORAL_DIMENSIONS]
    other = [tag for tag in varying if tag not in TEMPORAL_DIMENSIONS]
    if not other:
        return [(stack, temporal, "time")]
    if not temporal and len(other) == 1:
        return [(stack, other, "other")]
    parts: Dict[Tuple, List[Frame]] = defaultdict(list)
    for frame, key in zip(stack, keys):
        parts[tuple(key.get(tag) for tag in other)].append(frame)
    names = ", ".join(_keyword(tag) for tag in other)
    report.warn(f"the frames vary in several non-spatial dimensions ({', '.join(_keyword(t) for t in varying)}); "
                f"split by {names}")
    groups = choose_groups([parts[key] for key in sorted(parts, key=lambda k: tuple((v is None, v) for v in k))],
                           policy, report, f"images by {names}")
    out = []
    for group in groups:
        varying_here = [tag for tag in temporal if len({frame.dimensions.get(tag) for frame in group}) > 1]
        out.append((group, varying_here, "time" if varying_here else None))
    return out


def grid_of(frames: List[Frame], dimensions: List[int], policy: Policy, report: Report) -> List[Tuple[List[List[Frame]], np.ndarray]]:
    """Frames arranged on regular grids: [(grid[dimension index][slice], slice step vector)].

    Every slice position must hold one frame per value of the varying dimensions.
    """
    clusters = cluster_positions(frames)
    if not dimensions:
        if any(len(cluster) > 1 for cluster in clusters):
            raise GeometryError("several frames share one slice position and no non-spatial dimension (time, echo, "
                                "...) tells them apart; this is not a single volume")
        values: List[Tuple] = [()]
        by_position = [{(): cluster[0]} for cluster in clusters]
    else:
        by_position = []
        for cluster in clusters:
            entry: Dict[Tuple, Frame] = {}
            for frame in cluster:
                key = tuple(frame.dimensions.get(tag) for tag in dimensions)
                if key in entry:
                    raise GeometryError(f"two frames at one slice position have the same {', '.join(_keyword(t) for t in dimensions)} {key}")
                entry[key] = frame
            by_position.append(entry)
        values = sorted(by_position[0], key=lambda k: tuple((v is None, v) for v in k))
        if any(set(entry) != set(values) for entry in by_position):
            raise GeometryError(f"the values of {', '.join(_keyword(t) for t in dimensions)} differ between slice positions; "
                                "the frames do not form a complete 4-D grid")
    representatives = [entry[values[0]] for entry in by_position]
    position_of = {id(entry[values[0]]): entry for entry in by_position}
    grids = []
    for run, step in _sorted_stack(representatives, policy, report):
        grids.append(([[position_of[id(rep)][value] for rep in run] for value in values], step))
    return grids


# ------------------------------------------------------------------------------------------------
# reading
# ------------------------------------------------------------------------------------------------

def is_enhanced(ds) -> bool:
    """True for objects with functional groups (enhanced and legacy-converted multi-frame)."""
    return "PerFrameFunctionalGroupsSequence" in ds or "SharedFunctionalGroupsSequence" in ds


def read_enhanced(datasets: List, policy: Policy, report: Report) -> List[Image]:
    """The images of one series of enhanced multi-frame files."""
    frames = [frame for file_index, ds in enumerate(datasets) for frame in frames_of(ds, file_index)]
    pixels = [frame_pixels(ds) for ds in datasets]
    images = []
    for stack in group_stacks(frames, policy, report):
        for part, dimensions, axis_type in split_dimensions(stack, policy, report):
            if axis_type is not None:
                names = ", ".join(_keyword(tag) for tag in dimensions)
                report.choose("dicom_4d_axis", f"{axis_type} ({names})")
            for grid, step in grid_of(part, dimensions, policy, report):
                images.append(_image(grid, step, axis_type, datasets, pixels, policy, report))
    return images


def frame_header(datasets: List, frames: Sequence[Frame]) -> Dict[str, Any]:
    """``shared`` / ``per_file`` / ``file_names`` of the files and ``frames`` in the given order."""
    headers = []
    for ds in datasets:
        top = _pydicom().dataset.Dataset()
        for element in ds:
            if element.tag not in (PER_FRAME_GROUPS, PIXEL_DATA):
                top.add(element)
        if getattr(ds, "file_meta", None) is not None:
            top.file_meta = ds.file_meta
        headers.append(_file_dict(top))
    shared, per_file = split_shared(headers)
    entries = []
    for frame in frames:
        ds = datasets[frame.file]
        groups = ds.PerFrameFunctionalGroupsSequence[frame.index] if "PerFrameFunctionalGroupsSequence" in ds else None
        entries.append({"file": frame.file, "frame_index": frame.index,
                        "functional_groups": dataset_to_dict(groups) if groups is not None else {}})
    return {"shared": shared, "per_file": per_file,
            "file_names": [str(getattr(ds, "filename_relative", "")) for ds in datasets], "frames": entries}


def _image(grid: List[List[Frame]], step, axis_type: Optional[str], datasets: List, pixels: List[np.ndarray],
           policy: Policy, report: Report) -> Image:
    """The image of frames on a grid; ``step`` is the slice step vector or a ``dicom_resample.Resampling``."""
    from medvol.plugins import dicom_resample
    from medvol.plugins.dicom import _resample_volume

    first = grid[0][0]
    iop = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    row_spacing, column_spacing = first.PixelSpacing
    resampling = step if isinstance(step, dicom_resample.Resampling) else None
    uid = getattr(datasets[first.file], "FrameOfReferenceUID", None)
    reference = {"scheme": "dicom_frame_of_reference", "id": str(uid)} if uid else None

    def space_of(slice_step, origin):
        linear = np.column_stack([slice_step, iop[3:] * row_spacing, iop[:3] * column_spacing])
        return patient_space(homogeneous(linear, origin), ["z", "y", "x"], policy.world_frame, reference=reference)

    ordered = [frame for row in grid for frame in row]
    array = np.stack([np.stack([pixels[f.file][f.index] for f in row]) for row in grid])
    values, pairs = _values(ordered, datasets[first.file], policy, report)
    if pairs is not None:
        from medvol.plugins.dicom_rescale import to_physical

        flat, kind = to_physical(array.reshape(-1, *array.shape[-2:]), pairs)
        array = flat.reshape(array.shape)
        report.choose("dicom_rescale", f"apply: physical values slope * stored + intercept per frame, {kind}")
        report.warn(f"rescale slope/intercept differ between frames {sorted(set(pairs))}: stored the physical values "
                    f"as {array.dtype.name} (policy dicom_rescale='apply'); values.scale/offset not set")
    axes: List[Dict[str, Any]] = [{"name": "z", "type": "space"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    index_order = [2, 1, 0]
    if axis_type is None:
        array = array[0]
    else:
        axes.insert(0, {"name": "t" if axis_type == "time" else "u", "type": axis_type})
        index_order = [3, 2, 1, 0]
    header = frame_header(datasets, ordered)
    if resampling is None:
        space = space_of(step, first.ImagePositionPatient)
        fp = fingerprint.compute(array, Meta(axes=axes, spaces=[space], values=values, shape=array.shape))
    else:
        array, fp, values = _resample_volume(array, resampling, iop, (row_spacing, column_spacing), values,
                                             space_of, axes, report)
        space = space_of(resampling.step, resampling.origin)
    source = {
        "format": "dicom",
        "files": header["file_names"],
        "reader": {"name": "medvol"},
        "index_map": {"axis_order": index_order, "flipped": [False] * len(index_order), "offset": [0] * len(index_order)},
        "fingerprint": fp,
        "header": header,
    }
    return Image(array, Meta(axes=axes, spaces=[space], values=values, source=[source], shape=array.shape))


def _values(frames: List[Frame], ds, policy: Policy, report: Report) -> Tuple[Dict[str, Any], Optional[List]]:
    """(values, per-frame rescale pairs to apply (frames in array order) or None)."""
    pairs = {frame.rescale for frame in frames}
    values: Dict[str, Any] = {"kind": "measurement"}
    apply = None
    if len(pairs) > 1:
        if policy.dicom_rescale == "error":
            raise GeometryError("rescale slope/intercept differ between frames; one MLArray holds one scale "
                                "(policy dicom_rescale='error'; dicom_rescale='apply' stores the physical values)")
        apply = [frame.rescale for frame in frames]
    else:
        pair = pairs.pop()
        if pair is not None:
            values["scale"], values["offset"] = pair
    rescale_type = (frames[0].rescale_type or "").upper()
    if rescale_type == "HU" or (not rescale_type and str(getattr(ds, "Modality", "")) == "CT"):
        values["unit"] = "hounsfield_unit"
    if "PixelPaddingValue" in ds:
        if apply is None:
            values["fill"] = int(ds.PixelPaddingValue)
        else:
            report.warn(f"PixelPaddingValue {int(ds.PixelPaddingValue)} is a stored value; with per-frame rescaling "
                        "applied it maps to a different physical value per frame, so values.fill is not set")
    return values, apply


# ------------------------------------------------------------------------------------------------
# writing: re-emit the original multi-frame file(s)
# ------------------------------------------------------------------------------------------------

def write_enhanced(image: Image, directory: str, policy: Policy) -> Report:
    """Re-emit the original multi-frame file(s) of ``image`` into ``directory`` (spec 7.6).

    Exact while the geometry and the rescale are unchanged: every element and functional group
    comes from ``source``, the pixel data from the array. Otherwise:

    - geometry changed: Plane Position (per frame), Plane Orientation and Pixel Measures (Pixel
      Spacing, and Spacing Between Slices where present) are regenerated from the core in the
      functional groups where the source has them (shared or per frame);
    - ``values.scale``/``values.offset`` differ from the source rescale: the Pixel Value
      Transformation is regenerated from the core;
    - physical values stored by ``dicom_rescale='apply'`` (float array, integer source): every
      frame is inverted with its own original rescale and must reproduce the array exactly, else
      the write is refused (``dicom_rescale.invert``);
    - data or geometry changed: data-dependent elements are dropped and new SOP Instance UIDs
      generated, as for single-frame series.
    """
    from medvol.plugins.dicom_rescale import RescaleError, invert, stored_dtype
    from medvol.plugins.dicom_write import (DATA_DEPENDENT_DROP, EXPLICIT_LE, RLE_LOSSLESS, UNCOMPRESSED,
                                            _ds_string, _regenerated_geometry, dict_to_dataset)

    pydicom = _pydicom()
    report = Report("write", "dicom", str(directory))
    source = next(entry for entry in image.meta.source if entry.format == "dicom")
    header = source.header
    shared, per_file, names, frames = header["shared"], header["per_file"], header["file_names"], header["frames"]
    array = image.array
    if array.ndim < 3 or int(np.prod(array.shape[:-2])) != len(frames):
        raise GeometryError(f"the array {array.shape} has {int(np.prod(array.shape[:-2])) if array.ndim >= 3 else 0} "
                            f"frames but the source has {len(frames)}; re-emitting needs one array frame per source "
                            "frame (a resampled image with a different slice count cannot be written as the original object)")
    fp = source.fingerprint or {}
    geometry_unchanged = fp.get("geometry") == fingerprint.geometry_hash(image.meta) and list(fp.get("shape", [])) == list(array.shape)
    data_unchanged = geometry_unchanged and fp.get("data") == fingerprint.data_hash(array)
    names_of_axes = [axis.name for axis in image.meta.axes]
    nz = array.shape[names_of_axes.index("z")]
    if not geometry_unchanged:
        positions, orientation, spacing = _regenerated_geometry(image)
        slice_spacing = abs(float(np.dot(positions[1] - positions[0], np.cross(orientation[:3], orientation[3:])))) if nz > 1 else None
        report.choose("geometry", "regenerated from the core")
    if not data_unchanged:
        report.warn("the data or geometry changed since the object was read: new SOP Instance UIDs, "
                    f"{', '.join(DATA_DEPENDENT_DROP)} dropped")
    flat = array.reshape(len(frames), *array.shape[-2:])
    values = image.meta.values
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for file_index, (fields, name) in enumerate(zip(per_file, names)):
        merged = {**shared, **fields}
        merged = {key: merged[key] for key in list(shared) + [key for key in fields if key not in shared]}
        dataset = dict_to_dataset(merged, report)
        meta = dict_to_dataset(merged.get("file_meta", {}), report)
        slots = sorted((entry["frame_index"], slot) for slot, entry in enumerate(frames) if entry["file"] == file_index)
        count = int(getattr(dataset, "NumberOfFrames", 1) or 1)
        if [index for index, _ in slots] != list(range(count)):
            raise GeometryError(f"the source frames of {name} do not cover its {count} frames")
        if "PerFrameFunctionalGroupsSequence" in merged or any(frames[slot]["functional_groups"] for _, slot in slots):
            dataset.PerFrameFunctionalGroupsSequence = pydicom.sequence.Sequence(
                [dict_to_dataset(frames[slot]["functional_groups"], report) for _, slot in slots])
        if not geometry_unchanged:
            _set_geometry(dataset, [positions[slot % nz] for _, slot in slots], orientation, spacing, slice_spacing,
                          _ds_string)
        data = np.ascontiguousarray(flat[[slot for _, slot in slots]])
        target_dtype = stored_dtype(int(dataset.BitsAllocated), int(getattr(dataset, "PixelRepresentation", 0) or 0))
        if data.dtype.kind == "f":
            if values.scale is None and values.offset is None:
                pairs = [frame.rescale for frame in frames_of(dataset)]
                what = "with each frame's original rescale"
            else:
                pairs = [None] * len(slots)
                what = "as stored values (values.scale/offset set)"
                _set_rescale(dataset, values, report, _ds_string)
            try:
                data = invert(data, pairs, target_dtype)
            except RescaleError as error:
                raise RescaleError(f"{name}: cannot write the {flat.dtype.name} array {what} as {target_dtype.name}: {error}") from None
            report.choose("dicom_rescale", f"inverted {what} to {target_dtype.name}: exact")
        else:
            _set_rescale(dataset, values, report, _ds_string)
        if data.dtype.itemsize * 8 != int(dataset.BitsAllocated):
            raise NotImplementedError(f"the array dtype {data.dtype} no longer matches BitsAllocated {dataset.BitsAllocated}")
        dataset.Rows, dataset.Columns = data.shape[-2:]
        if not data_unchanged:
            for key in DATA_DEPENDENT_DROP:
                if key in dataset:
                    del dataset[key]
            uid = pydicom.uid.generate_uid()
            dataset.SOPInstanceUID = uid
            meta.MediaStorageSOPInstanceUID = uid
        dataset.PixelData = data.astype(data.dtype.newbyteorder("<")).tobytes()
        transfer = str(getattr(meta, "TransferSyntaxUID", EXPLICIT_LE))
        if transfer not in UNCOMPRESSED and transfer != RLE_LOSSLESS:
            report.warn(f"transfer syntax {transfer} is not re-encoded: written as explicit VR little endian")
            transfer = EXPLICIT_LE
        meta.TransferSyntaxUID = transfer if transfer != RLE_LOSSLESS else EXPLICIT_LE
        dataset.file_meta = pydicom.dataset.FileMetaDataset(meta)
        dataset.file_meta.TransferSyntaxUID = meta.TransferSyntaxUID
        if transfer == RLE_LOSSLESS:
            dataset.compress(RLE_LOSSLESS)
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_as(str(path), enforce_file_format=True)
    return report


def _group_items(dataset, keyword: str):
    """(shared item or None, [per-frame items]) that hold functional group ``keyword``."""
    shared = None
    if "SharedFunctionalGroupsSequence" in dataset and len(dataset.SharedFunctionalGroupsSequence):
        item = dataset.SharedFunctionalGroupsSequence[0]
        shared = item if keyword in item and len(item[keyword].value) else None
    per_frame = []
    for item in getattr(dataset, "PerFrameFunctionalGroupsSequence", []):
        per_frame.append(item[keyword].value[0] if keyword in item and len(item[keyword].value) else None)
    return (shared[keyword].value[0] if shared is not None else None), per_frame


def _set_geometry(dataset, positions: List[np.ndarray], orientation, spacing, slice_spacing: Optional[float], ds_string) -> None:
    """Plane Position, Plane Orientation and Pixel Measures of every frame, from the core.

    Each group is rewritten where the source has it (per frame and/or shared). Plane Position
    is per frame; a shared Plane Position is only kept when every frame of the file gets the same
    position, otherwise it moves to the per-frame groups.
    """
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    iop = [ds_string(v) for v in orientation]
    shared_position, per_position = _group_items(dataset, "PlanePositionSequence")
    per_frame = list(getattr(dataset, "PerFrameFunctionalGroupsSequence", []))
    distinct = {tuple(np.round(p, 12)) for p in positions}
    if shared_position is not None and len(distinct) > 1:
        del dataset.SharedFunctionalGroupsSequence[0].PlanePositionSequence
        shared_position = None
    for n, item in enumerate(per_frame):
        if per_position[n] is None and shared_position is None:
            item.PlanePositionSequence = Sequence([Dataset()])
            per_position[n] = item.PlanePositionSequence[0]
    if shared_position is not None:
        shared_position.ImagePositionPatient = [ds_string(v) for v in positions[0]]
    for n, position_item in enumerate(per_position):
        if position_item is not None:
            position_item.ImagePositionPatient = [ds_string(v) for v in positions[n]]
    if not per_frame and "ImagePositionPatient" in dataset:
        dataset.ImagePositionPatient = [ds_string(v) for v in positions[0]]

    shared_orientation, per_orientation = _group_items(dataset, "PlaneOrientationSequence")
    for item in [shared_orientation, *per_orientation]:
        if item is not None:
            item.ImageOrientationPatient = iop
    if shared_orientation is None and all(i is None for i in per_orientation) and "ImageOrientationPatient" in dataset:
        dataset.ImageOrientationPatient = iop

    shared_measures, per_measures = _group_items(dataset, "PixelMeasuresSequence")
    for item in [shared_measures, *per_measures]:
        if item is not None:
            item.PixelSpacing = [ds_string(v) for v in spacing]
            if slice_spacing is not None and "SpacingBetweenSlices" in item:
                item.SpacingBetweenSlices = ds_string(slice_spacing)
    if shared_measures is None and all(i is None for i in per_measures) and "PixelSpacing" in dataset:
        dataset.PixelSpacing = [ds_string(v) for v in spacing]


def _set_rescale(dataset, values, report: Report, ds_string) -> None:
    """Regenerate the Pixel Value Transformation from ``values.scale``/``offset`` where it differs."""
    expected = (float(values.scale if values.scale is not None else 1.0), float(values.offset or 0.0))
    current = [frame.rescale for frame in frames_of(dataset)]
    if all(pair == expected for pair in current) or (values.scale is None and values.offset is None
                                                     and all(pair is None for pair in current)):
        return
    shared, per_frame = _group_items(dataset, "PixelValueTransformationSequence")
    items = [item for item in [shared, *per_frame] if item is not None]
    if not items and ("RescaleSlope" in dataset or "RescaleIntercept" in dataset):
        items = [dataset]
    if not items:
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence

        if "SharedFunctionalGroupsSequence" not in dataset or not len(dataset.SharedFunctionalGroupsSequence):
            dataset.SharedFunctionalGroupsSequence = Sequence([Dataset()])
        item = Dataset()
        item.RescaleType = "US"
        dataset.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence = Sequence([item])
        items = [item]
    for item in items:
        item.RescaleSlope = ds_string(expected[0])
        item.RescaleIntercept = ds_string(expected[1])
    report.choose("rescale", f"regenerated from the core: slope {expected[0]}, intercept {expected[1]}")
