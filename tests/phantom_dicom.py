"""Synthetic enhanced multi-frame DICOM and DICOM SEG files for the phase 2 and 3 readers.

Written with pydicom and highdicom only, never with MedVol, so the tests compare MedVol against
independent ground truth:

- ``write_enhanced``: Enhanced CT / Enhanced MR, 3-D or 4-D, with the frames in a scrambled order,
  geometry in the Shared or the Per-frame Functional Groups (Canon vs Philips / Siemens XA, KB ch02
  4.1), optionally one file per time point (Siemens XA style). Each voxel encodes its own file index
  as in ``phantom.Truth``; a sheared truth gives a tilted acquisition.
- ``write_seg``: DICOM SEG (binary, fractional or label map) of masks given on the grid of a
  single-frame series written by ``phantom.write_dicom_series``, with the frames shuffled afterwards
  (frame order is not spatial, KB ch02 D2/D3).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import phantom

ENHANCED_CT = "1.2.840.10008.5.1.4.1.1.2.1"
ENHANCED_MR = "1.2.840.10008.5.1.4.1.1.4.1"

TAG_IMAGE_POSITION = 0x00200032
TAG_STACK_ID = 0x00209056
TAG_IN_STACK_POSITION = 0x00209057
TAG_TEMPORAL_POSITION = 0x00209128
TAG_PLANE_POSITION_SEQUENCE = 0x00209113
TAG_FRAME_CONTENT_SEQUENCE = 0x00209111


def _ds(value):
    import pydicom

    return pydicom.valuerep.DSfloat(value, auto_format=True)


def _item(**fields):
    from pydicom.dataset import Dataset

    item = Dataset()
    for key, value in fields.items():
        setattr(item, key, value)
    return item


def _dimension(pointer, group):
    return _item(DimensionIndexPointer=pointer, FunctionalGroupPointer=group)


def write_enhanced(path, truth: phantom.Truth = None, order: Optional[Sequence[int]] = None, mr: bool = False,
                   per_frame_geometry: bool = False, file_per_time_point: bool = False, rescale=(1.0, 0.0),
                   extra=None) -> List[str]:
    """Enhanced CT (or MR with ``mr=True``) of ``truth`` (shape (i, j, k) or (i, j, k, t)).

    Frames are generated time-fastest (all time points of slice 0 first, as vendors do) and then
    written in ``order`` (default: a fixed scrambled permutation). ``path`` is a file, or a directory
    when ``file_per_time_point`` is set. Returns the written paths.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence as Seq
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    truth = truth or phantom.Truth(dtype="int16")
    shape = tuple(truth.shape) + (1,) * (4 - len(truth.shape))
    ni, nj, nk, nt = shape
    volume = truth.volume().astype(np.int16).reshape(shape, order="F")
    a = truth.affine_lps
    u = a[:3, 0] / np.linalg.norm(a[:3, 0])
    v = a[:3, 1] / np.linalg.norm(a[:3, 1])
    row_spacing, col_spacing = np.linalg.norm(a[:3, 1]), np.linalg.norm(a[:3, 0])
    four_d = len(truth.shape) == 4
    frames = [(k, t) for k in range(nk) for t in range(nt)]
    study, series, frame_of_reference = generate_uid(), generate_uid(), generate_uid()
    sop_class = ENHANCED_MR if mr else ENHANCED_CT

    def geometry_groups():
        return {
            "PixelMeasuresSequence": Seq([_item(PixelSpacing=[_ds(row_spacing), _ds(col_spacing)], SliceThickness=_ds(1.5))]),
            "PlaneOrientationSequence": Seq([_item(ImageOrientationPatient=[_ds(x) for x in (*u, *v)])]),
        }

    def dataset(file_frames, instance):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = sop_class
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame_of_reference
        ds.Modality = "MR" if mr else "CT"
        ds.PatientName, ds.PatientID = "Phantom^Enhanced", "ENHANCED"
        ds.SeriesNumber, ds.InstanceNumber = 4, instance
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL", "NONE"]
        ds.Rows, ds.Columns = nj, ni
        ds.NumberOfFrames = len(file_frames)
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        dimensions = [_dimension(TAG_STACK_ID, TAG_FRAME_CONTENT_SEQUENCE),
                      _dimension(TAG_IN_STACK_POSITION, TAG_FRAME_CONTENT_SEQUENCE)]
        if four_d:
            dimensions.append(_dimension(TAG_TEMPORAL_POSITION, TAG_FRAME_CONTENT_SEQUENCE))
        ds.DimensionIndexSequence = Seq(dimensions)
        shared = Dataset()
        if not per_frame_geometry:
            for key, value in geometry_groups().items():
                setattr(shared, key, value)
        if not mr:
            shared.PixelValueTransformationSequence = Seq([_item(RescaleSlope=_ds(rescale[0]), RescaleIntercept=_ds(rescale[1]),
                                                                 RescaleType="HU")])
        ds.SharedFunctionalGroupsSequence = Seq([shared])
        per_frame = []
        pixels = []
        for k, t in file_frames:
            item = Dataset()
            content = _item(StackID="1", InStackPositionNumber=k + 1,
                            DimensionIndexValues=[1, k + 1] + ([t + 1] if four_d else []))
            if four_d:
                content.TemporalPositionIndex = t + 1
            item.FrameContentSequence = Seq([content])
            item.PlanePositionSequence = Seq([_item(ImagePositionPatient=[_ds(x) for x in a[:3, 3] + k * a[:3, 2]])])
            if per_frame_geometry:
                for key, value in geometry_groups().items():
                    setattr(item, key, value)
            per_frame.append(item)
            pixels.append(np.ascontiguousarray(volume[:, :, k, t].T))     # (rows j, columns i)
        ds.PerFrameFunctionalGroupsSequence = Seq(per_frame)
        ds.add_new(0x00290010, "LO", "SYNTH PRIVATE")
        ds.add_new(0x00291010, "OB", bytes([instance, 1, 2, 3]))
        for key, value in (extra or {}).items():
            setattr(ds, key, value)
        ds.PixelData = np.stack(pixels).astype("<i2").tobytes()
        return ds

    rng = np.random.default_rng(7)
    paths = []
    if file_per_time_point:
        os.makedirs(path, exist_ok=True)
        for t in rng.permutation(nt):
            file_frames = [frames[n] for n in rng.permutation(len(frames)) if frames[n][1] == t]
            target = os.path.join(path, f"enh{t:03d}.dcm")
            dataset(file_frames, int(t) + 1).save_as(target, enforce_file_format=True)
            paths.append(target)
        return paths
    order = list(order) if order is not None else list(rng.permutation(len(frames)))
    dataset([frames[n] for n in order], 1).save_as(str(path), enforce_file_format=True)
    return [str(path)]


# ---------------------------------------------------------------------------------------------
# DICOM SEG
# ---------------------------------------------------------------------------------------------

SEGMENTS = {
    1: {"name": "liver", "code": ("SCT", "10200004", "Liver"), "color": (221, 130, 101)},
    2: {"name": "spleen", "code": ("SCT", "78961009", "Spleen"), "color": (157, 108, 162)},
    3: {"name": "kidney", "code": ("SCT", "64033007", "Kidney"), "color": (185, 102, 83)},
}


#: Patient and study attributes (type 2) that highdicom copies from the source images.
SERIES_EXTRA = {"PatientBirthDate": "19700101", "PatientSex": "O", "StudyDate": "20261106", "StudyTime": "101500",
                "ReferringPhysicianName": "", "StudyID": "1", "AccessionNumber": "A1"}


def write_ct_series(directory, truth: phantom.Truth = None):
    """``phantom.write_dicom_series`` plus the patient and study attributes a SEG needs."""
    return phantom.write_dicom_series(directory, truth or phantom.Truth(dtype="int16"), extra=SERIES_EXTRA)


def _sorted_series(series_dir, truth: phantom.Truth):
    """The datasets of a single-frame series sorted by the truth's k index (the masks' slice order).

    highdicom pairs mask frame f with source image f in the given order (KB ch02 pitfall D4).
    """
    import pydicom

    datasets = [pydicom.dcmread(os.path.join(series_dir, name)) for name in sorted(os.listdir(series_dir))
                if name.endswith(".dcm")]
    inverse = np.linalg.inv(truth.affine_lps)
    return sorted(datasets, key=lambda ds: float((inverse @ np.r_[np.array(ds.ImagePositionPatient, dtype=float), 1.0])[2]))


def segment_description(number: int, label: Dict):
    import highdicom as hd
    from pydicom.sr.coding import Code

    scheme, value, meaning = label["code"]
    color = hd.color.CIELabColor.from_rgb(*label["color"]) if label.get("color") else None
    return hd.seg.SegmentDescription(
        segment_number=number, segment_label=label["name"],
        segmented_property_category=Code("91723000", "SCT", "Anatomical Structure"),
        segmented_property_type=Code(value, scheme, meaning),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.MANUAL, display_color=color,
        tracking_id=f"track-{number}", tracking_uid=hd.UID())


def write_seg(path, series_dir, masks: Dict[int, np.ndarray], segmentation_type: str = "BINARY", shuffle: bool = True,
              omit_empty_frames: bool = True, labels: Dict[int, Dict] = None, truth: phantom.Truth = None) -> str:
    """A SEG of ``masks`` (segment number -> array in file order (i, j, k) of the series' ``truth``).

    BINARY masks are booleans; FRACTIONAL masks are floats in [0, 1]; LABELMAP takes one integer
    array under key 0 whose values are segment numbers. With ``shuffle`` the stored frames are
    permuted afterwards (pixel data and per-frame functional groups together).
    """
    import highdicom as hd
    import pydicom

    labels = labels or SEGMENTS
    sources = _sorted_series(series_dir, truth or phantom.Truth())
    if segmentation_type == "LABELMAP":
        label_map = masks[0]
        numbers = sorted(int(n) for n in np.unique(label_map) if n > 0)
        pixel_array = np.stack([label_map[:, :, k].T for k in range(label_map.shape[2])]).astype(np.uint8)
    else:
        numbers = sorted(masks)
        pixel_array = np.stack([np.stack([masks[n][:, :, k].T for k in range(masks[n].shape[2])]) for n in numbers], axis=-1)
        if segmentation_type == "BINARY":
            pixel_array = pixel_array.astype(np.uint8)
    seg = hd.seg.Segmentation(
        source_images=sources, pixel_array=pixel_array, segmentation_type=segmentation_type,
        segment_descriptions=[segment_description(n, labels[n]) for n in numbers],
        series_instance_uid=hd.UID(), series_number=30, sop_instance_uid=hd.UID(), instance_number=1,
        manufacturer="SYNTH", manufacturer_model_name="phantom", software_versions="1", device_serial_number="0",
        content_label="PHANTOM", content_description="synthetic segments", omit_empty_frames=omit_empty_frames)
    seg.SeriesDescription = "phantom segmentation"
    seg.save_as(str(path))
    if shuffle:
        ds = pydicom.dcmread(str(path))
        frames = ds.pixel_array.reshape(int(ds.NumberOfFrames), ds.Rows, ds.Columns)
        permutation = np.random.default_rng(3).permutation(len(frames))
        if len(frames) > 1 and np.all(permutation == np.arange(len(frames))):
            permutation = permutation[::-1]
        frames = frames[permutation]
        items = list(ds.PerFrameFunctionalGroupsSequence)
        ds.PerFrameFunctionalGroupsSequence = pydicom.sequence.Sequence([items[n] for n in permutation])
        if int(ds.BitsAllocated) == 1:
            ds.PixelData = pydicom.pixels.pack_bits(frames.reshape(-1), pad=True)
        else:
            ds.PixelData = np.ascontiguousarray(frames).astype(frames.dtype.newbyteorder("<")).tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return str(path)


# ---------------------------------------------------------------------------------------------
# RTSTRUCT
# ---------------------------------------------------------------------------------------------

RTSTRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"


def write_rtstruct(path, series_dir, truth: phantom.Truth, rois: List[Dict], frame_of_reference: str = None) -> str:
    """An RT Structure Set on the single-frame series in ``series_dir`` (written from ``truth``).

    ``rois``: ``{number, name, color, code: (scheme, value, meaning) or None, contours: [(k, points,
    type)]}`` with ``points`` an (N, 2) array of (i, j) = (column, row) index coordinates of the
    truth on slice ``k`` (converted to patient mm with the truth's affine; ``k`` may be fractional
    to place a contour off the slice planes) and ``type`` the ContourGeometricType (default
    CLOSED_PLANAR).
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence as Seq
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    images = [pydicom.dcmread(os.path.join(series_dir, name)) for name in sorted(os.listdir(series_dir)) if name.endswith(".dcm")]
    first = images[0]
    frame_of_reference = frame_of_reference or str(first.FrameOfReferenceUID)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = RTSTRUCT_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = RTSTRUCT_STORAGE, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = first.StudyInstanceUID, generate_uid()
    ds.Modality = "RTSTRUCT"
    ds.PatientName, ds.PatientID = first.PatientName, first.PatientID
    ds.SeriesNumber, ds.InstanceNumber = 40, 1
    ds.StructureSetLabel, ds.StructureSetDate, ds.StructureSetTime = "PHANTOM", "20261106", "101500"
    contour_images = Seq([_item(ReferencedSOPClassUID=im.SOPClassUID, ReferencedSOPInstanceUID=im.SOPInstanceUID) for im in images])
    series = _item(SeriesInstanceUID=first.SeriesInstanceUID, ContourImageSequence=contour_images)
    study = _item(ReferencedSOPClassUID="1.2.840.10008.3.1.2.3.1", ReferencedSOPInstanceUID=first.StudyInstanceUID,
                  RTReferencedSeriesSequence=Seq([series]))
    ds.ReferencedFrameOfReferenceSequence = Seq([_item(FrameOfReferenceUID=frame_of_reference, RTReferencedStudySequence=Seq([study]))])
    structure, contours, observations = [], [], []
    a = truth.affine_lps
    for n, roi in enumerate(rois):
        structure.append(_item(ROINumber=roi["number"], ReferencedFrameOfReferenceUID=frame_of_reference, ROIName=roi["name"],
                               ROIGenerationAlgorithm="MANUAL"))
        items = []
        for contour in roi["contours"]:
            k, points = contour[0], np.asarray(contour[1], dtype=float)
            kind = contour[2] if len(contour) > 2 else "CLOSED_PLANAR"
            index = np.c_[points, np.full(len(points), float(k)), np.ones(len(points))]
            world = (a @ index.T)[:3].T
            items.append(_item(ContourGeometricType=kind, NumberOfContourPoints=len(points),
                               ContourData=[_ds(v) for v in world.reshape(-1)]))
        roi_contour = _item(ReferencedROINumber=roi["number"], ContourSequence=Seq(items))
        if roi.get("color") is not None:
            roi_contour.ROIDisplayColor = list(roi["color"])
        contours.append(roi_contour)
        observation = _item(ObservationNumber=n + 1, ReferencedROINumber=roi["number"], RTROIInterpretedType="ORGAN",
                            ROIInterpreter="")
        if roi.get("code") is not None:
            scheme, value, meaning = roi["code"]
            observation.RTROIIdentificationCodeSequence = Seq([_item(CodeValue=value, CodingSchemeDesignator=scheme, CodeMeaning=meaning)])
        observations.append(observation)
    ds.StructureSetROISequence = Seq(structure)
    ds.ROIContourSequence = Seq(contours)
    ds.RTROIObservationsSequence = Seq(observations)
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)
