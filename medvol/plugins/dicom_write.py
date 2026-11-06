"""DICOM writing (spec 7.6): re-emit the original series from ``source``.

Every file is rebuilt from ``shared`` + its ``per_file`` entry with the original UIDs; pixel data
comes from the array. Core-owned elements (geometry, rescale, pixel description) are reused while the
geometry is unchanged and regenerated from the core otherwise; data-dependent elements are kept while
the source fingerprint matches, and otherwise dropped (new SOP Instance UIDs are generated).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from mlarray import fingerprint

from medvol.geometry import GeometryError, lps_matrix
from medvol.image import Image
from medvol.policy import Policy
from medvol.report import Report

UNCOMPRESSED = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}   # big endian (retired) is rewritten as little endian
RLE_LOSSLESS = "1.2.840.10008.1.2.5"
EXPLICIT_LE = "1.2.840.10008.1.2.1"
DATA_DEPENDENT_DROP = ("SmallestImagePixelValue", "LargestImagePixelValue", "SmallestPixelValueInSeries",
                       "LargestPixelValueInSeries", "WindowCenter", "WindowWidth")


def _pydicom():
    import pydicom

    return pydicom


# ------------------------------------------------------------------------------------------------
# readable header -> pydicom Dataset
# ------------------------------------------------------------------------------------------------

def _tag(key: str):
    group, element = key.strip("()").split(",")
    return _pydicom().tag.Tag(int(group, 16), int(element, 16))


def _vr(tag, keyword: str, dataset) -> str:
    from pydicom.datadict import dictionary_VR

    vr = dictionary_VR(tag)
    if " or " not in vr:
        return vr
    options = vr.split(" or ")
    if "US" in options and "SS" in options:
        return "SS" if int(getattr(dataset, "PixelRepresentation", 0) or 0) == 1 else "US"
    if "OW" in options:
        return "OW"
    return options[0]


def _ds_string(value: float) -> str:
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) > 16:
        text = str(_pydicom().valuerep.format_number_as_ds(float(value)))
    return text


def _element_value(value: Any, vr: str, report: Report) -> Any:
    pydicom = _pydicom()
    if value is None:
        return None
    if vr == "SQ":
        return pydicom.sequence.Sequence([dict_to_dataset(item, report) for item in value])
    if isinstance(value, (list, tuple)) and vr not in ("OB", "OW", "OF", "OD", "OL", "OV", "UN"):
        return [_element_value(v, vr, report) for v in value]
    if vr == "DS":
        return _ds_string(value)
    if vr == "IS":
        return str(int(value))
    if vr == "AT":
        return _tag(value)
    return value


def dict_to_dataset(data: Dict[str, Any], report: Report):
    """A pydicom Dataset from the readable representation (keyword or tag keys)."""
    pydicom = _pydicom()
    dataset = pydicom.dataset.Dataset()
    from pydicom.datadict import tag_for_keyword

    # PixelRepresentation first: it resolves ambiguous VRs (US or SS)
    if "PixelRepresentation" in data:
        dataset.PixelRepresentation = data["PixelRepresentation"]
    for key, value in data.items():
        if key == "file_meta":
            continue
        if key.startswith("("):
            tag = _tag(key)
            vr, raw = value["vr"], value["value"]
        else:
            tag = tag_for_keyword(key)
            if tag is None:
                report.warn(f"unknown DICOM keyword {key} skipped")
                continue
            vr, raw = _vr(tag, key, dataset), value
        dataset.add_new(tag, vr, _element_value(raw, vr, report))
    return dataset


# ------------------------------------------------------------------------------------------------
# writer
# ------------------------------------------------------------------------------------------------

def write_series(image: Image, directory: str, policy: Policy) -> Report:
    pydicom = _pydicom()
    report = Report("write", "dicom", str(directory))
    source = next((entry for entry in image.meta.source if entry.format == "dicom"), None)
    if source is None:
        raise NotImplementedError("DICOM is written only for images read from DICOM (re-emitting the original "
                                  "series); new series from other formats are out of scope (spec 7.6)")
    header = source.header
    shared, per_file, names = header["shared"], header["per_file"], header["file_names"]
    array = image.array
    if array.ndim != 3 or array.shape[0] != len(per_file):
        raise GeometryError(f"the array {array.shape} does not match the {len(per_file)} source files any more")
    fp = source.fingerprint or {}
    geometry_unchanged = fp.get("geometry") == fingerprint.geometry_hash(image.meta) and list(fp.get("shape", [])) == list(array.shape)
    data_unchanged = geometry_unchanged and fp.get("data") == fingerprint.data_hash(array)
    if not geometry_unchanged:
        positions, orientation, spacing = _regenerated_geometry(image)
        report.choose("geometry", "regenerated from the core")
    if not data_unchanged:
        report.warn("the data or geometry changed since the series was read: new SOP Instance UIDs, "
                    f"{', '.join(DATA_DEPENDENT_DROP)} dropped")
    values = image.meta.values
    os.makedirs(directory, exist_ok=True)
    series_uid_note = False
    for k, (fields, name) in enumerate(zip(per_file, names)):
        merged = {**shared, **fields}
        merged = {key: merged[key] for key in list(shared) + [key for key in fields if key not in shared]}
        dataset = dict_to_dataset(merged, report)
        meta = dict_to_dataset(merged.get("file_meta", {}), report)
        if not geometry_unchanged:
            dataset.ImagePositionPatient = [_ds_string(v) for v in positions[k]]
            dataset.ImageOrientationPatient = [_ds_string(v) for v in orientation]
            dataset.PixelSpacing = [_ds_string(v) for v in spacing]
        if values.scale is not None:
            dataset.RescaleSlope = _ds_string(values.scale)
            dataset.RescaleIntercept = _ds_string(values.offset or 0.0)
        frame = np.ascontiguousarray(array[k])
        dataset.Rows, dataset.Columns = frame.shape
        if not data_unchanged:
            for key in DATA_DEPENDENT_DROP:
                if key in dataset:
                    del dataset[key]
            uid = pydicom.uid.generate_uid()
            dataset.SOPInstanceUID = uid
            meta.MediaStorageSOPInstanceUID = uid
            series_uid_note = True
        dataset.PixelData = frame.astype(frame.dtype.newbyteorder("<")).tobytes()
        transfer = str(getattr(meta, "TransferSyntaxUID", EXPLICIT_LE))
        if transfer not in UNCOMPRESSED and transfer != RLE_LOSSLESS:
            report.warn(f"transfer syntax {transfer} is not re-encoded: written as explicit VR little endian")
            transfer = EXPLICIT_LE
        meta.TransferSyntaxUID = transfer if transfer != RLE_LOSSLESS else EXPLICIT_LE
        dataset.file_meta = pydicom.dataset.FileMetaDataset(meta)
        dataset.file_meta.TransferSyntaxUID = meta.TransferSyntaxUID
        if transfer == RLE_LOSSLESS:
            dataset.compress(RLE_LOSSLESS)
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_as(str(path), enforce_file_format=True)
    if series_uid_note:
        report.warn("the SeriesInstanceUID is kept; give the series a new one if it replaces the original elsewhere")
    return report


def _regenerated_geometry(image: Image):
    """Per-slice ImagePositionPatient, ImageOrientationPatient and PixelSpacing from the core."""
    space = image.space()
    lps = lps_matrix(space)
    columns = [list(space.from_array.array_axes).index(name) for name in ("z", "y", "x")]
    step, row_vector, column_vector = (lps[:3, c] for c in columns)
    row_spacing, column_spacing = np.linalg.norm(row_vector), np.linalg.norm(column_vector)
    orientation = list(column_vector / column_spacing) + list(row_vector / row_spacing)
    origin = lps[:3, -1]
    positions: List[np.ndarray] = [origin + k * step for k in range(image.array.shape[0])]
    return positions, orientation, [row_spacing, column_spacing]
