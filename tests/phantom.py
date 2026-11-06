"""Conformance phantom (spec section 9.1), ported from the orientation KB experiments.

A small, deliberately asymmetric, oblique and mirrored volume in which every voxel encodes its own
file index: value = linear index of (i, j, k[, t]) in Fortran order. A reader is correct when, for
every voxel, the world position computed from the reader's array index and its metadata equals the
truth for the index the voxel's value encodes.

The generators write files with the reference libraries (nibabel, pynrrd, pydicom) or by hand, never
with MedVol, so they test MedVol against independent ground truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

FLIP = np.diag([-1.0, -1.0, 1.0])     # LPS <-> RAS
FLIP4 = np.diag([-1.0, -1.0, 1.0, 1.0])


def _rot_x(deg):
    a = np.deg2rad(deg)
    return np.array([[1.0, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


# KB ch05 phantom: i -> R, j -> I, k -> P (left-handed, det -1), rotated obliquely.
BASE_LPS = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
DIRECTION_LPS = _rot_z(15) @ _rot_x(10) @ BASE_LPS
SPACING = np.array([0.5, 1.0, 2.0])
ORIGIN_LPS = np.array([12.5, -30.25, 47.0])
SHAPE = (5, 6, 7)


@dataclass
class Truth:
    """Ground truth: file index (i, j, k) -> LPS world (centre of the voxel)."""

    shape: tuple = SHAPE                       # file order (i, j, k[, t ...])
    affine_lps: np.ndarray = field(default_factory=lambda: _affine(DIRECTION_LPS, SPACING, ORIGIN_LPS))
    dtype: str = "int32"

    def volume(self) -> np.ndarray:
        """Array in file order [i, j, k, ...]; each value is its own Fortran-order linear index."""
        n = int(np.prod(self.shape))
        return np.arange(n, dtype=self.dtype).reshape(self.shape, order="F")

    @property
    def affine_ras(self) -> np.ndarray:
        return FLIP4 @ self.affine_lps

    def world_of_values(self, values: np.ndarray) -> np.ndarray:
        """LPS world positions (N x 3) of the voxels whose values are ``values``."""
        ijk = np.array(np.unravel_index(values.astype(np.int64), self.shape, order="F"))[:3]
        return (self.affine_lps @ np.vstack([ijk, np.ones(ijk.shape[1])]))[:3].T


def _affine(direction, spacing, origin):
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(direction) @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def sheared_truth() -> Truth:
    """Gantry-tilt-like shear: the k axis leans along j."""
    affine = _affine(DIRECTION_LPS, SPACING, ORIGIN_LPS)
    affine[:3, 2] += 0.6 * affine[:3, 1]
    return Truth(affine_lps=affine)


def world_error(image, truth: Truth) -> float:
    """Largest distance (mm) between where the image puts each voxel and where it truly is."""
    from medvol.geometry import lps_matrix

    space = image.space()
    matrix = lps_matrix(space)
    axes = [axis.name for axis in image.meta.axes]
    columns = [axes.index(name) for name in space.from_array.array_axes]
    index = np.indices(image.array.shape).reshape(image.array.ndim, -1)
    points = np.vstack([index[columns], np.ones(index.shape[1])])
    world_read = (matrix @ points)[:3].T
    world_true = truth.world_of_values(image.array.reshape(-1))
    return float(np.max(np.linalg.norm(world_read - world_true, axis=1)))


# ---------------------------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------------------------

def write_nifti(path, truth: Truth = None, qform: Optional[np.ndarray] = "truth", sform: Optional[np.ndarray] = "truth",
                qcode: int = 1, scode: int = 1, version: int = 1, extra=None):
    """NIfTI with explicit qform/sform (RAS affines; None = not set) and codes."""
    import nibabel as nib

    truth = truth or Truth()
    cls = nib.Nifti1Image if version == 1 else nib.Nifti2Image
    image = cls(truth.volume(), None)
    header = image.header
    header.set_xyzt_units("mm", "sec")
    qform = truth.affine_ras if isinstance(qform, str) else qform
    sform = truth.affine_ras if isinstance(sform, str) else sform
    header.set_qform(qform, code=qcode if qform is not None else 0)
    header.set_sform(sform, code=scode if sform is not None else 0)
    if qform is None:
        header["pixdim"][1:4] = np.linalg.norm(truth.affine_ras[:3, :3], axis=0)
    for key, value in (extra or {}).items():
        header[key] = value
    image.update_header = lambda: None   # keep the codes exactly as set
    nib.save(image, path)
    return path


# ---------------------------------------------------------------------------------------------
# NRRD
# ---------------------------------------------------------------------------------------------

NRRD_FRAMES = {
    # space name -> 3x3 matrix taking LPS coordinates to the space's coordinates
    "left-posterior-superior": np.eye(3),
    "right-anterior-superior": FLIP,
    "left-anterior-superior": np.diag([1.0, -1.0, 1.0]),
    "scanner-xyz": np.eye(3),          # treated as LPS by ITK and Slicer
    "3D-right-handed": np.eye(3),
}


def write_nrrd(path, truth: Truth = None, space: str = "left-posterior-superior", detached: bool = False, extra=None):
    import nrrd

    truth = truth or Truth()
    to_space = NRRD_FRAMES[space]
    linear = to_space @ truth.affine_lps[:3, :3]
    header = {
        "space": space,
        "space directions": linear.T.copy(),      # pynrrd: one row per axis
        "space origin": to_space @ truth.affine_lps[:3, 3],
        "kinds": ["domain"] * 3,
        "endian": "little",
        "encoding": "raw",
    }
    header.update(extra or {})
    nrrd.write(str(path), truth.volume(), header, detached_header=detached)
    return path


# ---------------------------------------------------------------------------------------------
# MGH / MGZ
# ---------------------------------------------------------------------------------------------

def write_mgh(path, truth: Truth = None):
    import nibabel as nib

    truth = truth or Truth()
    nib.save(nib.MGHImage(truth.volume().astype(np.float32), truth.affine_ras), str(path))
    return path


# ---------------------------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------------------------

def write_dicom_series(directory, truth: Truth = None, order: Sequence[int] = None, extra=None, private=True):
    """Single-frame CT-like series, one file per k-slice, written in a scrambled order.

    IOP = (world direction of increasing column i, of increasing row j); PixelSpacing = [row
    spacing (along j), column spacing (along i)]; a frame is [row j, column i]. A sheared truth
    (``sheared_truth``) gives a gantry-tilted series.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    truth = truth or Truth(dtype="int16")
    os.makedirs(directory, exist_ok=True)
    volume = truth.volume().astype(np.int16)
    a = truth.affine_lps
    u = a[:3, 0] / np.linalg.norm(a[:3, 0])
    v = a[:3, 1] / np.linalg.norm(a[:3, 1])
    row_spacing, col_spacing = np.linalg.norm(a[:3, 1]), np.linalg.norm(a[:3, 0])
    nk = truth.shape[2]
    order = list(order) if order is not None else [3, 0, 6, 1, 5, 2, 4][:nk] if nk == 7 else list(range(nk))[::-1]
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    paths = []
    for n, k in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.Modality = "CT"
        ds.PatientName, ds.PatientID = "Phantom^Orient", "ORIENT"
        ds.SeriesNumber = 3
        ds.InstanceNumber = n + 1                 # deliberately not the slice order
        position = a[:3, 3] + k * a[:3, 2]
        ds.ImagePositionPatient = [pydicom.valuerep.DSfloat(x, auto_format=True) for x in position]
        ds.ImageOrientationPatient = [pydicom.valuerep.DSfloat(x, auto_format=True) for x in (*u, *v)]
        ds.PixelSpacing = [pydicom.valuerep.DSfloat(row_spacing, auto_format=True),
                           pydicom.valuerep.DSfloat(col_spacing, auto_format=True)]
        ds.SliceThickness = 1.5                    # decoy: not the slice spacing
        ds.Rows, ds.Columns = truth.shape[1], truth.shape[0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = 0, 1, "HU"
        if private:
            ds.add_new(0x00290010, "LO", "SYNTH PRIVATE")
            ds.add_new(0x00291010, "OB", bytes([n, 1, 2, 3]))
        for key, value in (extra or {}).items():
            setattr(ds, key, value)
        ds.PixelData = np.ascontiguousarray(volume[:, :, k].T).tobytes()   # (rows j, columns i)
        path = os.path.join(directory, f"img{n:03d}.dcm")
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths
