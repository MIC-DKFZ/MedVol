# Formats

| format | extensions | backend | reads | writes |
|---|---|---|---|---|
| NIfTI-1 / NIfTI-2 | `.nii`, `.nii.gz`, `.hdr`/`.img` (also gzipped) | nibabel (header parsing and data IO only) | yes | yes |
| NRRD | `.nrrd`, `.nhdr` + data file | pynrrd (parsing and data IO only) | yes | yes |
| MetaImage | `.mha`, `.mhd` + raw file | MedVol's own parser | yes | yes |
| MGH / MGZ | `.mgh`, `.mgz` | nibabel | yes | yes |
| DICOM | directories, `.dcm` | pydicom, highdicom (`pip install "medvol[dicom]"`) | series, enhanced multi-frame, SEG, RTSTRUCT | original series, enhanced files, SEG |
| MLArray | `.mla` | mlarray | yes | yes |

`medvol formats` lists the installed plugins, including those of other packages
([Writing a plugin](plugins.md)).

Backends are used for byte-level IO. MedVol computes the geometry itself from the format's raw header fields,
because libraries disagree on several of them (the cases are named below).

## Capabilities

Each writer declares what it can store. Before writing, MedVol compares the image with these
capabilities. What the format cannot store is reported as dropped (`on_loss="warn"`, default) or raises
`LossError` (`on_loss="error"`). Dimensions, axis types and dtypes the format cannot hold always raise.

| | NIfTI | NRRD | MetaImage | MGH/MGZ | DICOM | `.mla` |
|---|---|---|---|---|---|---|
| space axes | 1-3 | 0-3 | 1-3 | 3 | 2-3 | any |
| array axes (max) | 7 | 16 | 11 | 4 | 4 | 32 |
| non-space axes | time, channel, component, other | time, channel, component, other | time, channel, component, other | time, channel, component, other | time, channel, other | all |
| oblique grids | yes | yes | yes | yes | yes | yes |
| sheared grids | yes (sform) | yes | no (closest orthonormal direction, reported) | yes | yes (tilted series) | yes |
| label table | no | no | no | no | yes (SEG) | yes |
| arbitrary metadata (`user`, profiles) | no | no | no | no | no | yes |
| dtypes | int8-64, uint8-64, float32/64, complex64/128 | int8-64, uint8-64, float32/64 | int8-64, uint8-64, float32/64 | uint8, int16, uint16, int32, float32 | any | any |

## NIfTI-1 / NIfTI-2

- MedVol chooses between qform and sform itself (`nifti_form`); readers disagree by up to tens of mm when
  the two differ. Files without codes are an error by default (`nifti_no_codes`).
- On write, both forms are regenerated from the core and the original codes are kept while the geometry is
  unchanged. nibabel's affine/header synchronisation is bypassed, because it rewrites qform, sform and codes
  on save. For a sheared grid, the sform is exact and the qform is the closest rigid approximation (reported).
- `scl_slope`/`scl_inter` become `values.scale`/`offset`; the array keeps the stored values.
- Extensions are kept verbatim, binary included.
- Time uses `pixdim[4]` and `xyzt_units`; vector intents become a `component` axis with an explicit
  `vector_frame`.
- NIfTI-2 is supported.

## NRRD

- MedVol interprets `space` itself; pynrrd is only used with its default `index_order='F'`.
- All 3-D and 4-D spaces are supported, including `left-anterior-superior`. `scanner-xyz` and
  `3D-right-handed`/`3D-left-handed` are read as LPS (as ITK and Slicer do), with a warning. A
  `space dimension` without `space` becomes a non-patient space named `nrrd`.
- `measurement frame` is kept and applied to vector components explicitly.
- `kinds`, `centerings`, custom key-value pairs and comments are kept verbatim. Non-world axes sampled with
  `spacings`/`axis mins`/`axis maxs` get their start and step at voxel centres, following `centerings` (cell when
  absent, as in NRRD and ITK).
- Time series from NIfTI are written in `left-posterior-superior-time` space, the 4-D space ITK reads
  correctly.

## MetaImage

- `TransformMatrix` is column-major, one axis per triple. `Origin` wins over `Offset` and `Position`, as in ITK.
- `AnatomicalOrientation` is kept verbatim but never used for geometry: ITK writes it in "comes from"
  convention and never reads it.
- MetaImage stores orthonormal directions only: a sheared grid is written with the closest orthonormal
  direction, and this is reported.

## MGH / MGZ

- `goodRASFlag` <= 0 is read as FreeSurfer does (coronal LIA, 1 mm); nibabel reads it differently.
- `c_ras` is the centre at index N/2, as FreeSurfer defines it.

## DICOM

Reading:

1. group by SeriesInstanceUID, then by geometry (orientation, size, spacing), per `dicom_stacks`;
2. sort slices by the projection of ImagePositionPatient on the slice normal, never by InstanceNumber;
3. check spacing regularity (`dicom_irregular_spacing`) and gantry tilt (`dicom_tilt`).

- Enhanced multi-frame files (also 4-D) use the per-frame functional groups, where some vendors put geometry
  per frame.
- DICOM SEG becomes a label map with codes in `values.labels`; frames are placed by position, not by order.
  Fractional SEG becomes `values.kind: probability`.
- RTSTRUCT contours are rasterised onto the referenced series.
- The complete header, private tags included, goes to `source.header` (`shared`, `per_file`, `file_names`).
- Siemens mosaic is not supported (it needs a dcm2niix backend).

Writing:

- **Same series:** the original files are re-emitted from `source`, with the original UIDs. Pixel data comes
  from the array, in the original transfer syntax when it is lossless, otherwise explicit VR little endian
  (reported). If the data or geometry changed, the data-dependent elements are dropped and new SOP Instance
  UIDs are generated.
- **New DICOM:** only DICOM SEG, from a label map and its referenced series ([Usage](usage.md#dicom)).
  New image series from other formats are not created.

## MLArray (`.mla`)

Read and written by `mlarray`; nothing is converted. `.mla` stores everything MedVol reads, including the
original header in `source`, so X → `.mla` → X is lossless.
