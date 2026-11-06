# Geometry and metadata

MedVol's guarantees:

- **Faithful round trips.** X → `.mla` → X reproduces the array bit-identically and the header to the
  accuracy of the format.
- **Exact geometry.** Every voxel keeps its world position across formats, within the format's precision,
  including oblique, mirrored and gantry-tilted (sheared) grids.
- **Nothing implicit.** No reorientation, resampling, rescaling, dtype change or metadata removal unless a
  policy asks for it. MedVol does not de-identify.
- **Everything reported.** See [Reports](reports.md).

These are tested with a conformance phantom: an oblique, mirrored volume in which every voxel encodes its own
file index, written with the reference libraries and read and written through every reader × writer pair.
World positions must agree to < 1e-4 mm.

## Array order

The array keeps the stored order of the file, in C order: the format's fastest-varying axis is last. A
NIfTI, NRRD or MetaImage volume indexed `[i, j, k]` in the file is `array[k, j, i]`, and DICOM slices are
`array[slice, row, column]`. Axis names (`z`, `y`, `x`, `t`, ...) are names only; they say nothing about
anatomy. `source[].index_map` records how array axes relate to the file's own index order.

## Spaces

The geometry is the MLArray space `patient` (`image.space()`):

- its axes are the patient directions: x, y, z increase toward left, posterior, superior (LPS,
  `world_frame="dicom_patient"`, default) or right, anterior, superior (`world_frame="ras"`);
- `space.from_array.matrix` maps array indices of the axes in `space.from_array.array_axes` to millimetres,
  with index = voxel centre;
- non-space axes (time, channel, vector component) never enter the matrix.

Readers build the matrix from the format's own fields (NIfTI qform/sform, NRRD `space directions` and
`space origin`, MetaImage `TransformMatrix`/`Offset`, MGH `Mdc`/`c_ras`, DICOM ImagePositionPatient and
ImageOrientationPatient). Values are never rounded. Going between LPS and RAS changes the signs of matrix rows;
it never flips the array.

```python
space = image.space()
space.matrix()            # (n+1) x (n+1), columns in space.from_array.array_axes order
space.letter_codes()      # e.g. ('R', 'I', 'P'): the direction each mapped axis points to
```

`medvol.reorient(image, "LPS" | "RAS")` changes the array order instead, losslessly ([Policies](policies.md)).

## Values

The array holds the stored values. Scaling (NIfTI `scl_slope`/`scl_inter`, DICOM RescaleSlope/Intercept) goes
to `values.scale` and `values.offset`: physical value = stored × scale + offset. Label maps from DICOM SEG and
RTSTRUCT have `values.kind == "label"` and a label table with names, colours and codes.

## Original headers

Every reader stores the complete original header, readable, in `image.meta.source` (one entry per input):

```python
entry = image.meta.source[0]
entry.format, entry.files          # "nifti", ["ct.nii.gz"]
entry.header["descrip"]            # any header field; DICOM: entry.header["shared"] / ["per_file"]
entry.fingerprint                  # hashes of shape, dtype, geometry and data at read time
```

When an image is written to the **same** format, each header field is handled by its class:

| class | on write | examples |
|---|---|---|
| core-owned | regenerated from the MLArray core | NIfTI `dim`, `pixdim`, qform/sform, `datatype`, `scl_*`; NRRD `sizes`, `space directions`, `kinds`; DICOM `Rows`, `PixelSpacing`, `ImagePositionPatient`, `RescaleSlope` |
| data-dependent | kept while the fingerprint still matches the array and geometry; otherwise recomputed or dropped (reported) | NIfTI `cal_min`/`cal_max`; NRRD `min`/`max`; DICOM `SmallestImagePixelValue`, `SliceLocation`, `SOPInstanceUID` |
| verbatim | written back as stored | descriptions, vendor and acquisition fields, private tags, NRRD custom keys, NIfTI extensions |

When writing a **different** format, only the core is converted. Header fields of one format are never
copied into another format's custom keys; the report says so.
