# Usage

MedVol reads a file into an [`Image`](api.md#medvol.Image): the stored array plus its metadata in the
[MLArray v1](https://mic-dkfz.github.io/mlarray/) model. Every writer takes that same `Image`, so any reader
can be combined with any writer.

```bash
pip install medvol            # NIfTI, NRRD, MetaImage, MGH/MGZ, .mla
pip install "medvol[dicom]"   # plus DICOM (pydicom, highdicom)
```

## Read

```python
import medvol

image, report = medvol.read("ct.nii.gz")
print(report)
```

```text
read nifti ct.nii.gz
  choice: nifti_form = sform (code 1)
```

`image.array` is a NumPy array with the stored values, in the stored dtype and the stored order:
C order with the format's fastest-varying axis last, so a NIfTI volume `[i, j, k]` becomes `array[k, j, i]`.
`image.meta` is an `mlarray.Meta`:

```python
image.meta.axes                      # z, y, x: names of the array axes, in array order
space = image.space()                # the first space, here "patient" (DICOM patient frame, LPS, mm)
space.matrix()                       # 4x4 from-array matrix; columns follow space.from_array.array_axes
space.spacing, space.origin          # per from_array axis (here x, y, z = NIfTI i, j, k)
space.letter_codes()                 # ('R', 'I', 'P'): where each from_array axis points
image.meta.values                    # kind (measurement, label, ...), unit, scale/offset, labels
image.meta.source[0].header          # the complete original header, readable
image.meta.history                   # one entry per read/convert, with the report's choices
```

The format is detected from the file content. Pass `format="nrrd"` (or `"dicom"`, ...) if it cannot be.

### Several images in one input

A DICOM directory can hold several series or several geometric stacks. `read` raises in that case;
`read_all` returns all of them:

```python
images, report = medvol.read_all("dicom_dir/")             # one Image per stack (dicom_stacks="split")
image, report = medvol.read("dicom_dir/", dicom_stacks="largest")
```

## Write

```python
report = medvol.write(image, "ct.mla")      # the format comes from the extension
report = medvol.write(image, "ct.nrrd")
report = medvol.write(image, "out_dir/", format="dicom")
```

Before writing, MedVol compares the image with what the target format can store (its
[capabilities](formats.md#capabilities)). What it cannot store is listed in the report as dropped, or raises
[`LossError`](api.md#medvol.LossError) with `on_loss="error"`:

```python
medvol.write(image, "ct.mha", on_loss="error")
```

## Convert

`convert` is `read` followed by `write`; the report covers both steps.

```python
report = medvol.convert("ct.mla", "ct.nrrd")
print(report)
```

```text
convert mla -> nrrd ct.mla -> ct.nrrd
  choice: nrrd_space = left-posterior-superior
  choice: layout = encoding gzip, little endian
  warning: source header (nifti): its fields are not written to nrrd; only the core is converted (spec 6, 9)
```

The original NIfTI header travels inside the `.mla` and is used again only when the image is written back
to NIfTI (see [Geometry and metadata](geometry.md#original-headers)).

## Policies

Every ambiguity is resolved by a named [policy](policies.md). Pass them as keyword arguments, or as a
reusable `medvol.Policy`:

```python
image, report = medvol.read("ct.nii.gz", nifti_form="sform", world_frame="ras")

policy = medvol.Policy(dicom_tilt="exact", on_loss="error")
medvol.convert("dicom_dir/", "ct.nii.gz", policy=policy)
medvol.convert("dicom_dir/", "ct.nii.gz", policy=policy, dicom_stacks="largest")   # override one option
```

## Reorient

MedVol never reorients unless asked. `reorient` transposes and flips the array (never resamples) so that its
space axes, fastest first, point closest to x, y, z of LPS or RAS. Every voxel keeps its world position, and
axis names travel with their axes:

```python
ras = medvol.reorient(image, "RAS")
image, report = medvol.read("ct.nii.gz", reorient="RAS")   # the same, while reading
```

## MLArray

`Image` and `mlarray.MLArray` convert into each other:

```python
from mlarray import MLArray

array = image.to_mlarray(patch_size=(128, 128, 128))   # in-memory MLArray; layout options as for MLArray
array.save("ct.mla")
image = medvol.Image.from_mlarray(MLArray("ct.mla"))
```

`medvol.write(image, "ct.mla")` does the same with MLArray's default layout. MLArray's `mlarray_convert` command
uses MedVol.

## DICOM

DICOM support needs `pip install "medvol[dicom]"`.

- **Series** (single-frame CT, MR, PET, ...): slices are grouped by series and geometry and sorted by their
  position along the slice normal, never by InstanceNumber. Tilted (gantry tilt) series get a sheared matrix
  (`dicom_tilt="exact"`), irregular spacing raises unless `dicom_irregular_spacing` says otherwise.
- **Enhanced multi-frame**, including 4-D, with per-frame geometry.
- **Rescale**: slope and intercept become `values.scale` / `values.offset` when every slice shares them.
  When they differ per slice (PET), reading fails unless `dicom_rescale="apply"`, which stores the physical
  values as float.
- **Writing** re-emits the original series from `source`, with the original UIDs. New DICOM image series from
  other formats are not created.
- **SEG** files read as label maps with their segment names, colours and codes (`values.labels`); fractional
  SEGs read as probability maps. A label map (`values.kind` `label` or `probability`) is written as a DICOM SEG
  on its referenced image series:

  ```python
  seg, _ = medvol.read("labels.nii.gz")
  seg.meta.values = {"kind": "label", "labels": {"1": {"name": "liver"}, "2": {"name": "tumor"}}}
  medvol.write(seg, "seg.dcm", format="dicom", reference="ct_dir/")   # a directory, file list or datasets
  ```

  NIfTI has no label table, so an image read from NIfTI is a `measurement` until you say otherwise, as above.
- **RTSTRUCT** contours are rasterised onto the referenced series: read the RTSTRUCT together with its series
  (same directory), or pass `reference="ct_dir/"` to `read`.
