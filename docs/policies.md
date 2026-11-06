# Policies

MedVol does nothing implicitly: no reorientation, resampling, rescaling or dtype change, and no metadata
removal. Where a file can be read in more than one way, a named policy decides, and the choice is written to
the [report](reports.md) and to the image's `history`.

Policies are keyword arguments of `read`, `read_all`, `write`, `convert` and `inspect`, fields of
`medvol.Policy`, and flags of the [CLI](cli.md) (`nifti_form` → `--nifti-form`). The first option is the
default.

| policy | options | applies to |
|---|---|---|
| `nifti_form` | `nibabel`, `sform`, `qform`, `itk` | reading NIfTI |
| `nifti_form_disagreement` | `warn`, `error`, `ignore` | reading NIfTI |
| `nifti_form_tolerance_mm` | a distance in mm (default `0.001`) | reading NIfTI |
| `nifti_no_codes` | `error`, `spec`, `nibabel`, `itk` | reading NIfTI |
| `reorient` | `none`, `LPS`, `RAS` | reading |
| `world_frame` | `dicom_patient`, `ras` | reading |
| `dicom_tilt` | `exact`, `resample`, `error` | reading DICOM |
| `dicom_irregular_spacing` | `error`, `split`, `resample` | reading DICOM |
| `dicom_stacks` | `split`, `largest`, `error` | reading DICOM |
| `dicom_rescale` | `error`, `apply` | reading DICOM |
| `vector_frame` | `from_file`, `lps`, `ras`, `error` | vector images |
| `on_loss` | `warn`, `error` | writing |
| `layout` | `source`, `default` | writing |

## NIfTI

**`nifti_form`**: which of the two NIfTI geometries is used. Readers disagree when qform and sform differ
(by up to tens of mm in real files), so MedVol makes the choice explicit:

- `nibabel`: the sform if `sform_code > 0`, else the qform;
- `sform` / `qform`: that form;
- `itk`: what ITK does.

**`nifti_form_disagreement`**: what happens when qform and sform both exist and differ by more than
`nifti_form_tolerance_mm` at the image corners: a warning, an error, or nothing.

**`nifti_no_codes`**: a file with `qform_code = sform_code = 0` has no defined geometry and tools give three
different answers. `error` refuses it; `spec` uses the standard's fallback (RAS, `diag(pixdim)`); `nibabel` and
`itk` copy those libraries.

## Orientation and world frame

**`reorient`**: `none` keeps the stored array order. `LPS` and `RAS` transpose and flip the array (never
resample) so its space axes, fastest first, point closest to x, y, z of that frame. World positions do not
change. Also available as `medvol.reorient(image, "RAS")`.

**`world_frame`**: the frame of the patient space MedVol builds: `dicom_patient` (LPS, as DICOM and ITK) or
`ras` (as NIfTI and nibabel). Changing it changes the signs of matrix rows, never the array.

## DICOM

**`dicom_tilt`**: gantry-tilted series have slices that are not perpendicular to the slice direction.
`exact` stores the true, sheared matrix (every voxel at its true position); `resample` resamples to an
orthogonal grid; `error` refuses. ITK-based readers place tilted slices wrong by tens of mm.

**`dicom_irregular_spacing`**: slices with uneven gaps raise (`error`), are split into regular stacks
(`split`), or are resampled (`resample`).

**`dicom_stacks`**: a series with several geometries (e.g. localizers) is split into one image per stack
(`split`, use `read_all`), reduced to the largest stack (`largest`), or refused (`error`).

**`dicom_rescale`**: when RescaleSlope/Intercept differ per slice (common in PET), they cannot be one
`values.scale`/`offset`. `error` refuses; `apply` stores the physical values as float. Written back to DICOM,
the stored values are recomputed per slice and must reproduce the array exactly.

## Vectors

**`vector_frame`**: the frame vector components (displacement fields, gradients) are expressed in. `from_file`
uses what the file says (NRRD `measurement frame`, NIfTI intent) and otherwise fails; `lps` / `ras` state it.
ITK writes LPS and NiftyReg RAS under the same NIfTI intent, so a guess would be wrong half the time.

## Writing

**`on_loss`**: what happens when the target format cannot store something (a sheared grid in MetaImage,
`user` metadata in NIfTI, ...): `warn` drops it and reports it, `error` raises `LossError`. Dimensions, axis
types and dtypes the format cannot hold always raise.

**`layout`**: the output encoding. `source` reuses the original encoding, compression and transfer syntax when
writing the same format; `default` uses the writer's default. It never changes which header fields are kept.
