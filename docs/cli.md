# CLI

Installing MedVol adds the `medvol` command.

## `medvol convert`

Convert a file or DICOM directory and print the report.

```bash
medvol convert ct.nii.gz ct.mla
medvol convert dicom_dir/ ct.nii.gz --dicom-stacks largest --world-frame ras
medvol convert ct.mla ct.mha --on-loss error          # fail instead of dropping what MetaImage cannot store
medvol convert ct.mla out_dir/ --target-format dicom  # re-emit the original DICOM series
```

| option | meaning |
|---|---|
| `--source-format NAME` | input format, if it cannot be detected |
| `--target-format NAME` | output format, if the extension does not say (e.g. `dicom`) |
| `--nifti-form`, `--nifti-form-disagreement`, `--nifti-form-tolerance-mm`, `--nifti-no-codes` | NIfTI [policies](policies.md#nifti) |
| `--reorient`, `--world-frame` | [orientation](policies.md#orientation-and-world-frame) |
| `--dicom-tilt`, `--dicom-irregular-spacing`, `--dicom-stacks`, `--dicom-rescale` | [DICOM policies](policies.md#dicom) |
| `--vector-frame` | [vector frame](policies.md#vectors) |
| `--on-loss`, `--layout` | [writing](policies.md#writing) |

Writing a DICOM SEG needs the referenced series and is done from Python
([Usage](usage.md#dicom)).

## `medvol inspect`

Print a summary of a file or DICOM directory in the MLArray vocabulary: shape, dtype, values, axes, spaces and
warnings. It takes `--format` and the same policy options.

```bash
medvol inspect ct.nii.gz
```

```text
ct.nii.gz: nifti, 1 image(s)
[0] shape (7, 6, 5) int32, values measurement
    axes: z:space, y:space, x:space
    space patient: spacing (0.5, 1.0, 2.0), origin (12.5, -30.25, 47.0), letters RIP
```

## `medvol formats`

List the installed format plugins and their extensions.

## `mlarray_convert`

MLArray's `mlarray_convert INPUT OUTPUT` uses MedVol with the same policy options. `mlarray_header` prints the
metadata of a `.mla` file ([MLArray CLI](https://mic-dkfz.github.io/mlarray/cli/)).
