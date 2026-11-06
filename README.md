# MedVol

[![PyPI version](https://img.shields.io/pypi/v/medvol)](https://pypi.org/project/medvol/)
[![Python versions](https://img.shields.io/pypi/pyversions/medvol)](https://pypi.org/project/medvol/)
[![Build status](https://img.shields.io/github/actions/workflow/status/MIC-DKFZ/MedVol/test_and_deploy.yml?branch=main)](https://github.com/MIC-DKFZ/MedVol/actions)
[![License](https://img.shields.io/github/license/MIC-DKFZ/MedVol)](https://github.com/MIC-DKFZ/MedVol/blob/main/LICENSE)

Faithful N:N conversion of medical image formats through one neutral model: the
[MLArray v1](https://github.com/MIC-DKFZ/mlarray) metadata model. Formats: NIfTI-1/2, NRRD, MetaImage, MGH/MGZ,
DICOM (single-frame and enhanced multi-frame series, SEG) and `.mla`.

**Status:** MedVol 1.0 is in development (branch `rework/v2`). Design: [`docs/design/spec.md`](docs/design/spec.md).

## Guarantees

- **Faithful round trips.** X → `.mla` → X reproduces the array bit-identically and the header to the accuracy of
  the format (identical binary floats, identical numeric values for text formats).
- **Nothing implicit.** No reorientation, resampling, rescaling or dtype change, and no metadata removal. The array
  keeps its stored values and order; scaling becomes `values.scale`/`offset`.
- **Exact geometry.** MedVol computes geometry from the format's own fields (backends do byte-level IO only). Every
  voxel keeps its world position across formats, including oblique, mirrored and gantry-tilted (sheared) grids.
- **Everything reported.** Each read, write and convert returns a report of choices, warnings and dropped fields; it
  is also stored in the image's history.

## Usage

```python
import medvol

image, report = medvol.read("ct.nii.gz")              # Image(array, mlarray Meta) + Report
print(report)
medvol.write(image, "ct.mla")
medvol.convert("dicom_dir/", "ct.mla", dicom_tilt="exact")
medvol.convert("ct.mla", "out_dir/", target_format="dicom")   # re-emits the original series
print(medvol.inspect("scan.nrrd"))
```

Ambiguities are resolved by explicit policies (`medvol.Policy`), for example `nifti_form` when qform and sform
disagree, `world_frame` (`dicom_patient` = LPS by default, or `ras`), `dicom_stacks`, `dicom_tilt`,
`dicom_irregular_spacing` and `on_loss` (`warn` or `error` when the target cannot represent something).

The original header is kept, readable, in `image.meta.source`, and is used again only when writing the same format.

## Installation

```bash
pip install medvol            # NIfTI, NRRD, MetaImage, MGH/MGZ, .mla
pip install "medvol[dicom]"   # plus DICOM (pydicom, highdicom)
```

Plugins for further formats can be added through the `medvol.formats` entry point group.

## License

Apache-2.0
