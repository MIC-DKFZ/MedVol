# MedVol v2 — N:N image converter specification (draft)

Status: **draft for discussion, not implemented.** Branch `rework/v2`. Last update: 2026-11-06.
Companion spec: MLArray v1 format and metadata (`mlarray` repo, branch `rework/v1`, `docs/design/spec.md`).
Evidence base for every behaviour claim about other libraries: `~/projects/image-orientation-kb` ("KB").

---

## 1. Purpose

MedVol v2 reads and writes image formats **faithfully** and converts between them through one neutral
in-memory model: the MLArray v1 model. It replaces MedVol v1, which canonicalised to RAS on load and handled
metadata loosely. There is no compatibility with v1 behaviour.

- `mlarray` stays IO-free (numpy + blosc2). MedVol depends on `mlarray` for the model and the `.mla`
  reader/writer.
- `mlarray_convert` delegates to MedVol when it is installed.
- v2 targets medical formats: NIfTI-1/2, NRRD, MetaImage, MGH/MGZ, DICOM and `.mla`. The plugin design
  allows other domains later: TIFF/OME-TIFF, FITS, GeoTIFF.

## 2. Guarantees

1. **Faithful same-format round trip.** X → `.mla` → X reproduces the array bit-identically and the
   metadata to the MLArray accuracy standard (MLArray spec §9).
2. **Nothing implicit.**
   - No reorientation, resampling, rescaling or dtype change unless requested.
   - No metadata removal or modification (no de-identification).
   - Every ambiguity is resolved by a named policy (§5), never by whatever a backend library happens to do.
3. **Geometry is exact.** World positions of every voxel agree between the source file, the `.mla` and any
   written file, within the format's own precision.
4. **Everything is reported.** Each read, write and convert produces a report (choices, warnings, dropped
   fields). It is printed by the CLI and stored in `.mla` `history`.

## 3. Architecture: hub and spoke

```
  NIfTI ─► nifti reader ─┐                                   ┌─► nifti writer ─► NIfTI
  NRRD  ─► nrrd reader  ─┤                                   ├─► nrrd writer  ─► NRRD
  MHA   ─► mha reader   ─┼─► Image(array, Meta) + Report ────┼─► mha writer   ─► MHA
  MGZ   ─► mgh reader   ─┤        (MLArray v1 model)         ├─► mgh writer   ─► MGZ
  DICOM ─► dicom reader ─┤                                   ├─► dicom writer ─► DICOM (§7.6)
  .mla  ─► mla reader   ─┘                                   └─► mla writer   ─► .mla
```

N readers + N writers, never N² pairwise converters.

### 3.1 `Image`

```python
@dataclass
class Image:
    array: np.ndarray | LazyArray   # stored values, original dtype, array order as described by meta.axes
    meta: mlarray.Meta              # core + profiles + source + history (+ user)
```

The same `Image` is what `mlarray` reads and writes, so `.mla` is just another plugin.

### 3.2 Plugin interface

```python
class FormatPlugin:
    name: str                         # "nifti"
    extensions: tuple[str, ...]       # (".nii", ".nii.gz")
    capabilities: Capabilities        # §3.3
    fields: FieldTable                # §4: core-owned / data-dependent / verbatim, per header field

    def sniff(self, path) -> float: ...                       # confidence 0..1 from content, not only extension
    def read(self, path, policy: Policy) -> tuple[Image, Report]: ...
    def write(self, image: Image, path, policy: Policy) -> Report: ...
```

Plugins are registered via entry points, so domain plugins (`medvol-fits`, …) can live in separate
packages.

### 3.3 Capabilities

Each writer declares what it can represent:
- dimensionality, and non-space axes (time, channel, component);
- oblique, mirrored or sheared geometry;
- vector and tensor components and their frame;
- label tables;
- arbitrary key-value metadata;
- dtypes.

Before writing, MedVol compares the `Image` against the capabilities. Anything that cannot be represented
either raises (policy `on_loss: error`) or is reported and dropped (`warn`, default). Examples:
- a sheared matrix to a NIfTI qform-only reader;
- a direction matrix to `.vtk`/`.tif` (KB ch04 §13);
- a NIfTI `descrip` into NRRD (kept as a key-value pair, reported).

## 4. Field ownership (how `source` stays in sync)

For each format, the plugin's field table classifies every header field:

| class | meaning on export | examples |
|---|---|---|
| **core-owned** | always regenerated from the MLArray core | NIfTI `dim`, `pixdim`, `qform*`, `sform*`, `srow_*`, `quatern_*`, `datatype`, `scl_slope/inter`, `xyzt_units`; NRRD `sizes`, `space`, `space directions`, `space origin`, `type`, `kinds`; DICOM `Rows`, `Columns`, `PixelSpacing`, `ImagePositionPatient`, `ImageOrientationPatient`, `RescaleSlope/Intercept`, `BitsAllocated`, … |
| **data-dependent** | kept verbatim if the source `fingerprint` still matches; otherwise recomputed (if defined) or dropped, and reported | NIfTI `cal_min/cal_max`, `glmin/glmax`; NRRD `min/max`; DICOM `SmallestImagePixelValue`, `LargestImagePixelValue`, `SliceLocation`, `InstanceNumber`, per-file `SOPInstanceUID` |
| **verbatim** | written back exactly as stored | everything else: descriptions, vendor and acquisition fields, private tags, NRRD custom keys, NIfTI extensions |

**Round-trip test:** for every format and every test file, regenerating the core-owned fields from an
unchanged core must reproduce the original values within the accuracy standard. This proves the
core-owned/verbatim split loses nothing.

## 5. Policies

All are explicit options of `read`, `write` and `convert`, with these defaults. Each choice made is written
to the report.

| policy | options | default | reason (KB) |
|---|---|---|---|
| `nifti_form` | `sform`, `qform`, `nibabel` (sform if `sform_code > 0`, else qform), `itk` | `nibabel` | readers disagree by up to tens of mm when qform ≠ sform (ch03 §5) |
| `nifti_form_disagreement` | `warn`, `error`, `ignore` | `warn` (tolerance 1e-3 mm) | |
| `nifti_no_codes` | `spec` (RAS `diag(pixdim)`), `nibabel`, `itk`, `error` | `error` | three tools give three answers (ch03 §0 #10) |
| `reorient` | `none`, closest `LPS`, closest `RAS` (lossless permutation and flip) | `none` | closest codes are library- and version-dependent for oblique data (ch01 §5.3) |
| `world_frame` (output space axes) | `dicom_patient` (LPS), `ras` | `dicom_patient` | DICOM/ITK native; RAS remains expressible |
| `dicom_tilt` | `exact` (sheared matrix), `resample`, `error` | `exact` | ITK readers silently misplace tilted slices by > 40 mm (ch02, real data) |
| `dicom_irregular_spacing` | `error`, `split`, `resample` | `error` | dcm2niix plain output 2 mm off; GDCM fallbacks (ch02) |
| `dicom_stacks` | `split` (one Image per geometric stack), `largest`, `error` | `split` | localizers in a series: 53 mm error in SimpleITK (ch02) |
| `vector_frame` | `from_file`, `lps`, `ras`, `error` | `from_file`, else `error` | ITK writes LPS and NiftyReg RAS under the same intent (ch03 §8) |
| `on_loss` | `warn`, `error` | `warn` | |
| `layout` (output encoding) | `source` (reuse the original encoding/compression/transfer syntax), `default` | `source` | |

## 6. Geometry handling

- Readers build the MLArray `spaces[].from_array.matrix` directly from the format's native fields. The
  conversion between the format's world frame and the output frame (`world_frame` policy) is a sign change
  of matrix rows, never an array flip.
- Array order:
  - Readers keep the source's storage order: C order, last axis = the format's fastest-varying axis, so
    `[k, j, i]`.
  - `axes` names are `z, y, x` (or `t, z, y, x`, …) as **names only**.
  - `index_map` records the relation to the source's own index order.
- Integer index = voxel centre in every reader and writer. Formats with cell or corner semantics, such as
  NRRD `centerings: cell` with old-style `axis mins`, are converted explicitly.
- No backend library's geometry interpretation is trusted blindly. Backends are used for byte-level IO, and
  MedVol computes geometry itself from the raw fields, following the specs and the KB findings.

## 7. Formats

### 7.1 NIfTI-1 / NIfTI-2 (`.nii`, `.nii.gz`, `.hdr/.img`)

- Backend: **nibabel**, used for header parsing and data IO only.
- MedVol applies `nifti_form` itself. nibabel's affine-to-header synchronisation is bypassed on write,
  because nibabel rewrites qform/sform and codes on save (KB ch05, ch07).
- Writing:
  - both forms are regenerated from the core;
  - original codes are kept when the geometry is unchanged;
  - if the matrix has shear, the qform is written as the closest rigid approximation with
    `qform_code` = original or 0, and this is reported (KB ch03 §2).
- Extensions are kept verbatim (binary allowed).
- 4-D/5-D: time uses `pixdim[4]` and `xyzt_units`; vector intents map to a `component` axis with an
  explicit `vector_frame`.
- NIfTI-2 is supported (ITK cannot read it, KB ch03 §10, so MedVol must not depend on ITK for NIfTI).

### 7.2 NRRD (`.nrrd`, `.nhdr` + data)

- Backend: **pynrrd** for parsing and data IO, with its `index_order='F'` default only. MedVol interprets
  `space` itself.
- All 3-D/4-D spaces are supported, including `left-anterior-superior`, `scanner-xyz` and
  `3D-right-handed`. ITK ≤ 5.4.6 and MONAI misread several of these (KB ch04).
- `measurement frame` is kept and applied to vector components explicitly (pynrrd transposes it, KB ch04).
- `kinds`, `centerings`, custom key-value pairs and comments are kept verbatim.

### 7.3 MetaImage (`.mha`, `.mhd` + raw)

- Backend: a small MedVol parser, since the format is simple text.
- `TransformMatrix` is column-major, one axis per triple (KB ch04 §3).
- Precedence among `Origin`/`Offset`/`Position` follows ITK: `Origin` wins.
- `AnatomicalOrientation` is kept verbatim but never used for geometry: ITK writes it in comes-from
  convention and never reads it.

### 7.4 MGH/MGZ

- Backend: **nibabel**.
- `goodRASFlag = 0` is handled as FreeSurfer does (LIA, 1 mm). nibabel 5.4.2 reads it differently, so
  MedVol must not take nibabel's affine there (KB `docs/verification/freesurfer-afni.md`).
- `c_ras` is stored at index N/2, as FreeSurfer defines it.

### 7.5 DICOM — reading

- Backends: **pydicom** for parsing and pixel decoding, **highdicom** where it is exact (enhanced
  multi-frame, SEG). MedVol's own geometry builder does the rest, and **dcm2niix** is an optional backend
  and a test oracle.
- Series assembly:
  1. group by SeriesInstanceUID and then by geometry (orientation, size, spacing), per `dicom_stacks`;
  2. sort by the projection of ImagePositionPatient on the slice normal, never by InstanceNumber;
  3. validate spacing regularity (`dicom_irregular_spacing`) and tilt (`dicom_tilt`).
- `source.header`: `shared` + `per_file` + `file_names`, as in the MLArray spec §6.4, including private
  tags.
- Phases:
  1. single-frame CT/MR/PET series, regular or tilted;
  2. enhanced multi-frame, including 4-D (per-frame functional groups; Philips and Siemens XA put geometry
     per frame, KB);
  3. DICOM SEG ↔ label map with codes (`values.labels.code`), frames placed by position, not by order;
  4. RTSTRUCT rasterisation with an explicit rasterisation rule (tools differ by 2–13 %, KB);
  5. Siemens mosaic via the dcm2niix backend.

### 7.6 DICOM — writing

The guarantee in §2 implies DICOM → `.mla` → DICOM. Proposed scope:
- **Same-series round trip:** re-emit the original files. Every per-file header comes from `source`
  (`shared` + `per_file`), with the original UIDs. Pixel data comes from the array, written in the original
  transfer syntax when it is lossless, otherwise explicit VR little endian (reported).
- **New DICOM from non-DICOM sources:** only DICOM SEG (via highdicom). Creating new image series from
  NIfTI is out of scope for v2.

### 7.7 `.mla`

Via `mlarray`; no conversion needed.

## 8. API and CLI

```python
import medvol
img, report = medvol.read("ct.nii.gz", nifti_form="sform")
medvol.write(img, "ct.mla")
report = medvol.convert("dicom_dir/", "ct.mla", dicom_tilt="exact")
medvol.inspect("ct.nii.gz")            # geometry + metadata summary in the MLArray vocabulary
```

```
medvol convert INPUT OUTPUT [--nifti-form sform] [--reorient none] [--on-loss error] ...
medvol inspect INPUT                    # prints axes, spaces, values, warnings
```

## 9. Testing

1. **Conformance phantom:**
   - a synthetic, deliberately asymmetric, oblique volume in which each voxel encodes its own world
     position;
   - generated per format with every relevant header variant (qform-only, sform-only, disagreeing forms,
     all NRRD spaces, MetaImage matrix variants, mirrored grids, shear, 4-D, vectors).
   - Every reader × writer pair must preserve world positions to < 1e-4 mm. Ported from the KB experiments.
2. **Round-trip suite:**
   - X → `.mla` → X for every test file, comparing array bits and header fields under the accuracy
     standard;
   - plus the core-owned regeneration test (§4).
3. **Real data (optional CI job):** dcm_qa repositories (tilt, mosaic, enhanced, SEG), with dcm2niix as the
   oracle where the KB found it exact.
4. **Library-bug regressions:** NRRD LAS, NIfTI qform/sform precedence, nibabel MGH `goodRASFlag`, vector
   intent frames, SEG frame order.

## 10. Decisions log

| topic | decision |
|---|---|
| DICOM writing | re-emit original series from `source` (original UIDs) + create DICOM SEG from label maps; no new image series from non-DICOM sources |
| very large inputs | in-memory `Image.array` only in v2; lazy/streaming IO (whole-slide images, huge series) is a later version |
| non-medical formats | later; the MLArray model already covers their metadata (MLArray spec §5.3), so they can be added as plugins without model changes |
| pyramids | not read or written in v2 (MLArray reserves level geometry only) |
| de-identification | never; metadata is converted as is |
| default output world frame | DICOM patient (LPS) |

## 11. Open questions

None at the moment.
