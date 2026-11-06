# Writing a plugin

A format is a subclass of [`FormatPlugin`](api.md#medvol.plugins.base.FormatPlugin) with a reader and a
writer that translate between the file and an [`Image`](api.md#medvol.Image). MedVol does the rest: format
detection, policies, capability checks, reorientation, history and reports.

```python
from pathlib import Path

import numpy as np
from mlarray import Meta

import medvol
from medvol.plugins.base import Capabilities, FormatPlugin
from medvol.report import Report


class NpyPlugin(FormatPlugin):
    """A bare NumPy array: no geometry, no metadata."""

    name = "npy"
    extensions = (".npy",)
    capabilities = Capabilities(max_space_axes=3, min_space_axes=0, oblique=False, shear=False)

    def sniff(self, path):
        path = Path(path)
        if not path.is_file():
            return 0.0
        with open(path, "rb") as file:
            return 1.0 if file.read(6) == b"\x93NUMPY" else 0.0

    def read(self, path, policy, **options):
        array = np.load(path)
        meta = Meta(axes=[f"axis_{n}" for n in range(array.ndim)], shape=array.shape,
                    source=[{"format": self.name, "files": [Path(path).name]}])
        return [medvol.Image(array, meta)], Report("read", self.name, str(path))

    def write(self, image, path, policy, **options):
        report = Report("write", self.name, str(path))
        if image.meta.spaces:
            report.drop("spaces", "the format has no geometry", policy.on_loss)
        np.save(path, image.array)
        return report


medvol.registry.register(NpyPlugin())
medvol.convert("ct.nii.gz", "ct.npy")        # reports the dropped geometry
image, report = medvol.read("ct.npy")
```

Guidelines, which the built-in plugins follow:

- **Read the stored values in the stored order** (C order, fastest file axis last) and describe them with
  `axes`, `spaces` and `values`. Record the file's index order in `source[].index_map`.
- **Compute geometry yourself** from the raw header fields, with index = voxel centre, and build patient
  spaces with `medvol.geometry.patient_space(matrix_lps, array_axes, policy.world_frame)`.
- **Keep the complete header** in `source[].header`, readable, and use it only when writing the same format.
  Classify its fields with a [`FieldTable`](api.md#medvol.plugins.base.FieldTable): core-owned fields are
  regenerated from the core, data-dependent fields are kept only while `source[].fingerprint` matches.
- **Declare capabilities** honestly; MedVol drops or refuses what they exclude before `write` is called.
  Anything else you cannot write: `report.drop(...)` with `policy.on_loss`.
- **Report every choice** with `report.choose(name, value)` and every doubt with `report.warn(message)`.

## Registering through an entry point

A package registers its plugins in the `medvol.formats` entry point group, so they are found when MedVol is
imported:

```toml
[project.entry-points."medvol.formats"]
npy = "my_package.npy_plugin:NpyPlugin"
```

A built-in plugin of the same name takes precedence. `medvol.registry.register(plugin)` replaces any plugin at
runtime.
