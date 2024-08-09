# MedVol

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/medvol.svg?color=green)](https://github.com/Karol-G/medvol/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/medvol.svg?color=green)](https://pypi.org/project/medvol)
[![Python Version](https://img.shields.io/pypi/pyversions/medvol.svg?color=green)](https://python.org)

A wrapper for loading medical 3D image volumes such as NIFTI or NRRD images.

## Installation

You can install `medvol` via [pip](https://pypi.org/project/medvol/):

    pip install medvol

## Example

```python
from medvol import MedVol

# Load NIFTI image
image = MedVol("path/to/image.nifti")

# Print some metadata
print("Spacing: ", image.spacing)
print("Affine: ", image.affine)
print("Header: ", image.header)

# Access and modify the image array
arr = image.array
arr[0, 0, 0] = 1

# Create a new image with the new array, a new spacing, but copy all remaining metadata
new_image = MedVol(arr, spacing=[2, 2, 2], copy=image)

# Save the new image as NRRD
new_image.save("path/to/new_image.nrrd")
```


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"medvol" is free and open source software

## Issues

If you encounter any problems, please file an issue along with a detailed description.

[Cookiecutter]: https://github.com/audreyr/cookiecutter
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
