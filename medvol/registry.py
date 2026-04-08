from __future__ import annotations

from pathlib import Path

from medvol.backends.nibabel_backend import NibabelBackend
from medvol.backends.pynrrd_backend import PynrrdBackend
from medvol.backends.simpleitk_backend import SimpleITKBackend
from medvol.geometry import normalize_backend_name


SUPPORTED_EXTENSIONS = {".nii", ".nii.gz", ".nrrd", ".nhdr"}
DEFAULT_BACKENDS = {
    "nifti": "nibabel",
    "nrrd": "pynrrd",
}
VALID_BACKENDS = {
    "nifti": {"nibabel", "simpleitk"},
    "nrrd": {"pynrrd", "simpleitk"},
}
BACKENDS = {
    "simpleitk": SimpleITKBackend(),
    "nibabel": NibabelBackend(),
    "pynrrd": PynrrdBackend(),
}


def detect_format(filepath: str | Path) -> str:
    path = Path(filepath)
    suffixes = path.suffixes
    if suffixes[-2:] == [".nii", ".gz"] or path.suffix == ".nii":
        return "nifti"
    if path.suffix in {".nrrd", ".nhdr"}:
        return "nrrd"
    raise ValueError(f"Unsupported file extension for '{path}'.")


def resolve_backend(filepath: str | Path, backend: str | None) -> str:
    image_format = detect_format(filepath)
    normalized = normalize_backend_name(backend)
    if normalized is None:
        return DEFAULT_BACKENDS[image_format]
    if normalized not in VALID_BACKENDS[image_format]:
        raise ValueError(
            f"Backend '{normalized}' is not valid for {image_format} files."
        )
    return normalized


def get_backend(name: str):
    return BACKENDS[normalize_backend_name(name)]
