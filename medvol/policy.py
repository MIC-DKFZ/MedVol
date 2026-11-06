"""Conversion policies (spec section 5): every ambiguity is resolved by a named, explicit option."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

OPTIONS = {
    "nifti_form": ("nibabel", "sform", "qform", "itk"),
    "nifti_form_disagreement": ("warn", "error", "ignore"),
    "nifti_no_codes": ("error", "spec", "nibabel", "itk"),
    "reorient": ("none", "LPS", "RAS"),
    "world_frame": ("dicom_patient", "ras"),
    "dicom_tilt": ("exact", "resample", "error"),
    "dicom_irregular_spacing": ("error", "split", "resample"),
    "dicom_stacks": ("split", "largest", "error"),
    "vector_frame": ("from_file", "lps", "ras", "error"),
    "on_loss": ("warn", "error"),
    "layout": ("source", "default"),
}


@dataclass(frozen=True)
class Policy:
    """Options of read, write and convert; the first option of each is the default.

    ``nifti_form_tolerance_mm`` is the distance (in mm, over the image corners)
    above which qform and sform count as disagreeing.
    """

    nifti_form: str = "nibabel"
    nifti_form_disagreement: str = "warn"
    nifti_form_tolerance_mm: float = 1e-3
    nifti_no_codes: str = "error"
    reorient: str = "none"
    world_frame: str = "dicom_patient"
    dicom_tilt: str = "exact"
    dicom_irregular_spacing: str = "error"
    dicom_stacks: str = "split"
    vector_frame: str = "from_file"
    on_loss: str = "warn"
    layout: str = "source"

    def __post_init__(self):
        for name, options in OPTIONS.items():
            value = getattr(self, name)
            if value not in options:
                raise ValueError(f"Policy {name}={value!r} is not one of {options}.")
        if not self.nifti_form_tolerance_mm >= 0:
            raise ValueError("nifti_form_tolerance_mm must be >= 0.")

    @classmethod
    def from_options(cls, policy: "Policy" = None, **options: Any) -> "Policy":
        """A policy from keyword options, optionally on top of an existing policy."""
        known = {f.name for f in fields(cls)}
        unknown = set(options) - known
        if unknown:
            raise TypeError(f"Unknown policy options {sorted(unknown)}; known: {sorted(known)}.")
        base = asdict(policy) if policy is not None else {}
        base.update(options)
        return cls(**base)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
