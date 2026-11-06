"""read, write, convert and inspect (spec section 8)."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from medvol import registry
from medvol.image import Image
from medvol.plugins.base import Capabilities, PathLike
from medvol.policy import Policy
from medvol.report import Report


def _tool() -> str:
    from medvol import __version__
    return f"medvol {__version__}"


def read_all(path: PathLike, format: Optional[str] = None, policy: Optional[Policy] = None,
             reference=None, **options) -> Tuple[List[Image], Report]:
    """Every image in ``path`` (a DICOM directory can hold several stacks) and the report.

    ``reference``: for a DICOM RTSTRUCT, the referenced image series it is rasterised onto.
    """
    policy = Policy.from_options(policy, **options)
    plugin = registry.for_reading(path, format)
    extra = {"reference": reference} if reference is not None else {}
    images, report = plugin.read(path, policy, **extra)
    if policy.reorient != "none":
        from medvol.reorient import reorient

        images = [reorient(image, policy.reorient, report) for image in images]
    if plugin.name != "mla":
        for image in images:
            image.meta.add_history(**report.history_entry(_tool()))
    return images, report


def read(path: PathLike, format: Optional[str] = None, policy: Optional[Policy] = None,
         reference=None, **options) -> Tuple[Image, Report]:
    """The image in ``path`` and the report. Raises if ``path`` holds several images (use ``read_all``)."""
    images, report = read_all(path, format, policy, reference=reference, **options)
    if len(images) != 1:
        raise ValueError(f"{path} holds {len(images)} images; use medvol.read_all() or policy dicom_stacks='largest'.")
    return images[0], report


def check_capabilities(image: Image, capabilities: Capabilities, report: Report, policy: Policy) -> None:
    """Report (or raise for, with ``on_loss='error'``) what the target format cannot represent."""
    from medvol.geometry import has_shear, is_oblique

    axes = image.meta.axes
    space_axes = [axis for axis in axes if axis.type == "space"]
    if len(axes) > capabilities.max_ndim:
        report.drop("dimensions", f"{len(axes)} axes, the format holds at most {capabilities.max_ndim}", "error")
    if not capabilities.min_space_axes <= len(space_axes) <= capabilities.max_space_axes:
        report.drop("space axes", f"{len(space_axes)} space axes, the format holds "
                    f"{capabilities.min_space_axes}..{capabilities.max_space_axes}", "error")
    for axis in axes:
        if axis.type != "space" and axis.type not in capabilities.non_space_axes:
            report.drop(f"axis {axis.name!r}", f"the format has no {axis.type} axes", "error")
    if capabilities.dtypes is not None and image.array.dtype.name not in capabilities.dtypes:
        report.drop("dtype", f"{image.array.dtype.name} is not supported", "error")
    space = image.space()
    if space is not None:
        linear = space.matrix()[:-1, :-1]
        if not capabilities.shear and linear.shape[0] == linear.shape[1] and has_shear(linear):
            report.drop("geometry", "sheared grid; the format stores only orthogonal axes", policy.on_loss)
        if not capabilities.oblique and is_oblique(linear):
            report.drop("geometry", "oblique grid; the format stores only axis-aligned grids", policy.on_loss)
    if image.meta.values.labels and not capabilities.label_table:
        report.drop("values.labels", "the format has no label table", policy.on_loss)
    if not capabilities.key_value:
        if image.meta.user:
            report.drop("user", "the format has no place for arbitrary metadata", policy.on_loss)
        for name in image.meta.profiles:
            report.drop(f"profiles.{name}", "the format has no place for profile metadata", policy.on_loss)


def write(image: Image, path: PathLike, format: Optional[str] = None, policy: Optional[Policy] = None,
          reference=None, **options) -> Report:
    """Write ``image`` to ``path``; the format comes from ``format`` or the extension.

    ``reference``: for a DICOM SEG, the referenced image series (directory, files or datasets).
    """
    policy = Policy.from_options(policy, **options)
    plugin = registry.for_writing(path, format)
    report = Report("write", plugin.name, str(path))
    check_capabilities(image, plugin.capabilities, report, policy)
    if plugin.name != "mla":
        for entry in image.meta.source:
            if entry.format != plugin.name:
                report.warn(f"source header ({entry.format}): its fields are not written to {plugin.name}; "
                            "only the core is converted (spec 6, 9)")
    extra = {"reference": reference} if reference is not None else {}
    report.extend(plugin.write(image, path, policy, **extra))
    return report


def convert(source: PathLike, target: PathLike, source_format: Optional[str] = None, target_format: Optional[str] = None,
            policy: Optional[Policy] = None, **options) -> Report:
    """Read ``source`` and write it to ``target``; the report covers both steps."""
    policy = Policy.from_options(policy, **options)
    image, read_report = read(source, source_format, policy)
    write_report = write(image, target, target_format, policy)
    report = Report("convert", f"{read_report.format} -> {write_report.format}", f"{source} -> {target}")
    report.extend(read_report)
    report.extend(write_report)
    return report


def inspect(path: PathLike, format: Optional[str] = None, policy: Optional[Policy] = None, **options) -> str:
    """A summary of the image(s) in ``path`` in the MLArray vocabulary."""
    images, report = read_all(path, format, policy, **options)
    lines = [f"{Path(path)}: {report.format}, {len(images)} image(s)"]
    for n, image in enumerate(images):
        meta = image.meta
        lines.append(f"[{n}] shape {tuple(image.array.shape)} {image.array.dtype}, values {meta.values.kind}"
                     + (f" ({meta.values.unit})" if meta.values.unit else ""))
        lines.append("    axes: " + ", ".join(f"{a.name}:{a.type}" for a in meta.axes))
        for space in meta.spaces:
            np.set_printoptions(precision=4, suppress=True)
            lines.append(f"    space {space.name}: spacing {tuple(round(v, 6) for v in space.spacing)}, "
                         f"origin {tuple(round(v, 4) for v in space.origin)}, letters {''.join(space.letter_codes())}")
    lines += [f"warning: {w}" for w in report.warnings]
    return "\n".join(lines)
