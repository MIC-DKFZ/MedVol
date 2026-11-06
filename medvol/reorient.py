"""Lossless reorientation to the closest LPS or RAS array layout (policy ``reorient``, spec 5).

Only the order and direction of the array's space axes change: the array is transposed and flipped,
never resampled, and the from-array matrices change so every voxel keeps its world position. After
``reorient(image, "LPS")`` the space axes of the array, from the fastest (last in C order) to the
slowest, increase toward left, posterior and superior (``"RAS"``: right, anterior, superior).

Axis names travel with their axes. Source ``index_map`` entries are composed with the change; the
source fingerprint is left as it was, so writers regenerate the geometry fields.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from mlarray import Meta

from medvol.geometry import LPS_TO_RAS, GeometryError, closest_axes, lps_matrix
from medvol.image import Image
from medvol.report import Report


def reorient(image: Image, target: str, report: Report = None) -> Image:
    """``image`` reoriented to the closest ``"LPS"`` or ``"RAS"`` layout (a new Image)."""
    if target not in ("LPS", "RAS"):
        raise ValueError(f"reorient target must be 'LPS' or 'RAS', got {target!r}")
    report = report if report is not None else Report("reorient", target)
    space = image.space()
    if space is None:
        report.warn("reorient: the image has no space; left as it is")
        return image
    names = [axis.name for axis in image.meta.axes]
    mapped = list(space.from_array.array_axes)
    if len(mapped) != 3:
        report.warn(f"reorient: the space maps {len(mapped)} array axes, not 3; left as it is")
        return image
    linear = lps_matrix(space)[:3, :3]
    if target == "RAS":
        linear = LPS_TO_RAS @ linear
    world_axes, signs = closest_axes(linear)
    if sorted(world_axes) != [0, 1, 2]:
        raise GeometryError(f"reorient: the closest world axes {world_axes} are ambiguous (oblique by about 45 degrees)")

    positions = [names.index(name) for name in mapped]          # array positions of the mapped columns
    slots = sorted(positions)                                    # slots[2] is the fastest space axis
    perm = list(range(len(names)))
    flips = []
    for world in range(3):
        column = world_axes.index(world)
        slot = slots[2 - world]
        perm[slot] = positions[column]
        if signs[column] < 0:
            flips.append(slot)
    array = np.transpose(image.array, perm)
    for slot in flips:
        array = np.flip(array, axis=slot)
    array = np.ascontiguousarray(array)
    flipped_names = {names[perm[slot]] for slot in flips}
    if perm == list(range(len(names))) and not flips:
        report.choose("reorient", f"{target} (already)")
        return image

    data = image.meta.to_dict(include_lazy=True)
    data["axes"] = [data["axes"][p] for p in perm]
    shape = dict(zip(names, image.array.shape))
    data["spaces"] = [_flip_space(s, flipped_names, shape) for s in data.get("spaces", [])]
    ml = data.get("profiles", {}).get("ml")
    if ml and ml.get("boxes", {}).get("frame") == "index":
        ml.pop("boxes")
        report.warn("reorient: ml boxes in index coordinates dropped (they referred to the old array layout)")
    for entry in data.get("source", []):
        if entry.get("index_map"):
            entry["index_map"] = _compose(entry["index_map"], perm, flips)
    meta = Meta.from_dict(data, shape=array.shape)
    report.choose("reorient", f"{target}: axes {[names[p] for p in perm]}, flipped {sorted(flipped_names)}")
    return Image(array, meta)


def _flip_space(space: Dict[str, Any], flipped: set, shape: Dict[str, int]) -> Dict[str, Any]:
    from_array = space["from_array"]
    axes = from_array["array_axes"]
    if not any(name in flipped for name in axes):
        return space
    if "matrix" in from_array:
        matrix = np.array(from_array["matrix"], dtype=np.float64)
    else:
        n = len(from_array["scale"])
        matrix = np.eye(n + 1)
        matrix[:n, :n] = np.diag(from_array["scale"])
        matrix[:n, n] = from_array["translation"]
    for column, name in enumerate(axes):
        if name in flipped:
            matrix[:-1, -1] += matrix[:-1, column] * (shape[name] - 1)
            matrix[:-1, column] *= -1
    new = {k: v for k, v in from_array.items() if k not in ("matrix", "scale", "translation")}
    new["matrix"] = matrix.tolist()
    return dict(space, from_array=new)


def _compose(index_map: Dict[str, Any], perm: List[int], flips: List[int]) -> Dict[str, Any]:
    order = list(index_map["axis_order"])
    flipped = list(index_map.get("flipped", [False] * len(order)))
    offset = list(index_map.get("offset", [0] * len(order)))
    return {
        **index_map,
        "axis_order": [order[p] for p in perm],
        "flipped": [bool(flipped[p]) != (slot in flips) for slot, p in enumerate(perm)],
        "offset": [offset[p] for p in perm],
    }
