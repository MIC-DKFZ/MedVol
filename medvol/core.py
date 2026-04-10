from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medvol.geometry import (
    SNAP_ATOL,
    SIMPLEITK_AXIS_LABELS,
    CoordinateContext,
    _is_signed_permutation,
    affine_to_rotation,
    affine_to_shear,
    canonical_coordinate_context,
    canonicalize_array_and_affine,
    compose_affine,
    coordinate_system_from_affine,
    deoblique_affine,
    decompose_affine,
    normalize_backend_name,
    parse_coordinate_system,
    snap_values,
    validate_affine,
)
from medvol.registry import get_backend, resolve_backend


def _context_from_coordinate_system(
    coordinate_system: str | None,
    ndim: int,
) -> CoordinateContext:
    """Map a coordinate system string to a CoordinateContext.

    None or "RAS+" → canonical RAS+ context (default, backward-compatible).
    "LPS+"          → LPS+ context (SimpleITK / DICOM convention).
    Anything else   → ValueError.
    """
    if coordinate_system is None or coordinate_system in {"RAS", "RAS+"}:
        return canonical_coordinate_context(ndim)
    if coordinate_system in {"LPS", "LPS+"}:
        anatomical_ndim = min(ndim, 3)
        return CoordinateContext(
            axis_labels=SIMPLEITK_AXIS_LABELS[:anatomical_ndim],
            anatomical_ndim=anatomical_ndim,
            anatomical_axes=tuple(range(anatomical_ndim)),
        )
    raise ValueError(
        f"Unsupported coordinate_system {coordinate_system!r}. "
        "Supported values: 'RAS+', 'LPS+'."
    )


class MedVol:
    def __init__(
        self,
        source: np.ndarray | str | Path,
        *,
        affine: Sequence[Sequence[float]] | None = None,
        spacing: Sequence[float] | None = None,
        origin: Sequence[float] | None = None,
        direction: Sequence[Sequence[float]] | None = None,
        header: Any = None,
        coordinate_system: str | None = None,
        backend: str | None = None,
        canonicalize: bool = True,
    ) -> None:
        self._coordinate_context = None
        self._header = None
        self._backend = normalize_backend_name(backend)
        self._canonicalize = canonicalize

        if isinstance(source, (str, Path)):
            if coordinate_system is not None:
                raise ValueError(
                    "coordinate_system cannot be set when loading from a file."
                )
            if any(value is not None for value in (affine, spacing, origin, direction)):
                raise ValueError(
                    "affine, spacing, origin, and direction cannot be set when loading from a file."
                )
            resolved_backend = resolve_backend(source, self._backend)
            result = get_backend(resolved_backend).load(Path(source))
            self._array = self._validate_array(result.array)
            self._affine = validate_affine(result.affine, self.ndims)
            self._header = result.header
            self._coordinate_context = result.coordinate_context
            self._apply_orientation_policy()
            self._backend = result.backend
            return

        self._array = self._validate_array(source)
        if affine is not None and any(value is not None for value in (spacing, origin, direction)):
            raise ValueError("Use either affine or spacing/origin/direction, not both.")

        if affine is not None:
            self._affine = validate_affine(affine, self.ndims)
        else:
            spacing_values = np.ones((self.ndims,), dtype=float) if spacing is None else spacing
            origin_values = np.zeros((self.ndims,), dtype=float) if origin is None else origin
            direction_values = np.eye(self.ndims, dtype=float) if direction is None else direction
            self._affine = compose_affine(
                spacing_values,
                origin_values,
                direction_values,
                atol=SNAP_ATOL,
            )
        self._header = header
        self._coordinate_context = _context_from_coordinate_system(coordinate_system, self.ndims)
        self._apply_orientation_policy()

    def _apply_orientation_policy(self) -> None:
        if self._canonicalize:
            self._array, self._affine, self._coordinate_context = canonicalize_array_and_affine(
                self._array,
                self._affine,
                self._coordinate_context,
                atol=SNAP_ATOL,
            )

    @staticmethod
    def _validate_array(array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise ValueError("source must be a NumPy array or a filepath.")
        if array.ndim not in {2, 3, 4}:
            raise ValueError("Array must be 2D, 3D, or 4D.")
        return array

    @property
    def ndims(self) -> int:
        return self._array.ndim

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        array = self._validate_array(value)
        if hasattr(self, "_affine") and array.ndim != self.ndims:
            raise ValueError("array ndim cannot change without recreating the MedVol object.")
        self._array = array

    @property
    def affine(self) -> np.ndarray:
        return self._affine.copy()

    @affine.setter
    def affine(self, value: Sequence[Sequence[float]]) -> None:
        self._affine = validate_affine(value, self.ndims)

    @property
    def spacing(self) -> np.ndarray:
        spacing, _, _ = decompose_affine(self._affine)
        return spacing

    @spacing.setter
    def spacing(self, value: Sequence[float]) -> None:
        spacing = np.asarray(value, dtype=float)
        if spacing.shape != (self.ndims,):
            raise ValueError(f"spacing must have shape {(self.ndims,)}.")
        if np.any(spacing <= 0):
            raise ValueError("spacing must contain positive nonzero values.")
        _, origin, direction = decompose_affine(self._affine)
        self._affine = compose_affine(spacing, origin, direction)

    @property
    def origin(self) -> np.ndarray:
        _, origin, _ = decompose_affine(self._affine)
        return origin

    @origin.setter
    def origin(self, value: Sequence[float]) -> None:
        origin = np.asarray(value, dtype=float)
        if origin.shape != (self.ndims,):
            raise ValueError(f"origin must have shape {(self.ndims,)}.")
        spacing, _, direction = decompose_affine(self._affine)
        self._affine = compose_affine(spacing, origin, direction)

    @property
    def direction(self) -> np.ndarray:
        _, _, direction = decompose_affine(self._affine)
        return direction

    @direction.setter
    def direction(self, value: Sequence[Sequence[float]]) -> None:
        direction = np.asarray(value, dtype=float)
        if direction.shape != (self.ndims, self.ndims):
            raise ValueError(
                f"direction must have shape {(self.ndims, self.ndims)}."
            )
        spacing, origin, _ = decompose_affine(self._affine)
        self._affine = compose_affine(spacing, origin, direction)

    @property
    def translation(self) -> np.ndarray:
        return self.origin

    @property
    def scale(self) -> np.ndarray:
        return self.spacing

    @property
    def rotation(self) -> np.ndarray:
        return affine_to_rotation(self._affine)

    @property
    def shear(self) -> np.ndarray:
        return affine_to_shear(self._affine)

    @property
    def coordinate_system(self) -> str | None:
        return coordinate_system_from_affine(self._affine, self._coordinate_context)

    @property
    def header(self) -> Any:
        return self._header

    @header.setter
    def header(self, value: Any) -> None:
        self._header = value

    @property
    def backend(self) -> str | None:
        return self._backend

    def get_geometry(
        self,
        coordinate_system: str = "RAS+",
        *,
        deoblique: bool = False,
    ) -> dict:
        """Return geometry converted to the requested coordinate system.

        The internal state is not modified — only the returned values are
        converted.  Requires ``canonicalize=True`` (raises ``ValueError``
        otherwise).

        Args:
            coordinate_system: Target coordinate system, e.g. ``"RAS+"``,
                ``"LPS+"``, ``"ASR+"``.  Must contain exactly
                ``spatial_ndim`` anatomical letters (R/L, A/P, S/I), each
                from a different anatomical axis, with an optional trailing
                ``"+"``.
            deoblique: If ``True``, strip off-diagonal entries from the
                returned affine (diagonal affine, keeps origin).  Equivalent
                to calling ``get_geometry(deoblique=False)`` and then
                removing the oblique component.

        Returns:
            Dict with keys:

            * ``"affine"`` — (ndim+1)×(ndim+1) affine in the target system.
            * ``"spacing"`` — always-positive voxel spacing (column norms).
            * ``"origin"`` — world coordinates of voxel (0, 0, …, 0).
            * ``"direction"`` — unit-column direction cosine matrix.
            * ``"coordinate_system"`` — the *coordinate_system* argument.
            * ``"oblique"`` — ``True`` when the spatial direction block is
              not a signed permutation matrix (i.e. the image is oblique).

        Raises:
            ValueError: If ``canonicalize=False`` or the coordinate context
                is unknown.
        """
        if not self._canonicalize:
            raise ValueError("get_geometry requires canonicalize=True.")
        if self._coordinate_context is None:
            raise ValueError(
                "get_geometry requires a known coordinate context. "
                "Load from a file with a recognised coordinate system."
            )

        spatial_ndim = self._coordinate_context.anatomical_ndim
        axis_order, flips = parse_coordinate_system(coordinate_system, spatial_ndim)
        signs = [-1 if f else 1 for f in flips]

        A = self._affine
        shape = self._array.shape

        # ── Step 1: permute and sign spatial columns (data-axis transform) ──
        # Read from original A; write to A_mid so we never clobber a source col.
        A_mid = A.copy()
        for m in range(spatial_ndim):
            A_mid[:, m] = signs[m] * A[:, axis_order[m]]
        # Adjust translation column for flipped axes:
        # flip on axis m maps voxel i → (N-1-i), shifting the origin to the far corner.
        for m in range(spatial_ndim):
            if flips[m]:
                A_mid[:, -1] += A[:, axis_order[m]] * (shape[axis_order[m]] - 1)

        # ── Step 2: permute and sign spatial rows (world-basis transform) ──
        A_final = A_mid.copy()
        for m in range(spatial_ndim):
            A_final[m, :] = signs[m] * A_mid[axis_order[m], :]

        A_final = snap_values(A_final)

        if deoblique:
            A_final = deoblique_affine(A_final)

        spacing, origin, direction = decompose_affine(A_final)

        # Oblique iff the spatial direction block is not a signed permutation.
        spatial_dir = direction[:spatial_ndim, :spatial_ndim]
        is_oblique = not _is_signed_permutation(spatial_dir)

        return {
            "affine": A_final,
            "spacing": spacing,
            "origin": origin,
            "direction": direction,
            "coordinate_system": coordinate_system,
            "oblique": is_oblique,
        }

    def get_array(self, coordinate_system: str = "RAS+") -> np.ndarray:
        """Return the array converted to the requested coordinate system.

        Returns a zero-copy NumPy view — no interpolation is performed.
        Only axis permutations and flips are applied.  Requires
        ``canonicalize=True`` (raises ``ValueError`` otherwise).

        Args:
            coordinate_system: Target coordinate system string (see
                ``get_geometry`` for the accepted format).

        Returns:
            NumPy array in the requested axis order and orientation.
            Non-spatial axes (e.g. time for 4-D images) are appended
            unchanged at the end.

        Raises:
            ValueError: If ``canonicalize=False`` or the coordinate context
                is unknown.
        """
        if not self._canonicalize:
            raise ValueError("get_array requires canonicalize=True.")
        if self._coordinate_context is None:
            raise ValueError(
                "get_array requires a known coordinate context."
            )

        spatial_ndim = self._coordinate_context.anatomical_ndim
        axis_order, flips = parse_coordinate_system(coordinate_system, spatial_ndim)

        # Non-spatial axes (e.g. time) stay at the end, in their original order.
        full_axis_order = list(axis_order) + list(range(spatial_ndim, self.ndims))
        result = np.transpose(self._array, full_axis_order)

        for m, flip in enumerate(flips):
            if flip:
                result = np.flip(result, axis=m)

        return result

    def save(self, filepath: str | Path, *, backend: str | None = None) -> None:
        resolved_backend = resolve_backend(filepath, backend)
        get_backend(resolved_backend).save(Path(filepath), self)

    def __repr__(self) -> str:
        coordinate_system = self.coordinate_system
        return (
            f"MedVol(shape={self.array.shape}, backend={self.backend!r}, "
            f"coordinate_system={coordinate_system!r})"
        )
