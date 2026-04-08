from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from medvol.geometry import CoordinateContext


@dataclass(frozen=True)
class BackendLoadResult:
    array: np.ndarray
    affine: np.ndarray
    header: Any
    coordinate_context: CoordinateContext | None
    backend: str


class MedVolBackend(Protocol):
    name: str

    def load(self, filepath: Path) -> BackendLoadResult:
        ...

    def save(self, filepath: Path, medvol: "MedVol") -> None:
        ...
