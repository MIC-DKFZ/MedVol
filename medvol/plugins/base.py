"""Plugin interface, writer capabilities and header field ownership (spec sections 3.2-4)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple, Union

from medvol.image import Image
from medvol.policy import Policy
from medvol.report import Report

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Capabilities:
    """What a writer can represent; anything else is reported and dropped, or raises (``on_loss``)."""

    max_space_axes: int = 3
    min_space_axes: int = 1
    non_space_axes: FrozenSet[str] = frozenset()   # axis types besides "space" (time, channel, component, other)
    max_ndim: int = 7
    oblique: bool = True
    shear: bool = True
    label_table: bool = False
    key_value: bool = False
    dtypes: Optional[FrozenSet[str]] = None        # numpy dtype names; None = any


@dataclass(frozen=True)
class FieldTable:
    """Ownership of a format's header fields (spec section 4).

    - core-owned: always regenerated from the MLArray core on export;
    - data-dependent: kept while the source fingerprint matches, else recomputed or dropped;
    - everything else: verbatim.
    """

    core_owned: FrozenSet[str] = frozenset()
    data_dependent: FrozenSet[str] = frozenset()

    def classify(self, name: str) -> str:
        if name in self.core_owned:
            return "core"
        if name in self.data_dependent:
            return "data"
        return "verbatim"


class FormatPlugin(ABC):
    name: str = ""
    extensions: Tuple[str, ...] = ()
    capabilities: Capabilities = Capabilities()
    fields: FieldTable = FieldTable()

    def matches_extension(self, path: PathLike) -> bool:
        lower = str(path).lower()
        return any(lower.endswith(ext) for ext in self.extensions)

    @abstractmethod
    def sniff(self, path: PathLike) -> float:
        """Confidence 0..1 that ``path`` is in this format, from its content where possible."""

    @abstractmethod
    def read(self, path: PathLike, policy: Policy, **options) -> Tuple[List[Image], Report]:
        """All images in ``path`` (DICOM can hold several stacks) and the report.

        ``options`` are format-specific (e.g. the ``reference`` series of a DICOM RTSTRUCT).
        """

    @abstractmethod
    def write(self, image: Image, path: PathLike, policy: Policy, **options) -> Report:
        """Write ``image`` to ``path``; ``options`` are format-specific (e.g. DICOM SEG ``reference``)."""
