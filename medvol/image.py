"""The neutral in-memory image every reader produces and every writer consumes (spec section 3.1)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from mlarray import MLArray, Meta


@dataclass
class Image:
    """Stored values in their original dtype, in the array order ``meta.axes`` describes, plus metadata."""

    array: np.ndarray
    meta: Meta

    def __post_init__(self):
        self.array = np.asarray(self.array)
        if self.meta.shape != tuple(self.array.shape):
            self.meta._set_shape(self.array.shape)

    @property
    def axes(self):
        return self.meta.axes

    @property
    def spaces(self):
        return self.meta.spaces

    def space(self, name: Optional[str] = None):
        spaces = self.meta.spaces
        if name is None:
            return spaces[0] if spaces else None
        return next(s for s in spaces if s.name == name)

    def to_mlarray(self, **kwargs) -> MLArray:
        """An in-memory MLArray of this image (layout options as for ``MLArray``)."""
        return MLArray(self.array, meta=self.meta, **kwargs)

    @classmethod
    def from_mlarray(cls, image: MLArray) -> "Image":
        return cls(np.asarray(image.to_numpy()), image.meta.copy())
