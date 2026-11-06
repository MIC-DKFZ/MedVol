"""``.mla`` plugin: MLArray files hold the model itself, so nothing is converted (spec 7.7)."""

from typing import List, Tuple

from mlarray import MLArray

from medvol.image import Image
from medvol.plugins.base import Capabilities, FormatPlugin, PathLike
from medvol.policy import Policy
from medvol.report import Report


class MlaPlugin(FormatPlugin):
    name = "mla"
    extensions = (".mla",)
    capabilities = Capabilities(
        max_space_axes=32, min_space_axes=0, max_ndim=32,
        non_space_axes=frozenset({"time", "channel", "component", "other"}),
        label_table=True, key_value=True,
    )

    def sniff(self, path: PathLike) -> float:
        if not self.matches_extension(path):
            return 0.0
        try:
            with open(path, "rb") as handle:
                head = handle.read(32)
        except OSError:
            return 0.0
        return 1.0 if b"b2frame" in head else 0.5

    def read(self, path: PathLike, policy: Policy) -> Tuple[List[Image], Report]:
        return [Image.from_mlarray(MLArray(path))], Report("read", self.name, str(path))

    def write(self, image: Image, path: PathLike, policy: Policy) -> Report:
        image.to_mlarray().save(path)
        return Report("write", self.name, str(path))
