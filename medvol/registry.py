"""Format plugin registry: built-in plugins plus the ``medvol.formats`` entry point group."""

from importlib import import_module
from importlib.metadata import entry_points
from typing import Dict, List

from medvol.plugins.base import FormatPlugin, PathLike

BUILTIN = {
    "mla": "medvol.plugins.mla:MlaPlugin",
    "nifti": "medvol.plugins.nifti:NiftiPlugin",
}

_plugins: Dict[str, FormatPlugin] = {}


def _load(spec: str) -> FormatPlugin:
    module, _, name = spec.partition(":")
    return getattr(import_module(module), name)()


def plugins() -> Dict[str, FormatPlugin]:
    """All registered plugins by name (loaded once)."""
    if not _plugins:
        for name, spec in BUILTIN.items():
            _plugins[name] = _load(spec)
        for entry in entry_points(group="medvol.formats"):
            if entry.name not in _plugins:
                _plugins[entry.name] = entry.load()()
    return _plugins


def register(plugin: FormatPlugin) -> None:
    """Register a plugin at runtime (replaces a plugin of the same name)."""
    plugins()[plugin.name] = plugin


def get(name: str) -> FormatPlugin:
    try:
        return plugins()[name]
    except KeyError:
        raise ValueError(f"Unknown format {name!r}; known: {sorted(plugins())}.") from None


def for_reading(path: PathLike, format: str = None) -> FormatPlugin:
    """The plugin that reads ``path``: the given format, else the most confident sniffer."""
    if format is not None:
        return get(format)
    scored = sorted(((plugin.sniff(path), plugin.name) for plugin in plugins().values()), reverse=True)
    if not scored or scored[0][0] <= 0:
        raise ValueError(f"No reader recognises {path}; known formats: {sorted(plugins())}.")
    return get(scored[0][1])


def for_writing(path: PathLike, format: str = None) -> FormatPlugin:
    """The plugin that writes ``path``: the given format, else by extension."""
    if format is not None:
        return get(format)
    matching: List[FormatPlugin] = [p for p in plugins().values() if p.matches_extension(path)]
    if not matching:
        raise ValueError(f"No writer for the extension of {path}; known formats: {sorted(plugins())}.")
    return max(matching, key=lambda p: max(len(e) for e in p.extensions if str(path).lower().endswith(e)))
