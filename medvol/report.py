"""What a read, write or convert did: choices made, warnings and dropped fields (spec section 2.4)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class LossError(ValueError):
    """Something cannot be represented in the target format and policy ``on_loss`` is ``error``."""


@dataclass
class Report:
    operation: str
    format: str
    path: Optional[str] = None
    choices: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)

    def choose(self, name: str, value: Any) -> None:
        """Record a choice made by a policy or by the data."""
        self.choices[name] = value

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def drop(self, what: str, reason: str, on_loss: str = "warn") -> None:
        """Record something the target cannot represent; raise if ``on_loss`` is ``error``."""
        message = f"{what}: {reason}"
        if on_loss == "error":
            raise LossError(message)
        self.dropped.append(message)

    def extend(self, other: "Report") -> None:
        self.choices.update(other.choices)
        self.warnings.extend(other.warnings)
        self.dropped.extend(other.dropped)

    def history_entry(self, tool: str) -> Dict[str, Any]:
        """The entry this report adds to an image's history."""
        entry: Dict[str, Any] = {"action": "converted" if self.operation == "read" else self.operation,
                                 "tool": tool, "from": self.format}
        if self.choices:
            entry["settings"] = dict(self.choices)
        notes = list(self.warnings) + [f"dropped {item}" for item in self.dropped]
        if notes:
            entry["warnings"] = notes
        return entry

    def __str__(self) -> str:
        lines = [f"{self.operation} {self.format}" + (f" {self.path}" if self.path else "")]
        lines += [f"  choice: {key} = {value}" for key, value in self.choices.items()]
        lines += [f"  warning: {message}" for message in self.warnings]
        lines += [f"  dropped: {message}" for message in self.dropped]
        return "\n".join(lines)
