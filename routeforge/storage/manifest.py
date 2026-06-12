"""RouteTape manifest metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TAPE_VERSION = "0.1.0"


@dataclass(frozen=True)
class RouteTapeManifest:
    """Tape-level metadata without token-level route payloads."""

    tape_id: str
    model_id: str
    num_records: int = 0
    version: str = TAPE_VERSION
    abi_version: str = "routeforge.route_abi.v0"
    backend_id: str | None = None
    created_by: str = "routeforge"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteTapeManifest":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in allowed})
