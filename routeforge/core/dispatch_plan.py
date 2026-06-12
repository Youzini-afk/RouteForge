"""Reserved R3+ dispatch layout schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DispatchPlan:
    """R3 dispatch layout placeholder.

    Phase 1 defines this boundary but does not implement DeepEP or handle replay.
    """

    dispatcher_backend: str
    source_record_ref: str | None = None
    abi_version: str = "routeforge.route_abi.v0"
    dispatcher_version: str | None = None
    token_permutation: Any | None = None
    inverse_permutation: Any | None = None
    expert_counts: Any | None = None
    expert_offsets: Any | None = None
    capacity: int | None = None
    padding: Any | None = None
    drop_mask: Any | None = None
    rank_mapping: Any | None = None
    expert_placement: dict[str, Any] | None = None
    communication_group_id: str | None = None
    topology_signature: str | None = None
