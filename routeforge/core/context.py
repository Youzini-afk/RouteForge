"""Runtime context used for fail-closed compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import ReplayLevel, RoutePhase
from .route_record import ExpertNamespace, WeightSemantics
from .tensor import tensor_equal, tensor_hash


@dataclass(frozen=True, eq=False)
class RuntimeContext:
    """Current execution site for a possible route replay."""

    token_count: int
    layer_id: int
    top_k: int
    num_experts: int
    replay_level: ReplayLevel = ReplayLevel.R2
    abi_version: str = "routeforge.route_abi.v0"
    backend_id: str = "unknown"
    backend_version: str | None = None
    model_arch: str | None = None
    model_config_digest: str | None = None
    layer_count: int | None = None
    phase: str | RoutePhase = RoutePhase.DECODE
    step_id: int | None = 0
    request_ids: Any | None = None
    sequence_ids: Any | None = None
    token_positions: Any | None = None
    block_ids: Any | None = None
    token_order: Any | None = None
    weight_semantics: WeightSemantics = field(default_factory=WeightSemantics)
    expert_namespace: ExpertNamespace = field(default_factory=ExpertNamespace)
    index_dtype: str | None = "int16"
    weight_dtype: str | None = "float32"
    device: str | None = None
    distributed_topology: dict[str, Any] | None = None
    expert_placement: dict[str, Any] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeContext):
            return NotImplemented
        return (
            self.token_count == other.token_count
            and self.layer_id == other.layer_id
            and self.top_k == other.top_k
            and self.num_experts == other.num_experts
            and self.replay_level == other.replay_level
            and self.abi_version == other.abi_version
            and self.backend_id == other.backend_id
            and self.backend_version == other.backend_version
            and self.model_arch == other.model_arch
            and self.model_config_digest == other.model_config_digest
            and self.layer_count == other.layer_count
            and self.phase == other.phase
            and self.step_id == other.step_id
            and self.weight_semantics == other.weight_semantics
            and self.expert_namespace == other.expert_namespace
            and self.index_dtype == other.index_dtype
            and self.weight_dtype == other.weight_dtype
            and self.device == other.device
            and tensor_equal(self.request_ids, other.request_ids)
            and tensor_equal(self.sequence_ids, other.sequence_ids)
            and tensor_equal(self.token_positions, other.token_positions)
            and tensor_equal(self.block_ids, other.block_ids)
            and tensor_equal(self.token_order, other.token_order)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.token_count,
                self.layer_id,
                self.top_k,
                self.num_experts,
                self.replay_level,
                self.abi_version,
                self.backend_id,
                self.backend_version,
                self.model_arch,
                self.model_config_digest,
                self.layer_count,
                str(self.phase),
                self.step_id,
                self.weight_semantics,
                self.expert_namespace,
                self.index_dtype,
                self.weight_dtype,
                self.device,
                tensor_hash(self.request_ids),
                tensor_hash(self.sequence_ids),
                tensor_hash(self.token_positions),
                tensor_hash(self.block_ids),
                tensor_hash(self.token_order),
            )
        )
