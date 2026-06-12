"""Route decision records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import ReplayLevel, RoutePhase
from .tensor import tensor_equal, tensor_hash


@dataclass(frozen=True)
class WeightSemantics:
    """Meaning of recorded route weights."""

    score_function: str = "softmax"
    normalized: bool = True
    includes_bias: bool = False


@dataclass(frozen=True)
class ExpertNamespace:
    """Expert namespace metadata for routed/shared expert separation."""

    namespace_id: str = "routed"
    routed_expert_count: int | None = None
    shared_expert_count: int = 0


@dataclass(frozen=True)
class SharedExpertRecord:
    """Reserved schema for shared expert path capture.

    Phase 1 does not replay shared experts; validation fails closed when shared
    experts are present and a backend has not declared support.
    """

    enabled: bool = False
    policy: str = "always_on"
    gate: Any | None = None
    scale: Any | None = None
    expert_id: int | None = None
    semantics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class RouteRecord:
    """A model-agnostic MoE route decision for one layer/phase/step."""

    topk_idx: Any
    topk_weights: Any | None
    layer_id: int
    num_experts: int
    top_k: int
    token_count: int
    record_id: str = "rec_0"
    replay_level: ReplayLevel = ReplayLevel.R2
    abi_version: str = "routeforge.route_abi.v0"
    phase: str | RoutePhase = RoutePhase.DECODE
    step_id: int | None = 0
    request_ids: Any | None = None
    sequence_ids: Any | None = None
    token_positions: Any | None = None
    block_ids: Any | None = None
    token_order: Any | None = None
    weight_semantics: WeightSemantics = field(default_factory=WeightSemantics)
    expert_namespace: ExpertNamespace = field(default_factory=ExpertNamespace)
    router_logits: Any | None = None
    shared_expert: SharedExpertRecord | None = None
    dispatch: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def downgraded_to_r1(self) -> "RouteRecord":
        """Return an index-only copy for explicit R2->R1 downgrade."""

        return RouteRecord(
            topk_idx=self.topk_idx,
            topk_weights=None,
            layer_id=self.layer_id,
            num_experts=self.num_experts,
            top_k=self.top_k,
            token_count=self.token_count,
            record_id=self.record_id,
            replay_level=ReplayLevel.R1,
            abi_version=self.abi_version,
            phase=self.phase,
            step_id=self.step_id,
            request_ids=self.request_ids,
            sequence_ids=self.sequence_ids,
            token_positions=self.token_positions,
            block_ids=self.block_ids,
            token_order=self.token_order,
            weight_semantics=self.weight_semantics,
            expert_namespace=self.expert_namespace,
            router_logits=self.router_logits,
            shared_expert=self.shared_expert,
            dispatch=None,
            metadata=dict(self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RouteRecord):
            return NotImplemented
        return (
            self.abi_version == other.abi_version
            and self.layer_id == other.layer_id
            and self.phase == other.phase
            and self.step_id == other.step_id
            and self.num_experts == other.num_experts
            and self.top_k == other.top_k
            and self.token_count == other.token_count
            and self.record_id == other.record_id
            and self.replay_level == other.replay_level
            and self.weight_semantics == other.weight_semantics
            and self.expert_namespace == other.expert_namespace
            and tensor_equal(self.topk_idx, other.topk_idx)
            and tensor_equal(self.topk_weights, other.topk_weights)
            and tensor_equal(self.request_ids, other.request_ids)
            and tensor_equal(self.sequence_ids, other.sequence_ids)
            and tensor_equal(self.token_positions, other.token_positions)
            and tensor_equal(self.block_ids, other.block_ids)
            and tensor_equal(self.token_order, other.token_order)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.abi_version,
                self.layer_id,
                str(self.phase),
                self.step_id,
                self.num_experts,
                self.top_k,
                self.token_count,
                self.record_id,
                self.replay_level,
                self.weight_semantics,
                self.expert_namespace,
                tensor_hash(self.topk_idx),
                tensor_hash(self.topk_weights),
                tensor_hash(self.request_ids),
                tensor_hash(self.sequence_ids),
                tensor_hash(self.token_positions),
                tensor_hash(self.block_ids),
                tensor_hash(self.token_order),
            )
        )
