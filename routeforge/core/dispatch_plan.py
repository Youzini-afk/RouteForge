"""Reserved R3+ dispatch layout schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tensor import tensor_shape


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


@dataclass(frozen=True)
class DispatchPlanValidationResult:
    passed: bool
    reason: str | None = None
    code: str | None = None


def validate_dispatch_plan(plan: DispatchPlan, *, token_count: int, num_experts: int) -> DispatchPlanValidationResult:
    """Validate R3 dispatch layout structure without enabling R3 replay."""

    if not plan.dispatcher_backend:
        return DispatchPlanValidationResult(False, "dispatcher_backend is required", "DISPATCH_BACKEND_MISSING")
    if plan.token_permutation is None:
        return DispatchPlanValidationResult(False, "token_permutation is required", "TOKEN_PERMUTATION_MISSING")
    if tensor_shape(plan.token_permutation) != (token_count,):
        return DispatchPlanValidationResult(False, "token_permutation shape mismatch", "TOKEN_PERMUTATION_SHAPE")
    if plan.inverse_permutation is None:
        return DispatchPlanValidationResult(False, "inverse_permutation is required", "INVERSE_PERMUTATION_MISSING")
    if tensor_shape(plan.inverse_permutation) != (token_count,):
        return DispatchPlanValidationResult(False, "inverse_permutation shape mismatch", "INVERSE_PERMUTATION_SHAPE")
    if plan.expert_counts is None:
        return DispatchPlanValidationResult(False, "expert_counts is required", "EXPERT_COUNTS_MISSING")
    if tensor_shape(plan.expert_counts) != (num_experts,):
        return DispatchPlanValidationResult(False, "expert_counts shape mismatch", "EXPERT_COUNTS_SHAPE")
    if plan.expert_offsets is None:
        return DispatchPlanValidationResult(False, "expert_offsets is required", "EXPERT_OFFSETS_MISSING")
    if tensor_shape(plan.expert_offsets) != (num_experts,):
        return DispatchPlanValidationResult(False, "expert_offsets shape mismatch", "EXPERT_OFFSETS_SHAPE")
    if plan.capacity is not None and plan.capacity < 0:
        return DispatchPlanValidationResult(False, "capacity must be non-negative", "CAPACITY_INVALID")
    if plan.topology_signature is None:
        return DispatchPlanValidationResult(False, "topology_signature is required", "TOPOLOGY_SIGNATURE_MISSING")
    return DispatchPlanValidationResult(True)
