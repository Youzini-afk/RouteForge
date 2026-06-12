"""Central fail-closed replay gate.

Every production replay path should pass through this module instead of calling
adapter-specific checks directly. It binds together RouteRecord validation,
RuntimeContext compatibility, backend capability negotiation, token identity
requirements, shared-expert support, and ReplayPolicy downgrade/cast rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from .capabilities import BackendCapabilities, requires_dispatch
from .context import RuntimeContext
from .enums import ReplayDecision, ReplayLevel
from .errors import ReplayPolicyError, UnsupportedReplayLevelError
from .replay_policy import ReplayPolicy, apply_policy
from .route_record import RouteRecord
from .tensor import tensor_dtype_name


@dataclass(frozen=True)
class ReplayDecisionResult:
    """Observable replay decision for future serving/runtime integrations."""

    decision: ReplayDecision
    record: RouteRecord | None = None
    level: ReplayLevel | None = None
    reason_code: str | None = None
    message: str | None = None


def validate_replay_request(
    record: RouteRecord,
    context: RuntimeContext,
    policy: ReplayPolicy,
    capabilities: BackendCapabilities,
) -> RouteRecord:
    """Return a safe replay record or raise a typed fail-closed error."""

    _validate_level_supported(context, capabilities)
    _validate_shared_expert(record, capabilities)
    _validate_token_identity_presence(record, context, policy)
    _validate_capability_dtypes(record, capabilities, policy)
    return apply_policy(policy, record, context)


def decide_replay_request(
    record: RouteRecord,
    context: RuntimeContext,
    policy: ReplayPolicy,
    capabilities: BackendCapabilities,
) -> ReplayDecisionResult:
    """Decision-returning wrapper for integrations that prefer fallback objects."""

    try:
        safe_record = validate_replay_request(record, context, policy, capabilities)
    except ReplayPolicyError as exc:
        return ReplayDecisionResult(
            decision=ReplayDecision.FAIL_CLOSED,
            reason_code=exc.__class__.__name__,
            message=str(exc),
        )
    level = safe_record.replay_level
    decision = ReplayDecision.REPLAY_R2 if level == ReplayLevel.R2 else ReplayDecision.REPLAY_R1
    return ReplayDecisionResult(decision=decision, record=safe_record, level=level)


def _validate_level_supported(context: RuntimeContext, capabilities: BackendCapabilities) -> None:
    level = context.replay_level
    if requires_dispatch(level):
        raise UnsupportedReplayLevelError(
            f"{level.name} requires dispatch/topology support and is disabled in this phase"
        )
    if not capabilities.supports_level(level):
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support replay level {level.name}"
        )


def _validate_shared_expert(record: RouteRecord, capabilities: BackendCapabilities) -> None:
    shared = record.shared_expert
    if shared is not None and shared.enabled and not capabilities.supports_shared_expert:
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support shared expert replay"
        )
    if record.expert_namespace.shared_expert_count and not capabilities.supports_shared_expert:
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support shared expert namespace"
        )


def _validate_token_identity_presence(
    record: RouteRecord,
    context: RuntimeContext,
    policy: ReplayPolicy,
) -> None:
    if not policy.require_token_identity:
        return
    if record.step_id is None or context.step_id is None:
        raise ReplayPolicyError("token identity missing: step_id is required")
    if record.token_positions is None or context.token_positions is None:
        raise ReplayPolicyError("token identity missing: token_positions are required")
    has_order = record.token_order is not None and context.token_order is not None
    has_request = record.request_ids is not None and context.request_ids is not None
    has_sequence = record.sequence_ids is not None and context.sequence_ids is not None
    if not (has_order or has_request or has_sequence):
        raise ReplayPolicyError(
            "token identity missing: token_order, request_ids, or sequence_ids are required"
        )


def _validate_capability_dtypes(
    record: RouteRecord,
    capabilities: BackendCapabilities,
    policy: ReplayPolicy,
) -> None:
    idx_dtype = tensor_dtype_name(record.topk_idx)
    if idx_dtype and idx_dtype not in capabilities.supported_index_dtypes and not policy.allow_cast:
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support topk_idx dtype {idx_dtype}"
        )
    if record.topk_weights is not None:
        weight_dtype = tensor_dtype_name(record.topk_weights)
        if weight_dtype and weight_dtype not in capabilities.supported_weight_dtypes and not policy.allow_cast:
            raise ReplayPolicyError(
                f"backend {capabilities.backend_id} does not support topk_weights dtype {weight_dtype}"
            )
