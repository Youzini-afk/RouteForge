"""Strict fail-closed replay policy."""

from __future__ import annotations

from dataclasses import dataclass

from .capabilities import requires_weights
from .context import RuntimeContext
from .errors import ReplayPolicyError, UnsupportedReplayLevelError
from .enums import ReplayLevel
from .reason_codes import ReplayReasonCode
from .route_record import RouteRecord
from .validation import validate_compatibility, validate_record, validate_tensor_dtypes


@dataclass(frozen=True)
class ReplayPolicy:
    """Replay policy flags.

    Defaults are intentionally conservative: no partial replay, no dtype cast,
    no downgrade, and strict compatibility checks.
    """

    strict: bool = True
    allow_downgrade: bool = False
    allow_partial: bool = False
    allow_cast: bool = False
    fallback_to_live: bool = True
    weight_tolerance: float = 0.0
    require_token_identity: bool = True
    require_model_match: bool = True
    require_backend_capability: bool = True


def apply_policy(policy: ReplayPolicy, record: RouteRecord, context: RuntimeContext) -> RouteRecord:
    """Return a replayable record or raise `ReplayPolicyError` fail-closed.

    The returned record may be an explicit R1 downgrade only when the policy
    enables downgrade. Non-downgrade compatibility failures remain fatal.
    """

    if context.replay_level in {ReplayLevel.R3, ReplayLevel.R4, ReplayLevel.R5}:
        raise UnsupportedReplayLevelError(
            f"{context.replay_level.name} is reserved for later phases and is not supported"
        )

    if record.token_count != context.token_count and not policy.allow_partial:
        raise ReplayPolicyError(
            f"partial replay disabled: token_count mismatch record={record.token_count}, "
            f"context={context.token_count}",
            ReplayReasonCode.PARTIAL_REPLAY_DISABLED,
        )

    dtype_result = validate_tensor_dtypes(record, context)
    if not dtype_result.passed and not policy.allow_cast:
        raise ReplayPolicyError(
            f"cast disabled: {dtype_result.reason}",
            dtype_result.code or ReplayReasonCode.CAST_DISABLED,
        )

    working = record
    if _needs_downgrade(record, context):
        if not policy.allow_downgrade:
            raise ReplayPolicyError(
                "downgrade disabled: context requires weighted replay but record cannot provide weights",
                ReplayReasonCode.DOWNGRADE_DISABLED,
            )
        working = record.downgraded_to_r1()
        context = RuntimeContext(
            token_count=context.token_count,
            layer_id=context.layer_id,
            top_k=context.top_k,
            num_experts=context.num_experts,
            replay_level=ReplayLevel.R1,
            abi_version=context.abi_version,
            backend_id=context.backend_id,
            backend_version=context.backend_version,
            model_arch=context.model_arch,
            model_config_digest=context.model_config_digest,
            layer_count=context.layer_count,
            phase=context.phase,
            step_id=context.step_id,
            request_ids=context.request_ids,
            sequence_ids=context.sequence_ids,
            token_positions=context.token_positions,
            block_ids=context.block_ids,
            token_order=context.token_order,
            weight_semantics=context.weight_semantics,
            expert_namespace=context.expert_namespace,
            index_dtype=context.index_dtype,
            weight_dtype=context.weight_dtype,
            device=context.device,
            distributed_topology=context.distributed_topology,
            expert_placement=context.expert_placement,
        )

    record_result = validate_record(working)
    if not record_result.passed:
        raise ReplayPolicyError(record_result.reason or "record validation failed", record_result.code)

    compatibility = validate_compatibility(working, context)
    if not compatibility.passed:
        raise ReplayPolicyError(
            compatibility.reason or "compatibility validation failed", compatibility.code
        )

    return working


def _needs_downgrade(record: RouteRecord, context: RuntimeContext) -> bool:
    if not requires_weights(context.replay_level):
        return False
    return record.replay_level == ReplayLevel.R1 or record.topk_weights is None
