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
from .reason_codes import ReplayReasonCode
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


@dataclass(frozen=True)
class TapeCompatibility:
    """Tape-level metadata used by the central replay gate.

    This class intentionally lives in core and accepts generic manifest-like
    objects so replay validation does not depend on a specific storage backend.
    """

    version: str | None = None
    abi_version: str | None = None
    model_id: str | None = None
    backend_id: str | None = None
    backend_version: str | None = None
    model_arch: str | None = None
    model_config_digest: str | None = None
    recorded_replay_levels: tuple[str, ...] = ()

    @classmethod
    def from_manifest(cls, manifest: Any) -> "TapeCompatibility":
        return cls(
            version=getattr(manifest, "version", None),
            abi_version=getattr(manifest, "abi_version", None),
            model_id=getattr(manifest, "model_id", None),
            backend_id=getattr(manifest, "backend_id", None),
            backend_version=getattr(manifest, "backend_version", None),
            model_arch=getattr(manifest, "model_arch", None),
            model_config_digest=getattr(manifest, "model_config_digest", None),
            recorded_replay_levels=tuple(getattr(manifest, "recorded_replay_levels", ()) or ()),
        )


def validate_replay_request(
    record: RouteRecord,
    context: RuntimeContext,
    policy: ReplayPolicy,
    capabilities: BackendCapabilities,
    tape: TapeCompatibility | None = None,
) -> RouteRecord:
    """Return a safe replay record or raise a typed fail-closed error."""

    if tape is not None:
        _validate_tape_compatibility(tape, context, capabilities, policy)
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
    tape: TapeCompatibility | None = None,
) -> ReplayDecisionResult:
    """Decision-returning wrapper for integrations that prefer fallback objects."""

    try:
        safe_record = validate_replay_request(record, context, policy, capabilities, tape)
    except ReplayPolicyError as exc:
        return ReplayDecisionResult(
            decision=ReplayDecision.FAIL_CLOSED,
            reason_code=exc.reason_code,
            message=str(exc),
        )
    level = safe_record.replay_level
    decision = ReplayDecision.REPLAY_R2 if level == ReplayLevel.R2 else ReplayDecision.REPLAY_R1
    return ReplayDecisionResult(decision=decision, record=safe_record, level=level)


def _validate_tape_compatibility(
    tape: TapeCompatibility,
    context: RuntimeContext,
    capabilities: BackendCapabilities,
    policy: ReplayPolicy,
) -> None:
    if tape.version and tape.version != "0.1.0":
        raise ReplayPolicyError(
            f"RouteTape version mismatch: tape={tape.version}, expected=0.1.0",
            ReplayReasonCode.TAPE_VERSION_MISMATCH,
        )
    if tape.abi_version and tape.abi_version != context.abi_version:
        raise ReplayPolicyError(
            f"RouteTape ABI mismatch: tape={tape.abi_version}, context={context.abi_version}",
            ReplayReasonCode.TAPE_ABI_VERSION_MISMATCH,
        )
    if tape.backend_id and tape.backend_id != capabilities.backend_id:
        raise ReplayPolicyError(
            f"RouteTape backend mismatch: tape={tape.backend_id}, backend={capabilities.backend_id}",
            ReplayReasonCode.TAPE_BACKEND_MISMATCH,
        )
    if tape.backend_version and capabilities.backend_version and tape.backend_version != capabilities.backend_version:
        raise ReplayPolicyError(
            f"RouteTape backend version mismatch: tape={tape.backend_version}, backend={capabilities.backend_version}",
            ReplayReasonCode.TAPE_BACKEND_VERSION_MISMATCH,
        )
    if policy.require_model_match:
        if tape.model_arch and context.model_arch and tape.model_arch != context.model_arch:
            raise ReplayPolicyError(
                f"RouteTape model_arch mismatch: tape={tape.model_arch}, context={context.model_arch}",
                ReplayReasonCode.TAPE_MODEL_ARCH_MISMATCH,
            )
        if (
            tape.model_config_digest
            and context.model_config_digest
            and tape.model_config_digest != context.model_config_digest
        ):
            raise ReplayPolicyError(
                "RouteTape model_config_digest mismatch",
                ReplayReasonCode.TAPE_MODEL_CONFIG_DIGEST_MISMATCH,
            )
    if tape.recorded_replay_levels and context.replay_level.value not in tape.recorded_replay_levels:
        raise ReplayPolicyError(
            f"RouteTape does not contain replay level {context.replay_level.value}",
            ReplayReasonCode.TAPE_REPLAY_LEVEL_MISMATCH,
        )


def _validate_level_supported(context: RuntimeContext, capabilities: BackendCapabilities) -> None:
    level = context.replay_level
    if requires_dispatch(level):
        raise UnsupportedReplayLevelError(
            f"{level.name} requires dispatch/topology support and is disabled in this phase"
        )
    if not capabilities.supports_level(level):
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support replay level {level.name}",
            ReplayReasonCode.BACKEND_LEVEL_NOT_SUPPORTED,
        )


def _validate_shared_expert(record: RouteRecord, capabilities: BackendCapabilities) -> None:
    shared = record.shared_expert
    if shared is not None and shared.enabled and not capabilities.supports_shared_expert:
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support shared expert replay",
            ReplayReasonCode.SHARED_EXPERT_UNSUPPORTED,
        )
    if record.expert_namespace.shared_expert_count and not capabilities.supports_shared_expert:
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support shared expert namespace",
            ReplayReasonCode.SHARED_EXPERT_NAMESPACE_UNSUPPORTED,
        )


def _validate_token_identity_presence(
    record: RouteRecord,
    context: RuntimeContext,
    policy: ReplayPolicy,
) -> None:
    if not policy.require_token_identity:
        return
    if record.step_id is None or context.step_id is None:
        raise ReplayPolicyError(
            "token identity missing: step_id is required",
            ReplayReasonCode.TOKEN_IDENTITY_MISSING_STEP_ID,
        )
    if record.token_positions is None or context.token_positions is None:
        raise ReplayPolicyError(
            "token identity missing: token_positions are required",
            ReplayReasonCode.TOKEN_IDENTITY_MISSING_POSITIONS,
        )
    has_order = record.token_order is not None and context.token_order is not None
    has_request = record.request_ids is not None and context.request_ids is not None
    has_sequence = record.sequence_ids is not None and context.sequence_ids is not None
    if not (has_order or has_request or has_sequence):
        raise ReplayPolicyError(
            "token identity missing: token_order, request_ids, or sequence_ids are required",
            ReplayReasonCode.TOKEN_IDENTITY_MISSING_ORDER_OR_REQUEST,
        )


def _validate_capability_dtypes(
    record: RouteRecord,
    capabilities: BackendCapabilities,
    policy: ReplayPolicy,
) -> None:
    idx_dtype = tensor_dtype_name(record.topk_idx)
    if idx_dtype and idx_dtype not in capabilities.supported_index_dtypes and not policy.allow_cast:
        raise ReplayPolicyError(
            f"backend {capabilities.backend_id} does not support topk_idx dtype {idx_dtype}",
            ReplayReasonCode.INDEX_DTYPE_NOT_SUPPORTED,
        )
    if record.topk_weights is not None:
        weight_dtype = tensor_dtype_name(record.topk_weights)
        if weight_dtype and weight_dtype not in capabilities.supported_weight_dtypes and not policy.allow_cast:
            raise ReplayPolicyError(
                f"backend {capabilities.backend_id} does not support topk_weights dtype {weight_dtype}",
                ReplayReasonCode.WEIGHT_DTYPE_NOT_SUPPORTED,
            )
