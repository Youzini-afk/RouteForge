"""Fail-closed validation for RouteForge core ABI objects."""

from __future__ import annotations

from .capabilities import requires_weights
from .context import RuntimeContext
from .enums import ReplayLevel
from .route_record import RouteRecord
from .tensor import (
    tensor_all_finite,
    tensor_dtype_name,
    tensor_equal,
    tensor_max,
    tensor_min,
    tensor_shape,
)


class _PassedDescriptor:
    """Expose bool on instances and a factory on the class.

    This supports both test ergonomics:
    `result.passed is True` and `ValidationResult.passed()`.
    """

    def __get__(self, instance: "ValidationResult | None", owner: type) -> object:
        if instance is None:
            return owner.passed_result
        return instance._passed


class ValidationResult:
    """Boolean-compatible validation result with a fail-closed reason."""

    passed = _PassedDescriptor()

    def __init__(self, passed: bool, reason: str | None = None, code: str | None = None):
        self._passed = passed
        self.reason = reason
        self.code = code

    @classmethod
    def failed(cls, reason: str, code: str | None = None) -> "ValidationResult":
        return cls(False, reason, code)

    @classmethod
    def passed_result(cls) -> "ValidationResult":
        return cls(True, None, None)

    def __bool__(self) -> bool:
        return bool(self._passed)


def validate_record(record: RouteRecord) -> ValidationResult:
    """Validate structural invariants of a route record."""

    idx_shape = tensor_shape(record.topk_idx)
    if len(idx_shape) != 2:
        return ValidationResult.failed(
            f"topk_idx shape must be [token_count, top_k], got {idx_shape}",
            "TOPK_IDX_SHAPE",
        )
    if record.token_count != idx_shape[0]:
        return ValidationResult.failed(
            f"token_count mismatch: record.token_count={record.token_count}, topk_idx rows={idx_shape[0]}",
            "TOKEN_COUNT_MISMATCH",
        )
    if record.top_k != idx_shape[1]:
        return ValidationResult.failed(
            f"top_k mismatch: record.top_k={record.top_k}, topk_idx columns={idx_shape[1]}",
            "TOP_K_MISMATCH",
        )
    if record.top_k <= 0:
        return ValidationResult.failed("top_k must be > 0", "TOP_K_INVALID")
    if record.num_experts <= 0:
        return ValidationResult.failed("num_experts must be > 0", "NUM_EXPERTS_INVALID")

    min_idx = tensor_min(record.topk_idx)
    max_idx = tensor_max(record.topk_idx)
    if min_idx is not None and min_idx < 0:
        return ValidationResult.failed("topk_idx contains negative expert index", "EXPERT_INDEX_NEGATIVE")
    if max_idx is not None and max_idx >= record.num_experts:
        return ValidationResult.failed(
            f"topk_idx contains value >= num_experts ({record.num_experts})",
            "EXPERT_INDEX_OUT_OF_RANGE",
        )

    if requires_weights(record.replay_level):
        if record.topk_weights is None:
            return ValidationResult.failed(
                f"{record.replay_level.name} requires topk_weights", "WEIGHTS_MISSING"
            )

    if record.topk_weights is not None:
        weight_shape = tensor_shape(record.topk_weights)
        if weight_shape != idx_shape:
            return ValidationResult.failed(
                f"topk_weights shape {weight_shape} must match topk_idx shape {idx_shape}",
                "WEIGHTS_SHAPE_MISMATCH",
            )
        if not tensor_all_finite(record.topk_weights):
            return ValidationResult.failed("topk_weights contains NaN or Inf", "WEIGHTS_NOT_FINITE")

    return ValidationResult.passed_result()


def validate_compatibility(record: RouteRecord, context: RuntimeContext) -> ValidationResult:
    """Validate that a record can safely replay at a runtime context."""

    record_result = validate_record(record)
    if not record_result.passed:
        return record_result

    if record.abi_version != context.abi_version:
        return ValidationResult.failed(
            f"abi_version mismatch: record={record.abi_version}, context={context.abi_version}",
            "ABI_VERSION_MISMATCH",
        )
    if record.layer_id != context.layer_id:
        return ValidationResult.failed(
            f"layer_id mismatch: record={record.layer_id}, context={context.layer_id}",
            "LAYER_ID_MISMATCH",
        )
    if str(record.phase) != str(context.phase):
        return ValidationResult.failed(
            f"phase mismatch: record={record.phase}, context={context.phase}",
            "PHASE_MISMATCH",
        )
    if record.step_id is not None and context.step_id is not None and record.step_id != context.step_id:
        return ValidationResult.failed(
            f"step_id mismatch: record={record.step_id}, context={context.step_id}",
            "STEP_ID_MISMATCH",
        )
    if record.token_count != context.token_count:
        return ValidationResult.failed(
            f"token_count mismatch: record={record.token_count}, context={context.token_count}",
            "TOKEN_COUNT_MISMATCH",
        )
    if record.top_k != context.top_k:
        return ValidationResult.failed(
            f"top_k mismatch: record={record.top_k}, context={context.top_k}",
            "TOP_K_MISMATCH",
        )
    if record.num_experts != context.num_experts:
        return ValidationResult.failed(
            f"num_experts mismatch: record={record.num_experts}, context={context.num_experts}",
            "NUM_EXPERTS_MISMATCH",
        )
    if record.expert_namespace != context.expert_namespace:
        return ValidationResult.failed("expert_namespace mismatch", "EXPERT_NAMESPACE_MISMATCH")

    identity_result = _validate_token_identity(record, context)
    if not identity_result.passed:
        return identity_result

    if requires_weights(context.replay_level):
        if record.replay_level == ReplayLevel.R1 or record.topk_weights is None:
            return ValidationResult.failed(
                "weight semantics mismatch: context requires weighted replay but record is index-only",
                "WEIGHTS_REQUIRED_BY_CONTEXT",
            )
        if record.weight_semantics != context.weight_semantics:
            return ValidationResult.failed(
                "weight semantics mismatch", "WEIGHT_SEMANTICS_MISMATCH"
            )

    return ValidationResult.passed_result()


def validate_tensor_dtypes(record: RouteRecord, context: RuntimeContext) -> ValidationResult:
    """Validate tensor dtype expectations for fail-closed policy decisions."""

    idx_dtype = tensor_dtype_name(record.topk_idx)
    if context.index_dtype and idx_dtype and idx_dtype != context.index_dtype:
        return ValidationResult.failed(
            f"dtype cast required for topk_idx: record={idx_dtype}, context={context.index_dtype}",
            "INDEX_DTYPE_MISMATCH",
        )
    if record.topk_weights is not None:
        weight_dtype = tensor_dtype_name(record.topk_weights)
        if context.weight_dtype and weight_dtype and weight_dtype != context.weight_dtype:
            return ValidationResult.failed(
                f"dtype cast required for topk_weights: record={weight_dtype}, context={context.weight_dtype}",
                "WEIGHT_DTYPE_MISMATCH",
            )
    return ValidationResult.passed_result()


def _validate_token_identity(record: RouteRecord, context: RuntimeContext) -> ValidationResult:
    paired_fields = (
        ("request_ids", record.request_ids, context.request_ids),
        ("sequence_ids", record.sequence_ids, context.sequence_ids),
        ("token_positions", record.token_positions, context.token_positions),
        ("block_ids", record.block_ids, context.block_ids),
        ("token_order", record.token_order, context.token_order),
    )
    for name, record_value, context_value in paired_fields:
        if record_value is not None and context_value is not None and not tensor_equal(
            record_value, context_value
        ):
            return ValidationResult.failed(f"{name} mismatch", f"{name.upper()}_MISMATCH")
    return ValidationResult.passed_result()
