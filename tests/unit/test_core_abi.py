"""Unit tests for RouteForge core ABI: RouteRecord, RuntimeContext, validation, errors, enums.

These tests define the expected contract for:
  - routeforge.core.enums       (ReplayLevel, ReplayMode)
  - routeforge.core.errors      (hierarchical validation / policy errors)
  - routeforge.core.route_record (RouteRecord dataclass)
  - routeforge.core.context     (RuntimeContext dataclass)
  - routeforge.core.capabilities (capability queries for replay levels)
  - routeforge.core.validation  (validate_record, validate_compatibility)

Production code does not exist yet; these tests serve as the executable spec.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Expected imports – will raise ImportError until production code is written
# ---------------------------------------------------------------------------

from routeforge.core.enums import ReplayLevel, ReplayMode
from routeforge.core.errors import (
    CompatibilityError,
    NumExpertsError,
    ReplayPolicyError,
    RouteForgeError,
    ShapeMismatchError,
    TopKMismatchError,
    ValidationError,
    WeightSemanticsError,
)
from routeforge.core.route_record import RouteRecord
from routeforge.core.context import RuntimeContext
from routeforge.core.capabilities import requires_weights, requires_dispatch
from routeforge.core.validation import ValidationResult, validate_record, validate_compatibility


# ============================================================================
# Helpers
# ============================================================================

def _make_idx(token_count: int = 4, top_k: int = 2, num_experts: int = 8) -> np.ndarray:
    """Create a valid topk_idx array of shape [token_count, top_k]."""
    rng = np.random.default_rng(42)
    return rng.integers(0, num_experts, size=(token_count, top_k), dtype=np.int16)


def _make_weights(token_count: int = 4, top_k: int = 2) -> np.ndarray:
    """Create a valid topk_weights array of shape [token_count, top_k]."""
    rng = np.random.default_rng(42)
    w = rng.random((token_count, top_k)).astype(np.float32)
    # normalise rows so they sum to 1
    w /= w.sum(axis=1, keepdims=True)
    return w


def _make_record(
    token_count: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    layer_id: int = 0,
    replay_level: ReplayLevel = ReplayLevel.R2,
    topk_idx: np.ndarray | None = None,
    topk_weights: np.ndarray | None = None,
) -> RouteRecord:
    """Build a well-formed RouteRecord (R2 by default)."""
    if topk_idx is None:
        topk_idx = _make_idx(token_count, top_k, num_experts)
    if topk_weights is None and replay_level in (ReplayLevel.R2,):
        topk_weights = _make_weights(token_count, top_k)
    return RouteRecord(
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        layer_id=layer_id,
        num_experts=num_experts,
        top_k=top_k,
        token_count=token_count,
        replay_level=replay_level,
    )


def _make_context(
    token_count: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    layer_id: int = 0,
) -> RuntimeContext:
    return RuntimeContext(
        token_count=token_count,
        layer_id=layer_id,
        top_k=top_k,
        num_experts=num_experts,
    )


# ============================================================================
# 1. Enums
# ============================================================================

class TestReplayLevel:
    """ReplayLevel enum must define the standard MoE replay tiers."""

    def test_has_r0_through_r5(self) -> None:
        names = {level.name for level in ReplayLevel}
        for expected in ("R0", "R1", "R2", "R3", "R4", "R5"):
            assert expected in names, f"ReplayLevel missing {expected}"

    def test_r1_is_index_only(self) -> None:
        """R1 (Index Replay) must not require weights."""
        assert not requires_weights(ReplayLevel.R1)

    def test_r2_requires_weights(self) -> None:
        """R2 (Weighted Replay) must require weights."""
        assert requires_weights(ReplayLevel.R2)

    def test_r3_requires_dispatch(self) -> None:
        """R3 (Dispatch Replay) must require dispatch metadata."""
        assert requires_dispatch(ReplayLevel.R3)

    def test_r0_observe_never_requires_weights(self) -> None:
        assert not requires_weights(ReplayLevel.R0)


class TestReplayMode:
    """ReplayMode enum for the runtime state machine."""

    def test_has_all_modes(self) -> None:
        names = {m.name for m in ReplayMode}
        for expected in ("OFF", "RECORD", "REPLAY", "POLICY", "PLAN_ONLY"):
            assert expected in names, f"ReplayMode missing {expected}"


# ============================================================================
# 2. Error hierarchy
# ============================================================================

class TestErrorHierarchy:
    """All RouteForge errors must form a coherent hierarchy for catch-by-base."""

    def test_validation_error_is_routeforge_error(self) -> None:
        assert issubclass(ValidationError, RouteForgeError)

    def test_shape_mismatch_is_validation_error(self) -> None:
        assert issubclass(ShapeMismatchError, ValidationError)

    def test_topk_mismatch_is_validation_error(self) -> None:
        assert issubclass(TopKMismatchError, ValidationError)

    def test_num_experts_error_is_validation_error(self) -> None:
        assert issubclass(NumExpertsError, ValidationError)

    def test_weight_semantics_error_is_validation_error(self) -> None:
        assert issubclass(WeightSemanticsError, ValidationError)

    def test_compatibility_error_is_validation_error(self) -> None:
        assert issubclass(CompatibilityError, ValidationError)

    def test_replay_policy_error_is_routeforge_error(self) -> None:
        assert issubclass(ReplayPolicyError, RouteForgeError)

    def test_catch_all_validation_with_base(self) -> None:
        """Catching ValidationError must catch all specific subtypes."""
        for exc_cls in (
            ShapeMismatchError,
            TopKMismatchError,
            NumExpertsError,
            WeightSemanticsError,
            CompatibilityError,
        ):
            try:
                raise exc_cls("test")
            except ValidationError:
                pass  # expected
            else:
                pytest.fail(f"{exc_cls.__name__} not caught by ValidationError")


# ============================================================================
# 3. RouteRecord validation
# ============================================================================

class TestRouteRecordValidation:
    """validate_record must enforce structural invariants of RouteRecord."""

    # --- topk_idx / topk_weights shape consistency ---

    def test_valid_r2_record_passes(self) -> None:
        result = validate_record(_make_record(replay_level=ReplayLevel.R2))
        assert result.passed, f"Valid R2 record should pass, got: {result.reason}"

    def test_valid_r1_record_without_weights_passes(self) -> None:
        result = validate_record(_make_record(replay_level=ReplayLevel.R1, topk_weights=None))
        assert result.passed, f"Valid R1 record should pass, got: {result.reason}"

    def test_topk_idx_topk_weights_row_mismatch(self) -> None:
        """topk_idx [4,2] vs topk_weights [3,2] → shape mismatch."""
        idx = _make_idx(token_count=4, top_k=2)
        weights = _make_weights(token_count=3, top_k=2)  # wrong row count
        record = _make_record(token_count=4, top_k=2, topk_idx=idx, topk_weights=weights)
        result = validate_record(record)
        assert not result.passed
        assert "shape" in result.reason.lower()

    def test_topk_idx_topk_weights_col_mismatch(self) -> None:
        """topk_idx [4,2] vs topk_weights [4,3] → shape mismatch."""
        idx = _make_idx(token_count=4, top_k=2)
        weights = _make_weights(token_count=4, top_k=3)  # wrong col count
        record = _make_record(token_count=4, top_k=2, topk_idx=idx, topk_weights=weights)
        result = validate_record(record)
        assert not result.passed
        assert "shape" in result.reason.lower()

    def test_r2_missing_weights_fails(self) -> None:
        """R2 requires weights; None should fail."""
        record = RouteRecord(
            topk_idx=_make_idx(),
            topk_weights=None,
            layer_id=0,
            num_experts=8,
            top_k=2,
            token_count=4,
            replay_level=ReplayLevel.R2,
        )
        result = validate_record(record)
        assert not result.passed
        assert "weight" in result.reason.lower()

    # --- top_k consistency ---

    def test_topk_idx_columns_match_top_k(self) -> None:
        """topk_idx.shape[1] must equal record.top_k."""
        idx = _make_idx(token_count=4, top_k=2)
        record = _make_record(token_count=4, top_k=3, topk_idx=idx)  # top_k=3 but idx has 2 cols
        result = validate_record(record)
        assert not result.passed
        assert "top_k" in result.reason.lower()

    # --- num_experts bounds ---

    def test_num_experts_positive(self) -> None:
        """num_experts must be > 0."""
        record = _make_record(num_experts=0, topk_idx=np.zeros((4, 2), dtype=np.int16))
        result = validate_record(record)
        assert not result.passed
        assert "num_experts" in result.reason.lower()

    def test_topk_idx_values_within_num_experts(self) -> None:
        """All values in topk_idx must be < num_experts."""
        idx = np.array([[0, 9]], dtype=np.int16)  # 9 >= num_experts=8
        record = _make_record(
            token_count=1, top_k=2, num_experts=8,
            topk_idx=idx, topk_weights=_make_weights(1, 2),
        )
        result = validate_record(record)
        assert not result.passed
        assert "num_experts" in result.reason.lower()

    def test_topk_idx_negative_values_rejected(self) -> None:
        """Negative expert indices are invalid."""
        idx = np.array([[-1, 3]], dtype=np.int16)
        record = _make_record(
            token_count=1, top_k=2, num_experts=8,
            topk_idx=idx, topk_weights=_make_weights(1, 2),
        )
        result = validate_record(record)
        assert not result.passed

    # --- token_count consistency ---

    def test_topk_idx_rows_match_token_count(self) -> None:
        """topk_idx.shape[0] must equal record.token_count."""
        idx = _make_idx(token_count=4, top_k=2)
        record = _make_record(token_count=8, top_k=2, topk_idx=idx)  # claims 8 tokens but has 4
        result = validate_record(record)
        assert not result.passed
        assert "token_count" in result.reason.lower()

    # --- ValidationResult structure ---

    def test_validation_result_passed_has_no_reason(self) -> None:
        result = validate_record(_make_record())
        if result.passed:
            assert result.reason is None

    def test_validation_result_failed_has_reason(self) -> None:
        record = _make_record(num_experts=0, topk_idx=np.zeros((4, 2), dtype=np.int16))
        result = validate_record(record)
        assert not result.passed
        assert result.reason is not None
        assert len(result.reason) > 0


# ============================================================================
# 4. RuntimeContext + RouteRecord compatibility validation
# ============================================================================

class TestCompatibilityValidation:
    """validate_compatibility must fail-closed on any mismatch."""

    def test_compatible_record_and_context_pass(self) -> None:
        record = _make_record(token_count=4, top_k=2, num_experts=8, layer_id=3)
        ctx = _make_context(token_count=4, top_k=2, num_experts=8, layer_id=3)
        result = validate_compatibility(record, ctx)
        assert result.passed

    # --- token_count mismatch ---

    def test_token_count_mismatch_fails_closed(self) -> None:
        record = _make_record(token_count=4, top_k=2, num_experts=8, layer_id=0)
        ctx = _make_context(token_count=8, top_k=2, num_experts=8, layer_id=0)
        result = validate_compatibility(record, ctx)
        assert not result.passed
        assert "token_count" in result.reason.lower()

    # --- layer_id mismatch ---

    def test_layer_id_mismatch_fails_closed(self) -> None:
        record = _make_record(layer_id=0)
        ctx = _make_context(layer_id=5)
        result = validate_compatibility(record, ctx)
        assert not result.passed
        assert "layer_id" in result.reason.lower()

    # --- top_k mismatch ---

    def test_top_k_mismatch_fails_closed(self) -> None:
        record = _make_record(top_k=2)
        ctx = _make_context(top_k=4)
        result = validate_compatibility(record, ctx)
        assert not result.passed
        assert "top_k" in result.reason.lower()

    # --- num_experts mismatch ---

    def test_num_experts_mismatch_fails_closed(self) -> None:
        record = _make_record(num_experts=8)
        ctx = _make_context(num_experts=128)
        result = validate_compatibility(record, ctx)
        assert not result.passed
        assert "num_experts" in result.reason.lower()

    # --- weight semantics mismatch ---

    def test_weight_semantics_mismatch_fails_closed(self) -> None:
        """RuntimeContext expects R2 (weights) but record is R1 (index only)."""
        record = _make_record(replay_level=ReplayLevel.R1, topk_weights=None)
        ctx = _make_context()  # default expects R2 weights
        result = validate_compatibility(record, ctx)
        assert not result.passed
        assert "weight" in result.reason.lower()

    # --- all fields must match for pass ---

    def test_mixed_mismatches_report_first(self) -> None:
        """When multiple fields mismatch, result is still fail-closed."""
        record = _make_record(token_count=4, top_k=2, num_experts=8, layer_id=0)
        ctx = _make_context(token_count=8, top_k=4, num_experts=128, layer_id=5)
        result = validate_compatibility(record, ctx)
        assert not result.passed
        # reason must mention at least one mismatched field
        reason_lower = result.reason.lower()
        assert any(kw in reason_lower for kw in ("token_count", "layer_id", "top_k", "num_experts"))


# ============================================================================
# 5. RouteRecord dataclass invariants
# ============================================================================

class TestRouteRecordDataclass:
    """RouteRecord should be an immutable, well-behaved dataclass."""

    def test_frozen(self) -> None:
        """RouteRecord should be frozen (immutable)."""
        record = _make_record()
        with pytest.raises(AttributeError):
            record.layer_id = 99  # type: ignore[misc]

    def test_equality(self) -> None:
        r1 = _make_record(token_count=2, top_k=2, num_experts=8, layer_id=0)
        r2 = _make_record(token_count=2, top_k=2, num_experts=8, layer_id=0)
        assert r1 == r2

    def test_inequality(self) -> None:
        r1 = _make_record(layer_id=0)
        r2 = _make_record(layer_id=1)
        assert r1 != r2

    def test_hashable(self) -> None:
        """Frozen dataclass should be hashable for use in sets/dicts."""
        record = _make_record()
        # Must not raise
        hash(record)


# ============================================================================
# 6. RuntimeContext dataclass invariants
# ============================================================================

class TestRuntimeContextDataclass:
    """RuntimeContext should be an immutable, well-behaved dataclass."""

    def test_frozen(self) -> None:
        ctx = _make_context()
        with pytest.raises(AttributeError):
            ctx.layer_id = 99  # type: ignore[misc]

    def test_equality(self) -> None:
        c1 = _make_context(token_count=4, top_k=2, num_experts=8, layer_id=0)
        c2 = _make_context(token_count=4, top_k=2, num_experts=8, layer_id=0)
        assert c1 == c2


# ============================================================================
# 7. Capabilities helpers
# ============================================================================

class TestCapabilities:
    """Capability queries must correctly reflect replay-level requirements."""

    @pytest.mark.parametrize(
        "level,needs_weights",
        [
            (ReplayLevel.R0, False),
            (ReplayLevel.R1, False),
            (ReplayLevel.R2, True),
            (ReplayLevel.R3, True),  # dispatch replay needs weights for combine
            (ReplayLevel.R4, True),
            (ReplayLevel.R5, True),
        ],
    )
    def test_requires_weights(self, level: ReplayLevel, needs_weights: bool) -> None:
        assert requires_weights(level) is needs_weights

    @pytest.mark.parametrize(
        "level,needs_dispatch",
        [
            (ReplayLevel.R0, False),
            (ReplayLevel.R1, False),
            (ReplayLevel.R2, False),
            (ReplayLevel.R3, True),
            (ReplayLevel.R4, True),
            (ReplayLevel.R5, True),
        ],
    )
    def test_requires_dispatch(self, level: ReplayLevel, needs_dispatch: bool) -> None:
        assert requires_dispatch(level) is needs_dispatch


# ============================================================================
# 8. ValidationResult
# ============================================================================

class TestValidationResult:
    """ValidationResult must support ergonomic pass/fail patterns."""

    def test_pass_factory(self) -> None:
        result = ValidationResult.passed()
        assert result.passed is True
        assert result.reason is None

    def test_fail_factory(self) -> None:
        result = ValidationResult.failed("something broke")
        assert result.passed is False
        assert result.reason == "something broke"

    def test_bool_conversion(self) -> None:
        """ValidationResult should be truthy when passed."""
        assert bool(ValidationResult.passed()) is True
        assert bool(ValidationResult.failed("err")) is False
