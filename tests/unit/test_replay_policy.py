"""Unit tests for RouteForge ReplayPolicy: strict defaults, fail-closed on mismatch,
downgrade semantics when allow_downgrade is enabled.

These tests define the expected contract for:
  - routeforge.core.replay_policy (ReplayPolicy dataclass, apply_policy)
  - routeforge.core.errors        (ReplayPolicyError)
  - routeforge.core.enums         (ReplayLevel)

Production code does not exist yet; these tests serve as the executable spec.
"""

from __future__ import annotations

import numpy as np
import pytest

from routeforge.core.enums import ReplayLevel
from routeforge.core.errors import ReplayPolicyError
from routeforge.core.route_record import RouteRecord
from routeforge.core.context import RuntimeContext
from routeforge.core.replay_policy import ReplayPolicy, apply_policy
from routeforge.core.validation import validate_compatibility


# ============================================================================
# Helpers
# ============================================================================

def _make_idx(token_count: int = 4, top_k: int = 2, num_experts: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, num_experts, size=(token_count, top_k), dtype=np.int16)


def _make_weights(token_count: int = 4, top_k: int = 2) -> np.ndarray:
    rng = np.random.default_rng(42)
    w = rng.random((token_count, top_k)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    return w


def _make_r2_record(
    token_count: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    layer_id: int = 0,
    topk_idx: np.ndarray | None = None,
    topk_weights: np.ndarray | None = None,
) -> RouteRecord:
    if topk_idx is None:
        topk_idx = _make_idx(token_count, top_k, num_experts)
    if topk_weights is None:
        topk_weights = _make_weights(token_count, top_k)
    return RouteRecord(
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        layer_id=layer_id,
        num_experts=num_experts,
        top_k=top_k,
        token_count=token_count,
        replay_level=ReplayLevel.R2,
    )


def _make_r1_record(
    token_count: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    layer_id: int = 0,
    topk_idx: np.ndarray | None = None,
) -> RouteRecord:
    if topk_idx is None:
        topk_idx = _make_idx(token_count, top_k, num_experts)
    return RouteRecord(
        topk_idx=topk_idx,
        topk_weights=None,
        layer_id=layer_id,
        num_experts=num_experts,
        top_k=top_k,
        token_count=token_count,
        replay_level=ReplayLevel.R1,
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
# 1. ReplayPolicy defaults
# ============================================================================

class TestReplayPolicyDefaults:
    """ReplayPolicy must default to strict mode, disallowing any relaxation."""

    def test_default_strict_is_true(self) -> None:
        policy = ReplayPolicy()
        assert policy.strict is True

    def test_default_allow_downgrade_is_false(self) -> None:
        policy = ReplayPolicy()
        assert policy.allow_downgrade is False

    def test_default_allow_partial_is_false(self) -> None:
        policy = ReplayPolicy()
        assert policy.allow_partial is False

    def test_default_allow_cast_is_false(self) -> None:
        policy = ReplayPolicy()
        assert policy.allow_cast is False

    def test_all_relaxation_disabled_by_default(self) -> None:
        """By default, no form of replay relaxation is permitted."""
        policy = ReplayPolicy()
        assert policy.strict
        assert not policy.allow_downgrade
        assert not policy.allow_partial
        assert not policy.allow_cast


# ============================================================================
# 2. Strict mode: partial / cast / downgrade forbidden
# ============================================================================

class TestStrictPolicyProhibitsRelaxation:
    """Under strict policy (default), partial replay, type cast, and downgrade
    must all be rejected with ReplayPolicyError."""

    def test_strict_rejects_partial_replay(self) -> None:
        """If the record has fewer tokens than the context, strict mode must reject."""
        policy = ReplayPolicy()  # strict=True by default
        record = _make_r2_record(token_count=2)  # only 2 tokens
        ctx = _make_context(token_count=4)        # context expects 4 tokens
        with pytest.raises(ReplayPolicyError, match="partial"):
            apply_policy(policy, record, ctx)

    def test_strict_rejects_cast(self) -> None:
        """If the record dtype differs from context expectation, strict mode must reject."""
        policy = ReplayPolicy()
        idx = _make_idx(token_count=4, top_k=2).astype(np.int32)  # not int16
        weights = _make_weights(4, 2).astype(np.float64)  # not float32
        record = _make_r2_record(topk_idx=idx, topk_weights=weights)
        ctx = _make_context()
        with pytest.raises(ReplayPolicyError, match="cast"):
            apply_policy(policy, record, ctx)

    def test_strict_rejects_downgrade_r2_to_r1(self) -> None:
        """If context expects R2 but record only has R1 weights, strict rejects downgrade."""
        policy = ReplayPolicy()
        record = _make_r1_record()  # R1, no weights
        ctx = _make_context()       # expects R2 (weights)
        with pytest.raises(ReplayPolicyError, match="downgrade"):
            apply_policy(policy, record, ctx)

    def test_strict_rejects_weight_mismatch(self) -> None:
        """When topk_idx matches but weights are absent in R2 context, strict fails."""
        policy = ReplayPolicy()
        # Create an R2 record where weights are None (semantically invalid for R2)
        idx = _make_idx()
        record = RouteRecord(
            topk_idx=idx,
            topk_weights=None,
            layer_id=0,
            num_experts=8,
            top_k=2,
            token_count=4,
            replay_level=ReplayLevel.R2,
        )
        ctx = _make_context()
        with pytest.raises((ReplayPolicyError, ValueError)):
            apply_policy(policy, record, ctx)


# ============================================================================
# 3. allow_downgrade: R2 → R1 weight mismatch downgrade
# ============================================================================

class TestDowngradeSemantics:
    """When allow_downgrade=True, an R2 context may accept an R1 record by
    dropping weight requirements (downgrading from R2 to R1)."""

    def test_downgrade_r2_to_r1_when_allowed(self) -> None:
        """With allow_downgrade=True, R1 record should be accepted for R2 context."""
        policy = ReplayPolicy(allow_downgrade=True)
        record = _make_r1_record()  # no weights, replay_level=R1
        ctx = _make_context()       # default R2 expectations
        result = apply_policy(policy, record, ctx)
        # The returned record should be R1 (downgraded from R2 expectation)
        assert result.replay_level == ReplayLevel.R1
        assert result.topk_weights is None

    def test_downgrade_preserves_topk_idx(self) -> None:
        """Downgrade must not alter the expert index data."""
        policy = ReplayPolicy(allow_downgrade=True)
        idx = _make_idx()
        record = _make_r1_record(topk_idx=idx)
        ctx = _make_context()
        result = apply_policy(policy, record, ctx)
        np.testing.assert_array_equal(result.topk_idx, idx)

    def test_downgrade_still_requires_topk_idx_match(self) -> None:
        """Downgrade does not waive shape/field compatibility for topk_idx."""
        policy = ReplayPolicy(allow_downgrade=True)
        record = _make_r1_record(top_k=2, num_experts=8)
        ctx = _make_context(top_k=4, num_experts=8)  # top_k mismatch
        with pytest.raises((ReplayPolicyError, ValueError)):
            apply_policy(policy, record, ctx)

    def test_downgrade_does_not_allow_partial(self) -> None:
        """allow_downgrade alone does not enable partial replay."""
        policy = ReplayPolicy(allow_downgrade=True, allow_partial=False)
        record = _make_r1_record(token_count=2)
        ctx = _make_context(token_count=4)
        with pytest.raises(ReplayPolicyError, match="partial"):
            apply_policy(policy, record, ctx)

    def test_downgrade_r2_record_missing_weights(self) -> None:
        """An R2 record with None weights can be downgraded to R1 when allowed."""
        policy = ReplayPolicy(allow_downgrade=True)
        idx = _make_idx()
        # R2 record with missing weights (semantically broken but structurally present)
        record = RouteRecord(
            topk_idx=idx,
            topk_weights=None,
            layer_id=0,
            num_experts=8,
            top_k=2,
            token_count=4,
            replay_level=ReplayLevel.R2,
        )
        ctx = _make_context()
        result = apply_policy(policy, record, ctx)
        assert result.replay_level == ReplayLevel.R1
        assert result.topk_weights is None

    def test_downgrade_strictness_still_enforced(self) -> None:
        """allow_downgrade=True but strict=True still enforces non-downgrade checks."""
        policy = ReplayPolicy(strict=True, allow_downgrade=True)
        # token_count mismatch is not a downgrade issue → should still fail
        record = _make_r1_record(token_count=2, num_experts=8)
        ctx = _make_context(token_count=4, num_experts=8)
        with pytest.raises(ReplayPolicyError):
            apply_policy(policy, record, ctx)


# ============================================================================
# 4. Combined policy flags
# ============================================================================

class TestPolicyFlagCombinations:
    """Various combinations of policy flags must produce correct behaviour."""

    def test_all_relaxations_enabled(self) -> None:
        """With all relaxation flags True, previously-failing scenarios should succeed."""
        policy = ReplayPolicy(
            strict=False,
            allow_downgrade=True,
            allow_partial=True,
            allow_cast=True,
        )
        # R1 record for R2 context, different token count, different dtype
        idx = _make_idx(token_count=2).astype(np.int32)
        record = _make_r1_record(token_count=2, topk_idx=idx)
        ctx = _make_context(token_count=2)
        # Should not raise
        result = apply_policy(policy, record, ctx)
        assert result is not None

    def test_allow_partial_without_downgrade(self) -> None:
        """allow_partial alone does not enable downgrade."""
        policy = ReplayPolicy(allow_partial=True, allow_downgrade=False)
        record = _make_r1_record()
        ctx = _make_context()
        # R1 for R2 context is a downgrade, not just partial
        with pytest.raises(ReplayPolicyError, match="downgrade"):
            apply_policy(policy, record, ctx)

    def test_explicit_strict_false_with_downgrade(self) -> None:
        """strict=False + allow_downgrade=True should accept R1→R2 downgrade."""
        policy = ReplayPolicy(strict=False, allow_downgrade=True)
        record = _make_r1_record()
        ctx = _make_context()
        result = apply_policy(policy, record, ctx)
        assert result.replay_level == ReplayLevel.R1


# ============================================================================
# 5. apply_policy return type and invariants
# ============================================================================

class TestApplyPolicyReturnInvariants:
    """apply_policy must return a RouteRecord that satisfies structural invariants."""

    def test_returned_record_is_route_record(self) -> None:
        policy = ReplayPolicy(allow_downgrade=True)
        record = _make_r1_record()
        ctx = _make_context()
        result = apply_policy(policy, record, ctx)
        assert isinstance(result, RouteRecord)

    def test_returned_record_has_valid_topk_idx(self) -> None:
        policy = ReplayPolicy(allow_downgrade=True)
        record = _make_r1_record()
        ctx = _make_context()
        result = apply_policy(policy, record, ctx)
        assert result.topk_idx is not None
        assert result.topk_idx.shape[0] == ctx.token_count
        assert result.topk_idx.shape[1] == ctx.top_k

    def test_downgraded_record_topk_weights_is_none(self) -> None:
        policy = ReplayPolicy(allow_downgrade=True)
        record = _make_r1_record()
        ctx = _make_context()
        result = apply_policy(policy, record, ctx)
        assert result.topk_weights is None
        assert result.replay_level == ReplayLevel.R1
