"""Conformance tests for replay decision API and stable reason codes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from routeforge.core import (
    ExpertNamespace,
    ReplayDecision,
    ReplayLevel,
    ReplayReasonCode,
    SharedExpertRecord,
    decide_replay_request,
)

from .conftest import FakeBackendSpec, fake_context, fake_record


def _decide(record=None, context=None, spec: FakeBackendSpec | None = None):
    spec = spec or FakeBackendSpec()
    return decide_replay_request(
        record or fake_record(),
        context or fake_context(),
        spec.policy(),
        spec.capabilities(),
    )


def test_decide_valid_r2_returns_replay_r2() -> None:
    result = _decide()
    assert result.decision == ReplayDecision.REPLAY_R2
    assert result.level == ReplayLevel.R2
    assert result.record is not None
    assert result.reason_code is None


def test_decide_valid_r1_returns_replay_r1() -> None:
    rec = fake_record(replay_level=ReplayLevel.R1, topk_weights=None)
    ctx = fake_context(replay_level=ReplayLevel.R1)
    result = _decide(rec, ctx)
    assert result.decision == ReplayDecision.REPLAY_R1
    assert result.level == ReplayLevel.R1


def test_decide_is_frozen_dataclass() -> None:
    result = _decide()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.decision = ReplayDecision.FAIL_CLOSED  # type: ignore[misc]


@pytest.mark.parametrize(
    "record_kwargs,context_kwargs,spec,expected",
    [
        ({}, {"replay_level": ReplayLevel.R3}, None, ReplayReasonCode.UNSUPPORTED_REPLAY_LEVEL),
        ({}, {}, FakeBackendSpec(supported_replay_levels=frozenset({ReplayLevel.R0, ReplayLevel.R1}), supports_weight_replay=False), ReplayReasonCode.BACKEND_LEVEL_NOT_SUPPORTED),
        ({"shared_expert": SharedExpertRecord(enabled=True)}, {}, None, ReplayReasonCode.SHARED_EXPERT_UNSUPPORTED),
        ({"expert_namespace": ExpertNamespace(shared_expert_count=1)}, {"expert_namespace": ExpertNamespace(shared_expert_count=1)}, None, ReplayReasonCode.SHARED_EXPERT_NAMESPACE_UNSUPPORTED),
        ({"step_id": None}, {}, None, ReplayReasonCode.TOKEN_IDENTITY_MISSING_STEP_ID),
        ({"token_positions": None}, {}, None, ReplayReasonCode.TOKEN_IDENTITY_MISSING_POSITIONS),
        ({"token_order": None}, {"token_order": None}, None, ReplayReasonCode.TOKEN_IDENTITY_MISSING_ORDER_OR_REQUEST),
        ({"idx_dtype": "int32"}, {}, None, ReplayReasonCode.INDEX_DTYPE_NOT_SUPPORTED),
        ({"weight_dtype": "float64"}, {}, None, ReplayReasonCode.WEIGHT_DTYPE_NOT_SUPPORTED),
        ({"token_count": 3}, {"token_count": 4}, None, ReplayReasonCode.PARTIAL_REPLAY_DISABLED),
        ({"replay_level": ReplayLevel.R1, "topk_weights": None}, {}, None, ReplayReasonCode.DOWNGRADE_DISABLED),
        ({"abi_version": "bad"}, {}, None, ReplayReasonCode.ABI_VERSION_MISMATCH),
        ({"topk_idx": np.zeros((4,), dtype=np.int16)}, {}, None, ReplayReasonCode.TOPK_IDX_SHAPE),
    ],
)
def test_decide_failure_reason_codes(record_kwargs, context_kwargs, spec, expected) -> None:
    rec = fake_record(**record_kwargs)
    ctx = fake_context(**context_kwargs)
    result = _decide(rec, ctx, spec)
    assert result.decision == ReplayDecision.FAIL_CLOSED
    assert result.reason_code == expected.value
    assert result.reason_code != "ReplayPolicyError"
    assert result.record is None
    assert result.level is None
    assert result.message


def test_decide_downgrade_with_policy_returns_r1() -> None:
    spec = FakeBackendSpec(allow_downgrade=True)
    rec = fake_record(replay_level=ReplayLevel.R1, topk_weights=None)
    result = _decide(rec, fake_context(), spec)
    assert result.decision == ReplayDecision.REPLAY_R1
    assert result.level == ReplayLevel.R1


def test_decide_never_raises_for_bad_inputs() -> None:
    rec = fake_record(topk_idx=np.zeros((4,), dtype=np.int16))
    result = _decide(rec, fake_context())
    assert result.decision == ReplayDecision.FAIL_CLOSED
