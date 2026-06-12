"""Central replay guard fail-closed tests."""

from __future__ import annotations

import numpy as np
import pytest

from routeforge.core import (
    BackendCapabilities,
    ExpertNamespace,
    ReplayLevel,
    ReplayPolicy,
    RouteRecord,
    RuntimeContext,
    SharedExpertRecord,
    validate_replay_request,
)
from routeforge.core.errors import ReplayPolicyError, UnsupportedReplayLevelError


def _record(**kwargs):
    base = dict(
        topk_idx=np.array([[0, 1], [2, 3]], dtype=np.int16),
        topk_weights=np.array([[0.6, 0.4], [0.7, 0.3]], dtype=np.float32),
        layer_id=0,
        num_experts=8,
        top_k=2,
        token_count=2,
        replay_level=ReplayLevel.R2,
        token_positions=np.array([0, 1], dtype=np.int64),
        token_order=np.array([0, 1], dtype=np.int64),
    )
    base.update(kwargs)
    return RouteRecord(**base)


def _context(**kwargs):
    base = dict(
        token_count=2,
        layer_id=0,
        top_k=2,
        num_experts=8,
        replay_level=ReplayLevel.R2,
        token_positions=np.array([0, 1], dtype=np.int64),
        token_order=np.array([0, 1], dtype=np.int64),
    )
    base.update(kwargs)
    return RuntimeContext(**base)


def _caps(**kwargs):
    base = dict(
        backend_id="test",
        supported_replay_levels=frozenset({ReplayLevel.R0, ReplayLevel.R1, ReplayLevel.R2}),
        supports_index_replay=True,
        supports_weight_replay=True,
        supported_index_dtypes=frozenset({"int16"}),
        supported_weight_dtypes=frozenset({"float32"}),
    )
    base.update(kwargs)
    return BackendCapabilities(**base)


def test_valid_replay_guard_passes() -> None:
    safe = validate_replay_request(_record(), _context(), ReplayPolicy(), _caps())
    assert safe.replay_level == ReplayLevel.R2


def test_missing_token_identity_fails_closed() -> None:
    rec = _record(token_positions=None, token_order=None)
    ctx = _context(token_positions=None, token_order=None)
    with pytest.raises(ReplayPolicyError, match="token identity"):
        validate_replay_request(rec, ctx, ReplayPolicy(), _caps())


def test_backend_unsupported_r2_fails_closed() -> None:
    caps = _caps(supported_replay_levels=frozenset({ReplayLevel.R0, ReplayLevel.R1}), supports_weight_replay=False)
    with pytest.raises(ReplayPolicyError, match="does not support"):
        validate_replay_request(_record(), _context(), ReplayPolicy(), caps)


def test_shared_expert_unsupported_fails_closed() -> None:
    rec = _record(shared_expert=SharedExpertRecord(enabled=True))
    with pytest.raises(ReplayPolicyError, match="shared expert"):
        validate_replay_request(rec, _context(), ReplayPolicy(), _caps())


def test_shared_expert_namespace_unsupported_fails_closed() -> None:
    rec = _record(expert_namespace=ExpertNamespace(shared_expert_count=1))
    ctx = _context(expert_namespace=ExpertNamespace(shared_expert_count=1))
    with pytest.raises(ReplayPolicyError, match="shared expert"):
        validate_replay_request(rec, ctx, ReplayPolicy(), _caps())


@pytest.mark.parametrize("level", [ReplayLevel.R3, ReplayLevel.R4, ReplayLevel.R5])
def test_r3_plus_rejected_in_mvp(level: ReplayLevel) -> None:
    rec = _record(replay_level=level)
    ctx = _context(replay_level=level)
    with pytest.raises(UnsupportedReplayLevelError):
        validate_replay_request(rec, ctx, ReplayPolicy(), _caps())
