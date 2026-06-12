"""Serving RuntimeContext helper tests."""

from __future__ import annotations

import numpy as np

from routeforge.core import ReplayDecision, ReplayLevel, ReplayPolicy, RouteRecord, decide_replay_request
from routeforge.core.capabilities import BackendCapabilities
from routeforge.integrations import ServingTokenBatch


def test_serving_token_batch_builds_runtime_context() -> None:
    batch = ServingTokenBatch(
        token_count=2,
        layer_id=3,
        top_k=2,
        num_experts=8,
        step_id=9,
        request_ids=np.array([10, 11]),
        sequence_ids=np.array([0, 0]),
        token_positions=np.array([4, 5]),
        token_order=np.array([0, 1]),
        backend_id="vllm",
        model_config_digest="digest",
    )
    ctx = batch.to_runtime_context(replay_level=ReplayLevel.R2)
    assert ctx.backend_id == "vllm"
    assert ctx.layer_id == 3
    assert ctx.step_id == 9
    assert ctx.model_config_digest == "digest"


def test_serving_context_can_prove_token_identity_for_guard() -> None:
    batch = ServingTokenBatch(
        token_count=2,
        layer_id=0,
        top_k=2,
        num_experts=8,
        step_id=1,
        request_ids=np.array([1, 2]),
        sequence_ids=np.array([0, 0]),
        token_positions=np.array([5, 6]),
        token_order=np.array([0, 1]),
    )
    record = RouteRecord(
        topk_idx=np.array([[0, 1], [2, 3]], dtype=np.int16),
        topk_weights=np.array([[0.5, 0.5], [0.6, 0.4]], dtype=np.float32),
        layer_id=0,
        num_experts=8,
        top_k=2,
        token_count=2,
        replay_level=ReplayLevel.R2,
        step_id=1,
        request_ids=np.array([1, 2]),
        sequence_ids=np.array([0, 0]),
        token_positions=np.array([5, 6]),
        token_order=np.array([0, 1]),
    )
    caps = BackendCapabilities(
        backend_id="serving",
        supported_replay_levels=frozenset({ReplayLevel.R1, ReplayLevel.R2}),
        supports_index_replay=True,
        supports_weight_replay=True,
    )
    result = decide_replay_request(record, batch.to_runtime_context(), ReplayPolicy(), caps)
    assert result.decision == ReplayDecision.REPLAY_R2
