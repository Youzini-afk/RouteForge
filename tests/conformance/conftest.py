"""Reusable fake backend fixtures for conformance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from routeforge.core import BackendCapabilities, ReplayLevel, ReplayPolicy, RouteRecord, RuntimeContext


@dataclass(frozen=True)
class FakeBackendSpec:
    backend_id: str = "fake"
    supported_replay_levels: frozenset[ReplayLevel] = field(
        default_factory=lambda: frozenset({ReplayLevel.R0, ReplayLevel.R1, ReplayLevel.R2})
    )
    supports_index_replay: bool = True
    supports_weight_replay: bool = True
    supports_shared_expert: bool = False
    supported_index_dtypes: frozenset[str] = field(default_factory=lambda: frozenset({"int16"}))
    supported_weight_dtypes: frozenset[str] = field(default_factory=lambda: frozenset({"float32"}))
    require_token_identity: bool = True
    allow_downgrade: bool = False
    allow_partial: bool = False
    allow_cast: bool = False

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            supported_replay_levels=self.supported_replay_levels,
            supports_index_replay=self.supports_index_replay,
            supports_weight_replay=self.supports_weight_replay,
            supports_shared_expert=self.supports_shared_expert,
            supported_index_dtypes=self.supported_index_dtypes,
            supported_weight_dtypes=self.supported_weight_dtypes,
        )

    def policy(self) -> ReplayPolicy:
        return ReplayPolicy(
            require_token_identity=self.require_token_identity,
            allow_downgrade=self.allow_downgrade,
            allow_partial=self.allow_partial,
            allow_cast=self.allow_cast,
        )


def fake_record(**overrides: Any) -> RouteRecord:
    token_count = int(overrides.pop("token_count", 4))
    top_k = int(overrides.pop("top_k", 2))
    num_experts = int(overrides.pop("num_experts", 8))
    replay_level = overrides.pop("replay_level", ReplayLevel.R2)
    idx_dtype = overrides.pop("idx_dtype", "int16")
    weight_dtype = overrides.pop("weight_dtype", "float32")
    topk_weights = overrides.pop("topk_weights", ...)
    topk_idx = overrides.pop("topk_idx", None)
    if topk_idx is None:
        base = np.arange(token_count * top_k).reshape(token_count, top_k) % num_experts
        topk_idx = base.astype(np.dtype(idx_dtype))
    if topk_weights is ...:
        weights = None if replay_level == ReplayLevel.R1 else np.full((token_count, top_k), 1 / top_k, dtype=np.dtype(weight_dtype))
    else:
        weights = topk_weights
    kwargs: dict[str, Any] = dict(
        topk_idx=topk_idx,
        topk_weights=weights,
        layer_id=0,
        num_experts=num_experts,
        top_k=top_k,
        token_count=token_count,
        replay_level=replay_level,
        step_id=0,
        token_positions=np.arange(token_count, dtype=np.int64),
        token_order=np.arange(token_count, dtype=np.int64),
    )
    kwargs.update(overrides)
    return RouteRecord(**kwargs)


def fake_context(**overrides: Any) -> RuntimeContext:
    token_count = int(overrides.pop("token_count", 4))
    kwargs: dict[str, Any] = dict(
        token_count=token_count,
        layer_id=0,
        top_k=2,
        num_experts=8,
        replay_level=ReplayLevel.R2,
        backend_id="fake",
        step_id=0,
        token_positions=np.arange(token_count, dtype=np.int64),
        token_order=np.arange(token_count, dtype=np.int64),
        index_dtype="int16",
        weight_dtype="float32",
    )
    kwargs.update(overrides)
    return RuntimeContext(**kwargs)
