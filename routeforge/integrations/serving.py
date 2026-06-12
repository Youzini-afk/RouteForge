"""Serving-runtime context helpers.

These helpers do not integrate with a real serving engine. They provide a small
contract for future vLLM/SGLang adapters to prove token identity before replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from routeforge.core import ReplayLevel, RoutePhase, RuntimeContext


@dataclass(frozen=True)
class ServingTokenBatch:
    token_count: int
    layer_id: int
    top_k: int
    num_experts: int
    step_id: int
    request_ids: Any
    sequence_ids: Any
    token_positions: Any
    token_order: Any
    phase: RoutePhase | str = RoutePhase.DECODE
    backend_id: str = "serving"
    backend_version: str | None = None
    model_arch: str | None = None
    model_config_digest: str | None = None

    def to_runtime_context(self, *, replay_level: ReplayLevel = ReplayLevel.R2) -> RuntimeContext:
        return RuntimeContext(
            token_count=self.token_count,
            layer_id=self.layer_id,
            top_k=self.top_k,
            num_experts=self.num_experts,
            replay_level=replay_level,
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            model_arch=self.model_arch,
            model_config_digest=self.model_config_digest,
            phase=self.phase,
            step_id=self.step_id,
            request_ids=self.request_ids,
            sequence_ids=self.sequence_ids,
            token_positions=self.token_positions,
            token_order=self.token_order,
        )
