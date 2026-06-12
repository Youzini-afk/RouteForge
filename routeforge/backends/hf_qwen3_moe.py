"""Hugging Face Qwen3MoE reference adapter.

This adapter targets the Python reference Qwen3MoE sparse block boundary:

    gate(hidden_states) -> (router_logits, routing_weights, selected_experts)
    experts(hidden_states, selected_experts, routing_weights)

The hook is intentionally placed after `gate` and before `experts`, because
that is where the route decision actually consumed by expert dispatch is
available as top-k expert ids and weights.

Limitations for Phase 3:

- `output_router_logits` is diagnostic/aux-loss output only. It is not the
  replay source and does not expose the top-k ids/weights consumed by experts.
- Replay does not rewrite router logits; if a caller asks HF to return router
  logits, those diagnostics may not describe replayed dispatch.
- This is not a vLLM/SGLang/DeepEP fused runtime adapter.
- Shared expert replay and R3+ dispatch/handle replay are explicitly out of
  scope for this adapter version.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import numpy as np

from routeforge.core.capabilities import BackendCapabilities
from routeforge.core.enums import ReplayLevel, ReplayMode
from routeforge.core.route_record import RouteRecord
from routeforge.core.tensor import tensor_shape

from .base import BackendAdapter


@dataclass(frozen=True)
class SparseMoeBlockRef:
    """Descriptor for a discovered sparse MoE block."""

    layer_id: int
    block: Any
    owner: Any | None = None
    attr_name: str | None = None


class HFQwen3MoeAdapter(BackendAdapter):
    """Reference adapter for HF-style Qwen3MoE sparse blocks."""

    backend_id = "hf_qwen3_moe"


    def __init__(self, model: Any) -> None:
        self.model = model
        self._mode = ReplayMode.OFF
        self._records: dict[int, list[RouteRecord]] = {}
        self._replay_records: dict[int, RouteRecord] = {}
        self._blocks: list[SparseMoeBlockRef] | None = None
        self._original_forwards: dict[int, Callable[..., Any]] = {}
        self._patched = False
        self._record_counter = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            supported_modes=frozenset({ReplayMode.OFF, ReplayMode.RECORD, ReplayMode.REPLAY}),
            supported_replay_levels=frozenset({ReplayLevel.R0, ReplayLevel.R1, ReplayLevel.R2}),
            supports_index_replay=True,
            supports_weight_replay=True,
            supports_route_recording=True,
            supported_index_dtypes=frozenset({"int16", "int32", "int64"}),
            supported_weight_dtypes=frozenset({"float16", "bfloat16", "float32", "float64"}),
            supports_shared_expert=False,
            supports_dispatch_plan=False,
            supports_handle_replay=False,
            supports_prefill=True,
            supports_decode=True,
            supports_batch=True,
            unsafe_cases=(
                "output_router_logits_is_diagnostic_only",
                "aux_loss_router_logits_not_rewritten_during_replay",
                "fused_serving_backends_not_supported",
                "shared_expert_replay_not_supported",
            ),
        )

    def discover_sparse_moe_blocks(self) -> list[SparseMoeBlockRef]:
        if self._blocks is not None:
            return list(self._blocks)

        blocks: list[SparseMoeBlockRef] = []
        for layer_id, layer in _iter_layers(self.model):
            block, attr_name = _extract_sparse_block(layer)
            if block is not None:
                blocks.append(
                    SparseMoeBlockRef(
                        layer_id=getattr(layer, "layer_id", layer_id),
                        block=block,
                        owner=layer,
                        attr_name=attr_name,
                    )
                )
        self._blocks = blocks
        return list(blocks)

    def set_mode(self, mode: ReplayMode) -> None:
        self._mode = mode
        if mode in {ReplayMode.RECORD, ReplayMode.REPLAY, ReplayMode.OFF}:
            self._ensure_patched()
            return
        raise ValueError(f"HFQwen3MoeAdapter does not implement mode {mode}")

    def current_mode(self) -> ReplayMode:
        return self._mode

    def get_records(self, layer_id: int | None = None) -> list[RouteRecord]:
        if layer_id is None:
            out: list[RouteRecord] = []
            for records in self._records.values():
                out.extend(records)
            return out
        return list(self._records.get(layer_id, []))

    def set_replay_record(self, layer_id: int, record: RouteRecord) -> None:
        self._replay_records[layer_id] = record
        self._ensure_patched()

    def detach(self) -> None:
        if not self._patched:
            return
        for block_id, original_forward in list(self._original_forwards.items()):
            block = self._block_by_id(block_id)
            if block is not None:
                block.forward = original_forward
        self._original_forwards.clear()
        self._patched = False
        self._mode = ReplayMode.OFF

    def _ensure_patched(self) -> None:
        if self._patched:
            return
        for ref in self.discover_sparse_moe_blocks():
            block = ref.block
            block_id = id(block)
            if block_id in self._original_forwards:
                continue
            original_forward = block.forward
            self._original_forwards[block_id] = original_forward

            def patched_forward(block_self: Any, hidden_states: Any, *, _ref: SparseMoeBlockRef = ref) -> Any:
                return self._forward_sparse_block(_ref, hidden_states)

            block.forward = MethodType(patched_forward, block)
        self._patched = True

    def _forward_sparse_block(self, ref: SparseMoeBlockRef, hidden_states: Any) -> Any:
        block = ref.block
        if self._mode == ReplayMode.OFF:
            return self._original_forwards[id(block)](hidden_states)

        if self._mode == ReplayMode.REPLAY:
            record = self._replay_records.get(ref.layer_id)
            if record is None:
                raise RuntimeError(f"no replay RouteRecord configured for layer {ref.layer_id}")
            topk_idx = record.topk_idx
            topk_weights = record.topk_weights
            if topk_weights is None:
                topk_weights = _uniform_weights(record.token_count, record.top_k, like=topk_idx)
            return _call_experts(block, hidden_states, topk_idx, topk_weights)

        if self._mode == ReplayMode.RECORD:
            router_logits, topk_weights, topk_idx = _call_gate(block, hidden_states)
            token_count, top_k = _route_shape(topk_idx)
            record = RouteRecord(
                record_id=f"layer_{ref.layer_id}_{self._record_counter:06d}",
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                layer_id=ref.layer_id,
                num_experts=_infer_num_experts(self.model, block, topk_idx),
                top_k=top_k,
                token_count=token_count,
                replay_level=ReplayLevel.R2 if topk_weights is not None else ReplayLevel.R1,
                router_logits=None,  # diagnostic-only; not required for replay
                metadata={"backend_id": self.backend_id},
            )
            self._record_counter += 1
            self._records.setdefault(ref.layer_id, []).append(record)
            # Keep fake/diagnostic state close to HF behavior without making it
            # part of the replay contract.
            if hasattr(block, "_router_logits"):
                block._router_logits = router_logits
            return _call_experts(block, hidden_states, topk_idx, topk_weights)

        raise RuntimeError(f"unsupported adapter mode: {self._mode}")

    def _block_by_id(self, block_id: int) -> Any | None:
        for ref in self.discover_sparse_moe_blocks():
            if id(ref.block) == block_id:
                return ref.block
        return None


def _iter_layers(model: Any) -> list[tuple[int, Any]]:
    for owner in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)):
        if owner is None:
            continue
        layers = getattr(owner, "layers", None)
        if layers is not None:
            return list(enumerate(layers))
    return []


def _extract_sparse_block(layer: Any) -> tuple[Any | None, str | None]:
    for attr_name in ("block_sparse_moe", "mlp", "moe_block", "sparse_moe", "feed_forward"):
        block = getattr(layer, attr_name, None)
        if _looks_like_sparse_moe_block(block):
            return block, attr_name
    if _looks_like_sparse_moe_block(layer):
        return layer, None
    return None, None


def _looks_like_sparse_moe_block(block: Any) -> bool:
    return block is not None and hasattr(block, "gate") and hasattr(block, "experts") and hasattr(block, "forward")


def _call_gate(block: Any, hidden_states: Any) -> tuple[Any, Any, Any]:
    gate_out = block.gate(hidden_states)
    try:
        router_logits, topk_weights, topk_idx = gate_out
    except Exception as exc:
        raise RuntimeError("Qwen3MoE gate must return (router_logits, topk_weights, topk_idx)") from exc
    idx_shape = tensor_shape(topk_idx)
    weight_shape = tensor_shape(topk_weights)
    if len(idx_shape) != 2 or weight_shape != idx_shape:
        raise RuntimeError(
            f"invalid Qwen3MoE route shapes: topk_idx={idx_shape}, topk_weights={weight_shape}"
        )
    return router_logits, topk_weights, topk_idx


def _call_experts(block: Any, hidden_states: Any, topk_idx: Any, topk_weights: Any) -> Any:
    experts = block.experts
    if callable(experts) and not isinstance(experts, (list, tuple)):
        return experts(hidden_states, topk_idx, topk_weights)
    return _fake_list_expert_dispatch(experts, hidden_states, topk_idx, topk_weights)


def _fake_list_expert_dispatch(experts: Any, hidden_states: Any, topk_idx: Any, topk_weights: Any) -> Any:
    idx = np.asarray(topk_idx)
    weights = np.asarray(topk_weights, dtype=np.float32)
    hs = np.asarray(hidden_states)
    result = np.zeros_like(hs, dtype=np.float32)
    for token_i in range(idx.shape[0]):
        for k_i in range(idx.shape[1]):
            expert_id = int(idx[token_i, k_i])
            expert_out = experts[expert_id](hs[token_i])
            result[token_i] += float(weights[token_i, k_i]) * np.asarray(expert_out)
    return result


def _route_shape(topk_idx: Any) -> tuple[int, int]:
    shape = tensor_shape(topk_idx)
    if len(shape) != 2:
        raise RuntimeError(f"topk_idx must be 2D, got shape {shape}")
    return int(shape[0]), int(shape[1])


def _infer_num_experts(model: Any, block: Any, topk_idx: Any) -> int:
    config = getattr(model, "config", None)
    for obj in (config, block.gate, block):
        value = getattr(obj, "num_experts", None)
        if value is not None:
            return int(value)
    experts = getattr(block, "experts", None)
    if isinstance(experts, (list, tuple)):
        return len(experts)
    idx = np.asarray(topk_idx)
    return int(idx.max()) + 1 if idx.size else 0


def _uniform_weights(token_count: int, top_k: int, like: Any) -> Any:
    value = 1.0 / float(top_k)
    weights = np.full((token_count, top_k), value, dtype=np.float32)
    try:
        import torch  # type: ignore

        if torch.is_tensor(like):
            return torch.full((token_count, top_k), value, dtype=torch.float32, device=like.device)
    except Exception:
        pass
    return weights
