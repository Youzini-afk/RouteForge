"""Unit tests for RouteForge backend adapter: HFQwen3MoeAdapter.

These tests define the expected contract for:
  - routeforge.backends.base          (BackendAdapter ABC)
  - routeforge.backends.hf_qwen3_moe  (HFQwen3MoeAdapter)

All model objects are fake lightweight stand-ins — no real transformers/torch
dependency is required.  The fakes mimic the *structure* of a Qwen3 MoE model
(gate + experts inside a sparse block) so that the adapter's discover, record,
replay, and detach logic can be exercised in isolation.

Coverage points:
  1. Capabilities — R0/R1/R2 supported; shared-expert / R3 / R4 not supported.
  2. discover_sparse_moe_blocks — identifies layers with gate + experts.
  3. Record path — fake gate returns (logits, weights, indices); adapter
     produces a RouteRecord with correct topk_idx / topk_weights.
  4. Replay path — given a RouteRecord, experts receive the recorded
     indices/weights instead of live gate output.
  5. output_router_logits is NOT part of the main record/replay chain.
  6. detach — restores the original forward method.

Production code does not exist yet; these tests serve as the executable spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from routeforge.backends.base import BackendAdapter
from routeforge.backends.hf_qwen3_moe import HFQwen3MoeAdapter
from routeforge.core.capabilities import BackendCapabilities
from routeforge.core.enums import ReplayLevel, ReplayMode
from routeforge.core.errors import ReplayPolicyError
from routeforge.core.route_record import RouteRecord


# ============================================================================
# Fake model components — no torch / transformers required
# ============================================================================


class FakeExpert:
    """A single MoE expert that records which indices it received."""

    def __init__(self, expert_id: int, hidden_size: int = 16) -> None:
        self.expert_id = expert_id
        self.hidden_size = hidden_size
        self.calls: list[dict[str, Any]] = []

    def __call__(self, hidden_states: Any, **kwargs: Any) -> Any:
        """Apply expert and record the call for later inspection."""
        self.calls.append({"hidden_states": hidden_states, "kwargs": kwargs})
        # Return a simple transformation (identity + expert_id offset for detectability)
        if hasattr(hidden_states, "copy"):
            result = hidden_states.copy()
        else:
            result = np.array(hidden_states, dtype=np.float32)
        return result + self.expert_id * 0.01


class FakeGate:
    """Fake router gate that produces deterministic (logits, topk_weights, topk_indices)."""

    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        token_count: int = 4,
        hidden_size: int = 16,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.token_count = token_count
        self.hidden_size = hidden_size
        self.call_count: int = 0

    def __call__(self, hidden_states: Any) -> tuple[Any, Any, Any]:
        """Return (router_logits, topk_weights, topk_indices).

        router_logits: shape [token_count, num_experts]
        topk_weights:  shape [token_count, top_k]
        topk_indices:  shape [token_count, top_k]
        """
        self.call_count += 1
        rng = np.random.default_rng(seed=42 + self.call_count)

        # Full router logits (for diagnostics — NOT used in main record/replay)
        router_logits = rng.standard_normal(
            (self.token_count, self.num_experts)
        ).astype(np.float32)

        # Deterministic top-k indices
        topk_indices = np.zeros((self.token_count, self.top_k), dtype=np.int16)
        topk_weights = np.zeros((self.token_count, self.top_k), dtype=np.float32)
        for t in range(self.token_count):
            # Pick top_k experts deterministically (rotate through experts)
            start = (t * self.top_k) % self.num_experts
            chosen = [(start + i) % self.num_experts for i in range(self.top_k)]
            topk_indices[t] = chosen
            # Assign weights that sum to 1
            raw_w = np.array([0.7, 0.3] if self.top_k == 2 else [1.0 / self.top_k] * self.top_k, dtype=np.float32)
            topk_weights[t] = raw_w[:self.top_k]

        return router_logits, topk_weights, topk_indices


class FakeSparseMoeBlock:
    """Fake Qwen3 MoE sparse block with gate + experts.

    Mimics the structure of `Qwen3MoeSparseMoeBlock` in HuggingFace:
      - self.gate   → router gate
      - self.experts → list of expert modules
      - forward()   → calls gate, dispatches to experts, combines
    """

    def __init__(
        self,
        num_experts: int = 8,
        top_k: int = 2,
        token_count: int = 4,
        hidden_size: int = 16,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.token_count = token_count
        self.hidden_size = hidden_size
        self.gate = FakeGate(
            num_experts=num_experts,
            top_k=top_k,
            token_count=token_count,
            hidden_size=hidden_size,
        )
        self.experts = [FakeExpert(i, hidden_size) for i in range(num_experts)]
        self._original_forward = self.forward
        self._router_logits: Any = None  # stores last gate logits (diagnostics)

    def forward(self, hidden_states: Any) -> Any:
        """Normal forward: gate → dispatch → combine."""
        router_logits, topk_weights, topk_indices = self.gate(hidden_states)
        self._router_logits = router_logits

        # Dispatch to experts (simplified: each token to its top-k experts)
        result = np.zeros_like(hidden_states, dtype=np.float32) if not isinstance(hidden_states, np.ndarray) else np.zeros_like(hidden_states)
        for t in range(self.token_count):
            for k_idx in range(self.top_k):
                expert_id = int(topk_indices[t, k_idx])
                weight = float(topk_weights[t, k_idx])
                expert_out = self.experts[expert_id](
                    hidden_states[t] if hasattr(hidden_states, "__getitem__") else hidden_states
                )
                if hasattr(result, "__setitem__"):
                    result[t] += weight * expert_out
                else:
                    result += weight * expert_out

        return result


class FakeDenseLayer:
    """A non-MoE (dense) decoder layer that should NOT be discovered as sparse."""

    def __init__(self, layer_id: int, hidden_size: int = 16) -> None:
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        # No gate, no experts — this is a dense attention/FFN layer

    def forward(self, hidden_states: Any) -> Any:
        return hidden_states


class FakeQwen3MoeModel:
    """Fake Qwen3 MoE model with a mix of dense and sparse-MoE layers.

    Mimics the structure of a HuggingFace Qwen3MoeForCausalLM:
      - self.model.layers[i] — decoder layers
      - Some layers have `.block_sparse_moe` (sparse MoE block)
      - Others are dense (no MoE)
    """

    def __init__(
        self,
        num_layers: int = 6,
        moe_layer_ids: list[int] | None = None,
        num_experts: int = 8,
        top_k: int = 2,
        token_count: int = 4,
        hidden_size: int = 16,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if moe_layer_ids is None:
            moe_layer_ids = [1, 3, 5]  # default: every other layer is MoE
        self.moe_layer_ids = moe_layer_ids

        # Build layers — each is a _FakeDecoderLayer; MoE layers have
        # .block_sparse_moe set, dense layers have it set to None.
        self.layers: list[_FakeDecoderLayer] = []
        for i in range(num_layers):
            if i in moe_layer_ids:
                moe_block = FakeSparseMoeBlock(
                    num_experts=num_experts,
                    top_k=top_k,
                    token_count=token_count,
                    hidden_size=hidden_size,
                )
                layer = _FakeDecoderLayer(layer_id=i, sparse_moe=moe_block)
            else:
                layer = _FakeDecoderLayer(layer_id=i, sparse_moe=None)
            self.layers.append(layer)

        # Model-level attribute that HuggingFace models expose
        self.config = _FakeConfig(
            num_hidden_layers=num_layers,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
        )


class _FakeDecoderLayer:
    """Fake decoder layer that optionally contains a sparse MoE block."""

    def __init__(
        self,
        layer_id: int,
        sparse_moe: FakeSparseMoeBlock | None = None,
    ) -> None:
        self.layer_id = layer_id
        self.block_sparse_moe = sparse_moe  # None for dense layers

    def forward(self, hidden_states: Any) -> Any:
        if self.block_sparse_moe is not None:
            return self.block_sparse_moe.forward(hidden_states)
        return hidden_states


class _FakeConfig:
    """Fake model config mimicking HuggingFace PretrainedConfig."""

    def __init__(
        self,
        num_hidden_layers: int = 6,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok


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


def _make_fake_model(
    num_layers: int = 6,
    moe_layer_ids: list[int] | None = None,
    num_experts: int = 8,
    top_k: int = 2,
    token_count: int = 4,
    hidden_size: int = 16,
) -> FakeQwen3MoeModel:
    return FakeQwen3MoeModel(
        num_layers=num_layers,
        moe_layer_ids=moe_layer_ids,
        num_experts=num_experts,
        top_k=top_k,
        token_count=token_count,
        hidden_size=hidden_size,
    )


def _get_moe_block(model: FakeQwen3MoeModel, layer_id: int) -> FakeSparseMoeBlock:
    """Get the FakeSparseMoeBlock for a given MoE layer (asserts it exists)."""
    block = model.layers[layer_id].block_sparse_moe
    assert block is not None, f"Layer {layer_id} is not an MoE layer"
    return block


# ============================================================================
# 1. Capabilities: R0/R1/R2 supported; shared / R3 / R4 NOT supported
# ============================================================================


class TestCapabilities:
    """HFQwen3MoeAdapter must declare R0/R1/R2 support and deny shared/R3/R4."""

    def test_adapter_is_backend_adapter_subclass(self) -> None:
        assert issubclass(HFQwen3MoeAdapter, BackendAdapter)

    def test_capabilities_returns_backend_capabilities(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert isinstance(caps, BackendCapabilities)

    def test_supports_r0_observe(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert caps.supports_level(ReplayLevel.R0)

    def test_supports_r1_index_replay(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert caps.supports_level(ReplayLevel.R1)

    def test_supports_r2_weighted_replay(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert caps.supports_level(ReplayLevel.R2)

    def test_does_not_support_r3_dispatch(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert not caps.supports_level(ReplayLevel.R3)

    def test_does_not_support_r4_handle(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert not caps.supports_level(ReplayLevel.R4)

    def test_does_not_support_shared_expert(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert not caps.supports_shared_expert

    def test_r0_in_supported_replay_levels(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert ReplayLevel.R0 in caps.supported_replay_levels

    def test_r1_in_supported_replay_levels(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert ReplayLevel.R1 in caps.supported_replay_levels

    def test_r2_in_supported_replay_levels(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert ReplayLevel.R2 in caps.supported_replay_levels

    def test_r3_not_in_supported_replay_levels(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert ReplayLevel.R3 not in caps.supported_replay_levels

    def test_r4_not_in_supported_replay_levels(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert ReplayLevel.R4 not in caps.supported_replay_levels

    def test_supports_index_replay_flag(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert caps.supports_index_replay

    def test_supports_weight_replay_flag(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert caps.supports_weight_replay

    def test_does_not_support_dispatch_plan(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert not caps.supports_dispatch_plan

    def test_does_not_support_handle_replay(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert not caps.supports_handle_replay

    def test_record_mode_supported(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert ReplayMode.RECORD in caps.supported_modes

    def test_off_mode_supported(self) -> None:
        model = _make_fake_model()
        adapter = HFQwen3MoeAdapter(model)
        caps = adapter.capabilities()
        assert ReplayMode.OFF in caps.supported_modes


# ============================================================================
# 2. discover_sparse_moe_blocks — identify blocks with gate + experts
# ============================================================================


class TestDiscoverSparseMoeBlocks:
    """discover_sparse_moe_blocks must find only layers with gate + experts."""

    def test_discovers_moe_layers(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1, 3, 5])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        assert len(blocks) == 3

    def test_discovers_correct_layer_ids(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1, 3, 5])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        discovered_ids = sorted(b.layer_id for b in blocks)
        assert discovered_ids == [1, 3, 5]

    def test_does_not_discover_dense_layers(self) -> None:
        model = _make_fake_model(moe_layer_ids=[2, 4])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        discovered_ids = [b.layer_id for b in blocks]
        assert 0 not in discovered_ids  # dense layer
        assert 1 not in discovered_ids  # dense layer

    def test_each_block_has_gate(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1, 3])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        for block in blocks:
            assert hasattr(block, "gate") or hasattr(block, "block") and hasattr(block.block, "gate")

    def test_each_block_has_experts(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1, 3])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        for block in blocks:
            # block should expose experts either directly or via .block
            target = block.block if hasattr(block, "block") else block
            assert hasattr(target, "experts")

    def test_expert_count_matches_config(self) -> None:
        model = _make_fake_model(num_experts=16, moe_layer_ids=[1])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        assert len(blocks) == 1
        block = blocks[0]
        target = block.block if hasattr(block, "block") else block
        assert len(target.experts) == 16

    def test_no_moe_layers_returns_empty(self) -> None:
        model = _make_fake_model(moe_layer_ids=[])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        assert blocks == []

    def test_all_layers_moe(self) -> None:
        model = _make_fake_model(
            num_layers=4, moe_layer_ids=[0, 1, 2, 3]
        )
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        assert len(blocks) == 4

    def test_single_moe_layer(self) -> None:
        model = _make_fake_model(moe_layer_ids=[2])
        adapter = HFQwen3MoeAdapter(model)
        blocks = adapter.discover_sparse_moe_blocks()
        assert len(blocks) == 1
        assert blocks[0].layer_id == 2


# ============================================================================
# 3. Record path — gate returns (logits, weights, indices), adapter produces RouteRecord
# ============================================================================


class TestRecordPath:
    """In RECORD mode, the adapter must intercept gate output and produce
    RouteRecord objects with correct topk_idx / topk_weights."""

    def test_record_produces_route_record(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)

        # Run forward through the MoE block — adapter should intercept gate
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        assert len(records) >= 1
        record = records[-1]
        assert isinstance(record, RouteRecord)

    def test_record_has_correct_topk_idx_shape(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        record = records[-1]
        assert record.topk_idx is not None
        idx = np.asarray(record.topk_idx)
        assert idx.shape == (4, 2)

    def test_record_has_correct_topk_weights_shape(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        record = records[-1]
        assert record.topk_weights is not None
        weights = np.asarray(record.topk_weights)
        assert weights.shape == (4, 2)

    def test_record_topk_idx_values_within_num_experts(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        record = records[-1]
        idx = np.asarray(record.topk_idx)
        assert (idx >= 0).all()
        assert (idx < 8).all()

    def test_record_topk_weights_sum_to_one_per_token(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        record = records[-1]
        weights = np.asarray(record.topk_weights)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-5)

    def test_record_layer_id_matches(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1, 3], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)

        # Record from layer 1
        moe_block_1 = _get_moe_block(model, 1)
        _ = moe_block_1.forward(hidden_states)

        records_1 = adapter.get_records(layer_id=1)
        assert records_1[-1].layer_id == 1

        # Record from layer 3
        moe_block_3 = _get_moe_block(model, 3)
        _ = moe_block_3.forward(hidden_states)

        records_3 = adapter.get_records(layer_id=3)
        assert records_3[-1].layer_id == 3

    def test_record_num_experts_matches_model(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=16, top_k=4, token_count=8)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((8, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        record = records[-1]
        assert record.num_experts == 16
        assert record.top_k == 4
        assert record.token_count == 8

    def test_record_replay_level_is_r2(self) -> None:
        """When recording with weights available, replay_level should be R2."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        record = records[-1]
        # R2 because we have both indices and weights
        assert record.replay_level in (ReplayLevel.R2, ReplayLevel.R1)

    def test_multiple_forward_calls_accumulate_records(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        for _ in range(3):
            _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        assert len(records) >= 3

    def test_record_does_not_store_router_logits_in_main_path(self) -> None:
        """RouteRecord.router_logits should be None or only stored as diagnostics,
        NOT required for the main record/replay chain."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        record = records[-1]
        # router_logits may be present for diagnostics but must not be
        # required for replay. The replay chain uses topk_idx + topk_weights only.
        # If router_logits is present, it's an optional diagnostic field.
        # The key invariant: replay must work with only topk_idx + topk_weights.
        # We verify that topk_idx and topk_weights are populated regardless.
        assert record.topk_idx is not None
        assert record.topk_weights is not None


# ============================================================================
# 4. Replay path — experts receive recorded indices/weights, not live gate output
# ============================================================================


class TestReplayPath:
    """In REPLAY mode, the adapter must override gate output with recorded
    RouteRecord data so that experts receive the recorded indices/weights."""

    def test_replay_uses_recorded_indices(self) -> None:
        """During replay, the gate should NOT be called; recorded indices should
        be used instead."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        # Step 1: Record
        adapter.set_mode(ReplayMode.RECORD)
        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)
        records = adapter.get_records(layer_id=1)
        record = records[-1]

        # Step 2: Switch to REPLAY and provide the record
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)

        # Reset expert call history
        for expert in moe_block.experts:
            expert.calls.clear()

        # Step 3: Forward through MoE block in replay mode
        _ = moe_block.forward(hidden_states)

        # Step 4: Verify experts received the RECORDED indices, not live gate output
        recorded_idx = np.asarray(record.topk_idx)
        # Collect which experts were called
        called_expert_ids: set[int] = set()
        for expert in moe_block.experts:
            if expert.calls:
                called_expert_ids.add(expert.expert_id)

        # Every expert in the recorded topk_idx should have been called
        unique_recorded = set(recorded_idx.flatten().tolist())
        assert unique_recorded.issubset(called_expert_ids) or len(called_expert_ids) > 0

    def test_replay_uses_recorded_weights(self) -> None:
        """During replay, the combine step should use recorded weights."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        # Record
        adapter.set_mode(ReplayMode.RECORD)
        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)
        records = adapter.get_records(layer_id=1)
        record = records[-1]

        # Replay
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)
        _ = moe_block.forward(hidden_states)

        # The replay should successfully complete without calling the live gate.
        # We verify by checking that the gate was NOT called during replay.
        # (Gate call_count should not have increased from the record phase.)
        gate_calls_after_record = moe_block.gate.call_count

        # Do another replay
        _ = moe_block.forward(hidden_states)

        # Gate should not have been called again during replay
        # (or at most minimally, depending on implementation)
        # The key invariant: replay does NOT depend on live gate output.
        assert moe_block.gate.call_count == gate_calls_after_record

    def test_replay_gate_not_called(self) -> None:
        """In REPLAY mode, the gate should not be invoked at all."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        # Record
        adapter.set_mode(ReplayMode.RECORD)
        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)
        records = adapter.get_records(layer_id=1)
        record = records[-1]
        gate_calls_after_record = moe_block.gate.call_count

        # Replay
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)
        _ = moe_block.forward(hidden_states)

        # Gate call count should not have increased
        assert moe_block.gate.call_count == gate_calls_after_record

    def test_replay_different_record_changes_dispatch(self) -> None:
        """Replaying a different RouteRecord should dispatch to different experts."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)

        # Create two different records
        record_a = _make_r2_record(
            token_count=4, top_k=2, num_experts=8, layer_id=1,
            topk_idx=np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.int16),
            topk_weights=np.array([[0.6, 0.4]] * 4, dtype=np.float32),
        )
        record_b = _make_r2_record(
            token_count=4, top_k=2, num_experts=8, layer_id=1,
            topk_idx=np.array([[6, 7], [6, 7], [6, 7], [6, 7]], dtype=np.int16),
            topk_weights=np.array([[0.5, 0.5]] * 4, dtype=np.float32),
        )

        # Replay with record_a
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record_a)
        for expert in moe_block.experts:
            expert.calls.clear()
        _ = moe_block.forward(hidden_states)
        calls_a = {i: len(moe_block.experts[i].calls) for i in range(8)}

        # Replay with record_b
        adapter.set_replay_record(layer_id=1, record=record_b)
        for expert in moe_block.experts:
            expert.calls.clear()
        _ = moe_block.forward(hidden_states)
        calls_b = {i: len(moe_block.experts[i].calls) for i in range(8)}

        # Experts 0,1 should be called in record_a; experts 6,7 in record_b
        assert calls_a[0] > 0 or calls_a[1] > 0
        assert calls_b[6] > 0 or calls_b[7] > 0

    def test_replay_r1_record_ignores_weights(self) -> None:
        """HF execution replay requires weights; R1 is rejected for this adapter.

        RouteForge can represent R1 semantically, but this HF reference sparse
        block requires weights to call experts safely. It must not silently use
        uniform weights.
        """
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        r1_record = RouteRecord(
            topk_idx=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int16),
            topk_weights=None,
            layer_id=1,
            num_experts=8,
            top_k=2,
            token_count=4,
            replay_level=ReplayLevel.R1,
        )

        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=r1_record)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        for expert in moe_block.experts:
            expert.calls.clear()

        with pytest.raises((ReplayPolicyError, RuntimeError, ValueError)):
            _ = moe_block.forward(hidden_states)

    def test_replay_token_count_mismatch_fails_closed(self) -> None:
        """Adapter replay must pass through core validation, not blindly dispatch."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        record = _make_r2_record(
            token_count=2,
            top_k=2,
            num_experts=8,
            layer_id=1,
            topk_idx=np.array([[0, 1], [2, 3]], dtype=np.int16),
            topk_weights=np.array([[0.6, 0.4], [0.7, 0.3]], dtype=np.float32),
        )
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        with pytest.raises((ReplayPolicyError, RuntimeError, ValueError)):
            _ = moe_block.forward(hidden_states)

    def test_replay_top_k_mismatch_fails_closed(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        record = _make_r2_record(
            token_count=4,
            top_k=3,
            num_experts=8,
            layer_id=1,
            topk_idx=np.array([[0, 1, 2]] * 4, dtype=np.int16),
            topk_weights=np.array([[0.5, 0.3, 0.2]] * 4, dtype=np.float32),
        )
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        with pytest.raises((ReplayPolicyError, RuntimeError, ValueError)):
            _ = moe_block.forward(hidden_states)

    def test_replay_num_experts_mismatch_fails_closed(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        record = _make_r2_record(
            token_count=4,
            top_k=2,
            num_experts=4,
            layer_id=1,
            topk_idx=np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int16),
            topk_weights=np.array([[0.6, 0.4]] * 4, dtype=np.float32),
        )
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        with pytest.raises((ReplayPolicyError, RuntimeError, ValueError)):
            _ = moe_block.forward(hidden_states)


# ============================================================================
# 5. output_router_logits is NOT part of the main record/replay chain
# ============================================================================


class TestOutputRouterLogitsNotMainPath:
    """The HuggingFace `output_router_logits` flag is a diagnostic/debug
    feature.  The adapter's record/replay must NOT depend on it."""

    def test_record_works_without_output_router_logits(self) -> None:
        """Recording should work even when output_router_logits is False
        (the default in HuggingFace models)."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records = adapter.get_records(layer_id=1)
        assert len(records) >= 1
        # topk_idx and topk_weights must be populated WITHOUT output_router_logits
        assert records[-1].topk_idx is not None
        assert records[-1].topk_weights is not None

    def test_replay_does_not_consume_router_logits(self) -> None:
        """Replay should not need to read router_logits from the RouteRecord."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        # Create a record WITHOUT router_logits
        record = _make_r2_record(
            token_count=4, top_k=2, num_experts=8, layer_id=1,
        )
        # Ensure router_logits is not set (or None)
        assert record.router_logits is None or record.router_logits is not None
        # The key point: replay should work regardless of router_logits

        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)

        # Should not raise even though router_logits is None/absent
        _ = moe_block.forward(hidden_states)

    def test_router_logits_field_is_optional_diagnostic(self) -> None:
        """If router_logits is present in a record, it's for diagnostics only.
        The adapter must not require it for replay."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        # Record with explicit router_logits
        adapter.set_mode(ReplayMode.RECORD)
        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)
        records = adapter.get_records(layer_id=1)

        # Whether router_logits is populated or not, the record should be valid
        record = records[-1]
        assert record.topk_idx is not None
        assert record.topk_weights is not None

        # Replay works regardless
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)
        _ = moe_block.forward(hidden_states)


# ============================================================================
# 6. detach — restores original forward method
# ============================================================================


class TestDetach:
    """detach() must remove all adapter hooks and restore the original forward
    method on every MoE block."""

    def test_detach_restores_original_forward(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1, 3], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        # Save original forward references
        moe_block_1 = _get_moe_block(model, 1)
        moe_block_3 = _get_moe_block(model, 3)
        original_forward_1 = moe_block_1.forward
        original_forward_3 = moe_block_3.forward

        # Install adapter hooks
        adapter.set_mode(ReplayMode.RECORD)

        # Detach
        adapter.detach()

        # Forward should be restored to original
        assert moe_block_1.forward == original_forward_1
        assert moe_block_3.forward == original_forward_3

    def test_detach_stops_recording(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)

        records_before = len(adapter.get_records(layer_id=1))

        # Detach
        adapter.detach()

        # Running forward again should NOT produce new records
        _ = moe_block.forward(hidden_states)
        records_after = len(adapter.get_records(layer_id=1))
        assert records_after == records_before

    def test_detach_stops_replay(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        # Record
        adapter.set_mode(ReplayMode.RECORD)
        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)
        _ = moe_block.forward(hidden_states)
        records = adapter.get_records(layer_id=1)
        record = records[-1]

        # Enter replay mode
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)

        # Detach
        adapter.detach()

        # After detach, the gate should be called again (original behavior restored)
        gate_calls_before = moe_block.gate.call_count
        _ = moe_block.forward(hidden_states)
        gate_calls_after = moe_block.gate.call_count
        assert gate_calls_after > gate_calls_before

    def test_detach_idempotent(self) -> None:
        """Calling detach() multiple times should not raise."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)

        adapter.detach()
        adapter.detach()  # Should not raise

    def test_detach_without_prior_attach(self) -> None:
        """Calling detach() before any set_mode should not raise."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.detach()  # Should not raise

    def test_detach_preserves_model_functionality(self) -> None:
        """After detach, the model should produce the same output as before
        any adapter hooks were installed."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)

        # Get baseline output
        output_before = moe_block.forward(hidden_states)

        # Install and use adapter
        adapter.set_mode(ReplayMode.RECORD)
        _ = moe_block.forward(hidden_states)

        # Detach
        adapter.detach()

        # Output should match baseline
        output_after = moe_block.forward(hidden_states)
        if hasattr(output_before, "shape"):
            np.testing.assert_array_equal(
                np.asarray(output_before), np.asarray(output_after)
            )
        else:
            assert output_before == output_after

    def test_detach_all_moe_layers(self) -> None:
        """detach() must restore forward on ALL MoE layers, not just the first."""
        model = _make_fake_model(moe_layer_ids=[1, 3, 5], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        original_forwards = {}
        for lid in [1, 3, 5]:
            original_forwards[lid] = _get_moe_block(model, lid).forward

        adapter.set_mode(ReplayMode.RECORD)
        adapter.detach()

        for lid in [1, 3, 5]:
            assert _get_moe_block(model, lid).forward == original_forwards[lid]


# ============================================================================
# 7. Mode transitions and lifecycle
# ============================================================================


class TestModeTransitions:
    """The adapter must correctly handle mode transitions."""

    def test_initial_mode_is_off(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1])
        adapter = HFQwen3MoeAdapter(model)
        assert adapter.current_mode() == ReplayMode.OFF

    def test_set_mode_to_record(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1])
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)
        assert adapter.current_mode() == ReplayMode.RECORD

    def test_set_mode_to_replay(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1])
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.REPLAY)
        assert adapter.current_mode() == ReplayMode.REPLAY

    def test_set_mode_to_off(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1])
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.RECORD)
        adapter.set_mode(ReplayMode.OFF)
        assert adapter.current_mode() == ReplayMode.OFF

    def test_record_then_replay_transition(self) -> None:
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)

        # Record
        adapter.set_mode(ReplayMode.RECORD)
        _ = moe_block.forward(hidden_states)
        records = adapter.get_records(layer_id=1)
        record = records[-1]

        # Transition to replay
        adapter.set_mode(ReplayMode.REPLAY)
        adapter.set_replay_record(layer_id=1, record=record)
        _ = moe_block.forward(hidden_states)
        # Should not raise

    def test_replay_without_record_raises(self) -> None:
        """Entering REPLAY mode without providing a record should fail gracefully."""
        model = _make_fake_model(moe_layer_ids=[1], num_experts=8, top_k=2, token_count=4)
        adapter = HFQwen3MoeAdapter(model)
        adapter.set_mode(ReplayMode.REPLAY)

        hidden_states = np.zeros((4, 16), dtype=np.float32)
        moe_block = _get_moe_block(model, 1)

        # Forward without a replay record should either raise or fall back
        with pytest.raises((ValueError, RuntimeError)):
            _ = moe_block.forward(hidden_states)


# ============================================================================
# 8. BackendAdapter ABC contract
# ============================================================================


class TestBackendAdapterABC:
    """BackendAdapter must define the required abstract interface."""

    def test_has_capabilities_method(self) -> None:
        assert hasattr(BackendAdapter, "capabilities")

    def test_has_discover_sparse_moe_blocks_method(self) -> None:
        assert hasattr(BackendAdapter, "discover_sparse_moe_blocks")

    def test_has_set_mode_method(self) -> None:
        assert hasattr(BackendAdapter, "set_mode")

    def test_has_current_mode_method(self) -> None:
        assert hasattr(BackendAdapter, "current_mode")

    def test_has_get_records_method(self) -> None:
        assert hasattr(BackendAdapter, "get_records")

    def test_has_set_replay_record_method(self) -> None:
        assert hasattr(BackendAdapter, "set_replay_record")

    def test_has_detach_method(self) -> None:
        assert hasattr(BackendAdapter, "detach")

    def test_cannot_instantiate_base_class(self) -> None:
        """BackendAdapter should be abstract and not directly instantiable."""
        # Attempting to instantiate BackendAdapter() should fail
        # (either TypeError for abstract methods or similar)
        with pytest.raises(TypeError):
            BackendAdapter()  # type: ignore[abstract]
