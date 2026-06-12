"""Adapter registry, capability negotiation, and runtime facade tests."""

from __future__ import annotations

import numpy as np

from routeforge import MoeReplayRuntime, ReplayMode, create_adapter, list_adapters
from routeforge.backends import AdapterRegistry, HFQwen3MoeAdapter, negotiate_capabilities
from routeforge.core import BackendCapabilities, ReplayLevel

from .test_hf_qwen3_moe_adapter import _get_moe_block, _make_fake_model


def test_default_registry_contains_hf_qwen3_moe() -> None:
    assert "hf_qwen3_moe" in list_adapters()


def test_create_adapter_from_default_registry() -> None:
    model = _make_fake_model()
    adapter = create_adapter("hf_qwen3_moe", model)
    assert isinstance(adapter, HFQwen3MoeAdapter)


def test_custom_registry_rejects_duplicate_backend_id() -> None:
    registry = AdapterRegistry()
    registry.register("x", lambda model: HFQwen3MoeAdapter(model))
    try:
        registry.register("x", lambda model: HFQwen3MoeAdapter(model))
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate adapter registration did not fail")


def test_capability_negotiation_accepts_r2_replay() -> None:
    caps = HFQwen3MoeAdapter(_make_fake_model()).capabilities()
    result = negotiate_capabilities(caps, replay_level=ReplayLevel.R2, mode=ReplayMode.REPLAY)
    assert result.accepted
    assert result.reason is None


def test_capability_negotiation_rejects_unsupported_level() -> None:
    caps = BackendCapabilities(backend_id="minimal")
    result = negotiate_capabilities(caps, replay_level=ReplayLevel.R2, mode=ReplayMode.REPLAY)
    assert not result.accepted
    assert "does not support" in str(result.reason)


def test_runtime_record_context_manager_writes_tape(tmp_path) -> None:
    model = _make_fake_model(moe_layer_ids=[1])
    runtime = MoeReplayRuntime(
        mode=ReplayMode.RECORD,
        backend="hf_qwen3_moe",
        tape_path=tmp_path,
        tape_id="runtime",
        model_id="fake-model",
    )
    with runtime.attach(model):
        _ = _get_moe_block(model, 1).forward(np.zeros((4, 16), dtype=np.float32))

    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "index.json").is_file()


def test_runtime_replay_loads_records_from_tape(tmp_path) -> None:
    model = _make_fake_model(moe_layer_ids=[1])
    recorder = MoeReplayRuntime(
        mode=ReplayMode.RECORD,
        backend="hf_qwen3_moe",
        tape_path=tmp_path,
        tape_id="runtime",
        model_id="fake-model",
    )
    with recorder.attach(model):
        _ = _get_moe_block(model, 1).forward(np.zeros((4, 16), dtype=np.float32))

    replay_model = _make_fake_model(moe_layer_ids=[1])
    replayer = MoeReplayRuntime(mode=ReplayMode.REPLAY, backend="hf_qwen3_moe", tape_path=tmp_path)
    with replayer.attach(replay_model):
        _ = _get_moe_block(replay_model, 1).forward(np.zeros((4, 16), dtype=np.float32))
