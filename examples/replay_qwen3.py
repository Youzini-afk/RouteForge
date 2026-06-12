"""Minimal HF Qwen3MoE replay sketch."""

from __future__ import annotations

from routeforge.backends import HFQwen3MoeAdapter
from routeforge.core import ReplayMode
from routeforge.storage import RouteTapeReader


def replay_model_routes(model, inputs, tape_dir: str = "request_001.mrt"):
    adapter = HFQwen3MoeAdapter(model)
    with RouteTapeReader(tape_dir=tape_dir) as reader:
        for record_id in reader.record_ids():
            record = reader.read_record(record_id)
            adapter.set_replay_record(record.layer_id, record)
    adapter.set_mode(ReplayMode.REPLAY)
    try:
        return model.generate(**inputs)
    finally:
        adapter.detach()
