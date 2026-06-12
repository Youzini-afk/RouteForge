"""Minimal HF Qwen3MoE recording sketch.

This example is intentionally a sketch: loading a real Qwen3MoE checkpoint is
environment-specific and may require substantial GPU memory. The adapter hooks
the HF Python sparse block at the gate -> experts boundary and records the
actual top-k expert ids/weights consumed by expert dispatch.
"""

from __future__ import annotations

from routeforge.backends import HFQwen3MoeAdapter
from routeforge.core import ReplayMode
from routeforge.storage import RouteTapeWriter


def record_model_routes(model, inputs, tape_dir: str = "request_001.mrt"):
    adapter = HFQwen3MoeAdapter(model)
    adapter.set_mode(ReplayMode.RECORD)
    try:
        outputs = model.generate(**inputs)
        with RouteTapeWriter(tape_dir=tape_dir, tape_id="request_001", model_id=model.config.name_or_path) as writer:
            for record in adapter.get_records():
                writer.write_record(record)
        return outputs
    finally:
        adapter.detach()
