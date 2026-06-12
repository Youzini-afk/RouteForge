"""Small user-facing runtime facade.

`MoeReplayRuntime` wires modes, adapters, and RouteTape storage without moving
model-specific logic into core. It is deliberately conservative: it supports the
HF reference adapter workflow and leaves serving-grade RuntimeContext provision
to future integrations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import BackendAdapter, create_adapter, negotiate_capabilities
from .core import ReplayLevel, ReplayMode, RouteRecord
from .storage import RouteTapeReader, RouteTapeWriter


class AttachedRuntime:
    """Context-manager handle returned by `MoeReplayRuntime.attach`."""

    def __init__(self, runtime: "MoeReplayRuntime", adapter: BackendAdapter) -> None:
        self.runtime = runtime
        self.adapter = adapter

    def __enter__(self) -> BackendAdapter:
        self.adapter.set_mode(self.runtime.mode)
        return self.adapter

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self.runtime.mode == ReplayMode.RECORD and self.runtime.tape_path is not None:
                self.runtime.write_records(self.adapter.get_records())
        finally:
            self.adapter.detach()


class MoeReplayRuntime:
    """RouteForge runtime helper for record/replay workflows."""

    def __init__(
        self,
        *,
        mode: ReplayMode | str = ReplayMode.OFF,
        backend: str = "hf_qwen3_moe",
        tape_path: str | Path | None = None,
        tape_id: str = "routeforge_tape",
        model_id: str = "unknown",
    ) -> None:
        self.mode = ReplayMode(mode)
        self.backend = backend
        self.tape_path = Path(tape_path) if tape_path is not None else None
        self.tape_id = tape_id
        self.model_id = model_id

    def attach(self, model: Any, *, backend: str | None = None) -> AttachedRuntime:
        adapter = create_adapter(backend or self.backend, model)
        negotiation = negotiate_capabilities(adapter.capabilities(), replay_level=ReplayLevel.R2, mode=self.mode)
        if self.mode in {ReplayMode.RECORD, ReplayMode.REPLAY} and not negotiation.accepted:
            raise ValueError(negotiation.reason or "backend capability negotiation failed")
        if self.mode == ReplayMode.REPLAY:
            self.load_replay_records(adapter)
        return AttachedRuntime(self, adapter)

    def write_records(self, records: list[RouteRecord]) -> None:
        if self.tape_path is None:
            return
        with RouteTapeWriter(
            self.tape_path,
            tape_id=self.tape_id,
            model_id=self.model_id,
            backend_id=self.backend,
        ) as writer:
            for record in records:
                writer.write_record(record)

    def load_replay_records(self, adapter: BackendAdapter) -> None:
        if self.tape_path is None:
            raise ValueError("REPLAY mode requires tape_path")
        with RouteTapeReader(self.tape_path) as reader:
            for record_id in reader.record_ids():
                record = reader.read_record(record_id)
                adapter.set_replay_record(record.layer_id, record)
