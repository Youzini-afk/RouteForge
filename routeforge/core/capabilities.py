"""Backend capability declarations and replay-level requirement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import ReplayLevel, ReplayMode


def requires_weights(level: ReplayLevel) -> bool:
    """Return whether a replay level requires route weights."""

    return level in {ReplayLevel.R2, ReplayLevel.R3, ReplayLevel.R4, ReplayLevel.R5}


def requires_dispatch(level: ReplayLevel) -> bool:
    """Return whether a replay level requires dispatch/layout metadata."""

    return level in {ReplayLevel.R3, ReplayLevel.R4, ReplayLevel.R5}


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared support matrix for a model/backend adapter."""

    backend_id: str
    backend_version: str | None = None
    supported_modes: frozenset[ReplayMode] = field(
        default_factory=lambda: frozenset({ReplayMode.OFF, ReplayMode.RECORD})
    )
    supported_replay_levels: frozenset[ReplayLevel] = field(
        default_factory=lambda: frozenset({ReplayLevel.R0})
    )
    supports_index_replay: bool = False
    supports_weight_replay: bool = False
    supports_route_recording: bool = True
    supported_index_dtypes: frozenset[str] = field(default_factory=lambda: frozenset({"int16"}))
    supported_weight_dtypes: frozenset[str] = field(default_factory=lambda: frozenset({"float32"}))
    supported_layouts: frozenset[str] = field(default_factory=lambda: frozenset({"dense_2d"}))
    supports_shared_expert: bool = False
    supports_dispatch_plan: bool = False
    supports_handle_replay: bool = False
    supports_prefill: bool = True
    supports_decode: bool = True
    supports_batch: bool = True
    unsafe_cases: tuple[str, ...] = ()

    def supports_level(self, level: ReplayLevel) -> bool:
        if level not in self.supported_replay_levels:
            return False
        if level == ReplayLevel.R1 and not self.supports_index_replay:
            return False
        if level == ReplayLevel.R2 and not self.supports_weight_replay:
            return False
        if requires_dispatch(level) and not self.supports_dispatch_plan:
            return False
        return True
