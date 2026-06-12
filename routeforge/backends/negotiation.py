"""Capability negotiation helpers for backend adapters."""

from __future__ import annotations

from dataclasses import dataclass

from routeforge.core import BackendCapabilities, ReplayLevel, ReplayMode


@dataclass(frozen=True)
class CapabilityNegotiationResult:
    """Result of checking whether a backend can satisfy a replay request."""

    accepted: bool
    backend_id: str
    replay_level: ReplayLevel
    mode: ReplayMode
    reason: str | None = None


def negotiate_capabilities(
    capabilities: BackendCapabilities,
    *,
    replay_level: ReplayLevel,
    mode: ReplayMode = ReplayMode.REPLAY,
) -> CapabilityNegotiationResult:
    """Check replay level and mode support before attaching an adapter."""

    if mode not in capabilities.supported_modes:
        return CapabilityNegotiationResult(
            accepted=False,
            backend_id=capabilities.backend_id,
            replay_level=replay_level,
            mode=mode,
            reason=f"backend {capabilities.backend_id} does not support mode {mode.value}",
        )
    if not capabilities.supports_level(replay_level):
        return CapabilityNegotiationResult(
            accepted=False,
            backend_id=capabilities.backend_id,
            replay_level=replay_level,
            mode=mode,
            reason=f"backend {capabilities.backend_id} does not support replay level {replay_level.name}",
        )
    return CapabilityNegotiationResult(
        accepted=True,
        backend_id=capabilities.backend_id,
        replay_level=replay_level,
        mode=mode,
    )
