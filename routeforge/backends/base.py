"""Backend adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from routeforge.core.capabilities import BackendCapabilities
from routeforge.core.enums import ReplayMode
from routeforge.core.route_record import RouteRecord


class BackendAdapter(ABC):
    """Abstract contract for model/framework adapters.

    Adapters translate backend-specific router/dispatch boundaries into the
    model-agnostic RouteForge ABI. Core code must not special-case model names.
    """

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return this adapter's explicit support matrix."""

    @abstractmethod
    def discover_sparse_moe_blocks(self) -> list[Any]:
        """Return discovered sparse MoE blocks or block descriptors."""

    @abstractmethod
    def set_mode(self, mode: ReplayMode) -> None:
        """Set runtime mode and install hooks when needed."""

    @abstractmethod
    def current_mode(self) -> ReplayMode:
        """Return the current runtime mode."""

    @abstractmethod
    def get_records(self, layer_id: int | None = None) -> list[RouteRecord]:
        """Return captured route records, optionally filtered by layer."""

    @abstractmethod
    def set_replay_record(self, layer_id: int, record: RouteRecord) -> None:
        """Set the record consumed by the given layer during replay."""

    @abstractmethod
    def detach(self) -> None:
        """Remove installed hooks and restore original backend behavior."""
