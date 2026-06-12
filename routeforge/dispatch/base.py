"""Dispatcher adapter contract for future R3/R4 backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from routeforge.core import BackendCapabilities, DispatchPlan, RouteRecord, RuntimeContext


class DispatcherAdapter(ABC):
    """Abstract dispatcher adapter.

    Dispatcher adapters lower semantic routes into dispatch layouts. They are not
    enabled by the central replay guard until topology validation is complete.
    """

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return dispatch backend capabilities."""

    @abstractmethod
    def build_dispatch_plan(self, record: RouteRecord, context: RuntimeContext) -> DispatchPlan:
        """Build a backend-specific DispatchPlan from a validated RouteRecord."""
