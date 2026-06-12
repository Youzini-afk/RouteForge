"""Backend adapter registry.

The registry keeps model/framework-specific adapter construction outside core.
It is intentionally tiny and explicit: registering an adapter does not imply the
core package knows anything about the model internals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import BackendAdapter


AdapterFactory = Callable[[Any], BackendAdapter]


@dataclass(frozen=True)
class AdapterRegistration:
    backend_id: str
    factory: AdapterFactory
    description: str = ""


class AdapterRegistry:
    """In-memory adapter factory registry."""

    def __init__(self) -> None:
        self._registrations: dict[str, AdapterRegistration] = {}

    def register(self, backend_id: str, factory: AdapterFactory, *, description: str = "") -> None:
        if backend_id in self._registrations:
            raise ValueError(f"backend adapter already registered: {backend_id}")
        self._registrations[backend_id] = AdapterRegistration(backend_id, factory, description)

    def get(self, backend_id: str) -> AdapterRegistration:
        try:
            return self._registrations[backend_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._registrations)) or "<none>"
            raise KeyError(f"unknown backend adapter: {backend_id}; known: {known}") from exc

    def create(self, backend_id: str, model: Any) -> BackendAdapter:
        return self.get(backend_id).factory(model)

    def list_backend_ids(self) -> list[str]:
        return sorted(self._registrations)

    def registrations(self) -> list[AdapterRegistration]:
        return [self._registrations[key] for key in self.list_backend_ids()]


DEFAULT_ADAPTER_REGISTRY = AdapterRegistry()


def register_adapter(backend_id: str, factory: AdapterFactory, *, description: str = "") -> None:
    DEFAULT_ADAPTER_REGISTRY.register(backend_id, factory, description=description)


def create_adapter(backend_id: str, model: Any) -> BackendAdapter:
    return DEFAULT_ADAPTER_REGISTRY.create(backend_id, model)


def list_adapters() -> list[str]:
    return DEFAULT_ADAPTER_REGISTRY.list_backend_ids()
