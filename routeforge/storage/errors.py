"""Storage-specific RouteForge errors."""

from __future__ import annotations

from routeforge.core.errors import RouteForgeError


class StorageError(RouteForgeError):
    """Base class for RouteForge persistence failures."""


class RouteTapeError(StorageError):
    """RouteTape container is missing, corrupt, or incompatible."""
