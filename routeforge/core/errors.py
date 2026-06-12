"""Typed RouteForge errors."""

from __future__ import annotations


class RouteForgeError(Exception):
    """Base class for all RouteForge errors."""


class ValidationError(RouteForgeError):
    """Base class for ABI/runtime compatibility validation failures."""


class ShapeMismatchError(ValidationError):
    """Tensor shape or token-count mismatch."""


class TopKMismatchError(ValidationError):
    """`top_k` metadata disagrees with route tensors or runtime context."""


class NumExpertsError(ValidationError):
    """Expert-count metadata is invalid or incompatible."""


class WeightSemanticsError(ValidationError):
    """Route weights are missing or semantically incompatible."""


class CompatibilityError(ValidationError):
    """Generic fail-closed compatibility failure."""


class ReplayPolicyError(RouteForgeError):
    """A replay policy rejected a route record."""


class UnsupportedReplayLevelError(ReplayPolicyError):
    """Requested replay level is not supported by this RouteForge phase/backend."""
