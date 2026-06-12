"""Typed RouteForge errors."""

from __future__ import annotations

from .reason_codes import ReplayReasonCode, normalize_reason_code


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

    def __init__(self, message: str, reason_code: str | ReplayReasonCode | None = None):
        super().__init__(message)
        self.reason_code = normalize_reason_code(reason_code)


class UnsupportedReplayLevelError(ReplayPolicyError):
    """Requested replay level is not supported by this RouteForge phase/backend."""

    def __init__(
        self,
        message: str,
        reason_code: str | ReplayReasonCode | None = ReplayReasonCode.UNSUPPORTED_REPLAY_LEVEL,
    ):
        super().__init__(message, reason_code)
