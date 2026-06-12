"""Core model-agnostic RouteForge ABI objects."""

from .capabilities import BackendCapabilities, requires_dispatch, requires_weights
from .context import RuntimeContext
from .dispatch_plan import DispatchPlan
from .enums import ReplayDecision, ReplayLevel, ReplayMode, RoutePhase
from .errors import (
    CompatibilityError,
    NumExpertsError,
    ReplayPolicyError,
    RouteForgeError,
    ShapeMismatchError,
    TopKMismatchError,
    UnsupportedReplayLevelError,
    ValidationError,
    WeightSemanticsError,
)
from .replay_policy import ReplayPolicy, apply_policy
from .route_record import ExpertNamespace, RouteRecord, SharedExpertRecord, WeightSemantics
from .validation import ValidationResult, validate_compatibility, validate_record

__all__ = [
    "BackendCapabilities",
    "CompatibilityError",
    "DispatchPlan",
    "ExpertNamespace",
    "NumExpertsError",
    "ReplayDecision",
    "ReplayLevel",
    "ReplayMode",
    "ReplayPolicy",
    "ReplayPolicyError",
    "RouteForgeError",
    "RoutePhase",
    "RouteRecord",
    "RuntimeContext",
    "ShapeMismatchError",
    "SharedExpertRecord",
    "TopKMismatchError",
    "UnsupportedReplayLevelError",
    "ValidationError",
    "ValidationResult",
    "WeightSemantics",
    "apply_policy",
    "requires_dispatch",
    "requires_weights",
    "validate_compatibility",
    "validate_record",
]
