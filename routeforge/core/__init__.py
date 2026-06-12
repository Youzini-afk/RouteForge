"""Core model-agnostic RouteForge ABI objects."""

from .capabilities import BackendCapabilities, requires_dispatch, requires_weights
from .context import RuntimeContext
from .dispatch_plan import DispatchPlan, DispatchPlanValidationResult, validate_dispatch_plan
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
from .replay_guard import (
    ReplayDecisionResult,
    TapeCompatibility,
    decide_replay_request,
    validate_replay_request,
)
from .reason_codes import ReplayReasonCode
from .route_record import ExpertNamespace, RouteRecord, SharedExpertRecord, WeightSemantics
from .validation import ValidationResult, validate_compatibility, validate_record

__all__ = [
    "BackendCapabilities",
    "CompatibilityError",
    "DispatchPlan",
    "DispatchPlanValidationResult",
    "ExpertNamespace",
    "NumExpertsError",
    "ReplayDecision",
    "ReplayDecisionResult",
    "ReplayLevel",
    "ReplayMode",
    "ReplayPolicy",
    "ReplayPolicyError",
    "ReplayReasonCode",
    "RouteForgeError",
    "RoutePhase",
    "RouteRecord",
    "RuntimeContext",
    "ShapeMismatchError",
    "SharedExpertRecord",
    "TapeCompatibility",
    "TopKMismatchError",
    "UnsupportedReplayLevelError",
    "ValidationError",
    "ValidationResult",
    "WeightSemantics",
    "apply_policy",
    "decide_replay_request",
    "requires_dispatch",
    "requires_weights",
    "validate_compatibility",
    "validate_dispatch_plan",
    "validate_record",
    "validate_replay_request",
]
