"""Core RouteForge runtime enums.

The enum names intentionally use the short R0-R5 labels because these levels
are part of the public Route ABI vocabulary.
"""

from __future__ import annotations

from enum import Enum


class ReplayMode(str, Enum):
    """Runtime mode for RouteForge."""

    OFF = "off"
    RECORD = "record"
    REPLAY = "replay"
    POLICY = "policy"
    PLAN_ONLY = "plan_only"


class ReplayLevel(str, Enum):
    """MoE replay semantic tiers."""

    R0 = "r0_observe"
    R1 = "r1_index"
    R2 = "r2_weighted"
    R3 = "r3_dispatch"
    R4 = "r4_handle"
    R5 = "r5_planned"


class RoutePhase(str, Enum):
    """Execution phase that produced or consumes a route record."""

    PREFILL = "prefill"
    DECODE = "decode"


class ReplayDecision(str, Enum):
    """Decision produced by fail-closed replay policy evaluation."""

    USE_LIVE_ROUTER = "use_live_router"
    REPLAY_R1 = "replay_r1"
    REPLAY_R2 = "replay_r2"
    REPLAY_R3 = "replay_r3"
    DOWNGRADE = "downgrade"
    FAIL_CLOSED = "fail_closed"
    HARD_FAIL = "hard_fail"
