"""DeepEP dispatcher boundary stub."""

from __future__ import annotations

from routeforge.core import BackendCapabilities, DispatchPlan, ReplayLevel, RouteRecord, RuntimeContext

from .base import DispatcherAdapter


class DeepEPDispatcherAdapter(DispatcherAdapter):
    """Non-executing DeepEP adapter stub.

    It declares the boundary and fails closed for actual plan construction until
    DeepEP topology/version checks are implemented.
    """

    backend_id = "deepep"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            supported_replay_levels=frozenset(),
            supports_index_replay=True,
            supports_weight_replay=True,
            supports_dispatch_plan=False,
            unsafe_cases=("deepep_dispatch_plan_construction_disabled",),
        )

    def build_dispatch_plan(self, record: RouteRecord, context: RuntimeContext) -> DispatchPlan:
        raise NotImplementedError("DeepEP dispatch plan construction is not enabled in this phase")
