"""DispatchPlan schema and dispatcher stub tests."""

from __future__ import annotations

import numpy as np
import pytest

from routeforge.core import DispatchPlan, validate_dispatch_plan
from routeforge.dispatch import DeepEPDispatcherAdapter, DispatcherAdapter


def _plan(**kwargs) -> DispatchPlan:
    base = dict(
        dispatcher_backend="deepep",
        token_permutation=np.array([0, 1, 2, 3], dtype=np.int64),
        inverse_permutation=np.array([0, 1, 2, 3], dtype=np.int64),
        expert_counts=np.array([1, 1, 1, 1], dtype=np.int64),
        expert_offsets=np.array([0, 1, 2, 3], dtype=np.int64),
        capacity=4,
        topology_signature="single-rank-test",
    )
    base.update(kwargs)
    return DispatchPlan(**base)


def test_dispatch_plan_validation_passes() -> None:
    result = validate_dispatch_plan(_plan(), token_count=4, num_experts=4)
    assert result.passed


def test_dispatch_plan_missing_permutation_fails() -> None:
    result = validate_dispatch_plan(_plan(token_permutation=None), token_count=4, num_experts=4)
    assert not result.passed
    assert result.code == "TOKEN_PERMUTATION_MISSING"


def test_dispatch_plan_shape_mismatch_fails() -> None:
    result = validate_dispatch_plan(_plan(expert_counts=np.array([1, 2])), token_count=4, num_experts=4)
    assert not result.passed
    assert result.code == "EXPERT_COUNTS_SHAPE"


def test_deepep_dispatcher_is_stub_and_fails_closed() -> None:
    adapter = DeepEPDispatcherAdapter()
    assert isinstance(adapter, DispatcherAdapter)
    assert adapter.capabilities().supports_dispatch_plan
    with pytest.raises(NotImplementedError):
        adapter.build_dispatch_plan(None, None)  # type: ignore[arg-type]
