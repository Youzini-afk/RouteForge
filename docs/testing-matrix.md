# RouteForge test and capability matrix

## Implemented and enabled

| Area | Status | Tests |
| --- | --- | --- |
| Core RouteRecord / RuntimeContext validation | Enabled | `tests/unit/test_core_abi.py` |
| Replay policy and stable decision reason codes | Enabled | `tests/unit/test_replay_policy.py`, `tests/conformance/test_replay_decisions.py` |
| RouteTape binary storage + integrity | Enabled | `tests/storage/test_route_tape.py`, `tests/conformance/test_storage_guard_roundtrip.py` |
| RouteTape CLI inspect/validate | Enabled | `tests/cli/test_tape_cli.py` |
| HF Qwen3MoE reference adapter | Enabled for Python reference path | `tests/backends/test_hf_qwen3_moe_adapter.py` |
| Adapter registry + runtime facade | Enabled | `tests/backends/test_registry_runtime.py` |

## Defined but disabled/fail-closed

| Area | Status | Tests |
| --- | --- | --- |
| Shared expert replay | ABI reserved; unsupported backends fail closed | `tests/unit/test_replay_guard.py` |
| R3 DispatchPlan replay | Schema/validator only; central guard rejects R3 | `tests/dispatch/test_dispatch_plan.py` |
| DeepEP dispatcher | Contract/stub only; construction raises `NotImplementedError` | `tests/dispatch/test_dispatch_plan.py` |
| vLLM/SGLang serving integration | RuntimeContext helper only; no engine hook | `tests/integrations/test_serving_context.py` |
| R4 handle replay | RFC only; not implemented | RFC 0006 |
| R5 planned replay | RFC direction only; not implemented | future |

## Safety invariant

Anything not listed as enabled must either:

1. return `ReplayDecision.FAIL_CLOSED`, or
2. raise a typed/explicit error at the adapter boundary.

No unsupported backend path may silently replay.
