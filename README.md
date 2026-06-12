# RouteForge

RouteForge is a **MoE Routing ABI / Replay Runtime**. It makes Mixture-of-Experts routing decisions explicit runtime objects that can be recorded, validated, replayed, exported, imported, and eventually lowered into dispatch/runtime backends.

RouteForge is not a model-specific trace script and is not built around `output_router_logits=True`. The core package is model-agnostic; concrete models and serving engines integrate through adapters.

## Core idea

```text
MoE model / serving engine
        │
        ▼
RouteForge Route ABI
        │
        ├── RouteRecord / RoutePlan
        ├── RouteTape
        ├── ReplayPolicy
        ├── BackendAdapter
        └── DispatcherAdapter
```

Initial replay levels:

- **R0 Observe**: record route decisions without changing execution.
- **R1 Index Replay**: replay top-k expert indices.
- **R2 Weighted Replay**: replay top-k expert indices and route weights.

Future levels:

- **R3 Dispatch Replay**: replay/lower dispatch layout.
- **R4 Handle Replay**: reuse backend communication/kernel handles under strict topology checks.
- **R5 Planned Replay**: use a policy/planner to generate routes or dispatch plans.

## Repository layout

```text
routeforge/
  core/          # Model-agnostic ABI, policy, validation, runtime context
  storage/       # RouteTape manifest + binary tensor storage
  backends/      # Model/backend adapters, starting with HF Qwen3MoE
  dispatch/      # DispatchPlan and dispatcher adapters, e.g. DeepEP later
  policies/      # Route rewrite/planner policies later
  integrations/  # vLLM/SGLang/Megatron RFCs and future integrations
rfcs/            # Design specs that freeze compatibility boundaries
examples/        # Minimal record/replay examples
tests/           # Unit and conformance tests
```

## Safety stance

Replay is correctness-sensitive. RouteForge defaults to **fail closed**:

- no silent partial replay;
- no fuzzy token alignment;
- no implicit dtype/shape repair;
- no R2 replay when route weight semantics differ;
- no R3/R4 replay when topology or backend versions drift.

Unsupported or ambiguous cases must return a reasoned fallback decision or raise a typed validation error.
