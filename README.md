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

## Current implementation status

- Core Route ABI dataclasses and fail-closed validation are implemented.
- RouteTape v0 writes binary tensor chunks with JSON metadata sidecars.
- HF Qwen3MoE reference adapter records/replays the Python sparse block boundary: gate output (`topk_idx`, `topk_weights`) before expert dispatch.
- R3+ dispatch replay, shared expert replay, vLLM/SGLang integrations, and DeepEP handle replay are design-reserved but not enabled.

## Current limitations

- The HF Qwen3MoE adapter is a **reference Python adapter**, not a fused serving adapter.
- It does not provide serving-grade token alignment by itself. Production replay must be driven by a `RuntimeContext` from a serving runtime that can prove request/sequence/token-position identity.
- The adapter uses a documented weak-alignment policy only for local HF reference workflows.
- R1 index-only replay is representable in the ABI, but HF sparse block execution requires route weights; the adapter rejects R1 execution instead of fabricating uniform weights.
- `output_router_logits` is diagnostic-only and is not rewritten during replay.
- Shared expert replay, DeepEP R3 dispatch replay, R4 handle replay, and R5 planned replay are intentionally disabled until their RFC validation requirements are implemented.
- RouteTape v0 uses binary tensor chunks and integrity checks. The long-term production format target is safetensors plus richer checksums/manifest compatibility metadata.
