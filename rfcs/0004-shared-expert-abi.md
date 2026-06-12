# RFC 0004: Shared Expert ABI

## Status

Draft boundary, not implemented for execution replay.

## Goal

Represent shared/static expert paths without corrupting the routed top-k ABI.
Models with `routed experts + shared expert` must expose those as separate
semantic paths.

## Core rule

Shared experts are **not** routed top-k experts. They must not be silently
flattened into `RouteRecord.topk_idx` unless a backend-specific adapter declares
that exact namespace semantics and validation can prove equivalence.

## Schema

`SharedExpertRecord` fields:

- `enabled`: whether a shared expert path participated;
- `policy`: `always_on`, `gated`, `scaled`, or backend-specific extension;
- `expert_id`: optional shared expert identifier;
- `gate`: optional tensor/scalar gate;
- `scale`: optional tensor/scalar scale;
- `semantics`: adapter metadata.

`ExpertNamespace` fields:

- `namespace_id`;
- `routed_expert_count`;
- `shared_expert_count`.

## Validation requirements

- If `shared_expert.enabled` and backend lacks `supports_shared_expert`, replay fails closed.
- If `shared_expert_count > 0` and backend lacks `supports_shared_expert`, replay fails closed.
- R2 replay with shared experts requires both routed weights and shared path semantics.
- R3 dispatch replay must validate whether the shared expert is dispatched with routed experts or executed as a separate dense/static path.
- Shared expert gate/scale dtype, shape, and normalization semantics must match exactly.

## Qwen3.5 / Qwen3.6 direction

Qwen-style shared experts should be modeled as:

```text
routed path: topk_idx/topk_weights over routed expert namespace
shared path: SharedExpertRecord(policy="gated" or "always_on")
```

The initial HF Qwen3MoE adapter does not implement this and must fail closed on
shared expert records.
