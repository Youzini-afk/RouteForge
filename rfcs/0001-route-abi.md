# RFC 0001: Route ABI v0

## Status

Draft, implementation target for the first RouteForge MVP.

## Goal

Define a model-agnostic runtime contract for MoE routing decisions. The ABI must describe the route decision that expert dispatch actually consumes, not a model-specific debug artifact.

## Non-goals

- Do not define a Qwen-specific schema.
- Do not depend on Hugging Face `output_router_logits=True`.
- Do not require router logits for replay.
- Do not encode DeepEP handles in the core route decision.
- Do not define benchmarking or interpretability probes.

## Core objects

### RouteRecord

`RouteRecord` is one captured routing decision for one layer/phase/step over a logical token set.

Required fields:

- `abi_version`
- `layer_id`
- `phase`: `prefill` or `decode`
- `step_id`
- logical token identity: request ids, optional sequence ids, token positions, optional block ids
- `topk_idx`: `[tokens, top_k]`
- `topk_weights`: `[tokens, top_k]`
- `num_experts`
- `top_k`
- route weight semantics: score function and normalization flag
- expert namespace

Optional fields:

- router logits for diagnostics;
- shared expert record;
- dispatch plan for R3+;
- adapter metadata.

### RuntimeContext

`RuntimeContext` describes the currently executing layer/token set. Replay compares it against a `RouteRecord` before using recorded routes.

### ReplayPolicy

`ReplayPolicy` maps compatibility checks to replay/fallback decisions. The default policy is strict and fail-closed.

### BackendCapabilities

Each adapter declares supported modes, replay levels, dtypes, route semantics, and unsafe cases. Runtime never assumes an adapter can replay.

## Replay levels

- R0 observe: record only.
- R1 index replay: replay expert indices.
- R2 weighted replay: replay expert indices plus route weights.
- R3 dispatch replay: replay/lower token dispatch layout.
- R4 handle replay: reuse backend communication/kernel handles with exact topology checks.
- R5 planned replay: use planner/policy routes not necessarily captured from history.

MVP implements R0-R2 and reserves fields for R3-R5.

## Compatibility principle

Unknown, ambiguous, or mismatched state must not silently replay. It must produce a typed failure or fallback decision with a reason code.
