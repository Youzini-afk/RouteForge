# RFC 0006: DeepEP Dispatcher Adapter

## Status

Draft boundary, not implemented.

## Positioning

DeepEP is a `DispatcherAdapter`, not the RouteForge core ABI. It consumes a
validated `RouteRecord` and produces or consumes `DispatchPlan` / backend handle
metadata under strict topology checks.

## Mapping

```text
RouteRecord.topk_idx      -> DeepEP topk_idx
RouteRecord.topk_weights  -> DeepEP topk_weights
DispatchPlan              -> token permutation / expert offsets / counts
```

DeepEP handles are not portable route decisions.

## R3 dispatch replay requirements

- DeepEP version compatibility;
- rank count compatibility;
- communication group identity;
- expert placement and rank mapping;
- capacity/padding/drop semantics;
- token permutation/inverse permutation integrity;
- dtype/layout compatibility.

## R4 handle replay requirements

R4 is disabled by default. Any future handle replay requires exact matching of:

- DeepEP version;
- CUDA/runtime/kernel compatibility;
- process group and rank topology;
- expert placement;
- token count and layout;
- CUDA graph state if applicable.

Any drift falls back to R3/R2 or fails closed. Handles must not be serialized in
RouteTape v0.
