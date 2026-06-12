# RFC 0005: DispatchPlan v0

## Status

Draft boundary, R3 not enabled in MVP.

## Separation of concerns

`RouteRecord` is a semantic route decision:

```text
token identity + layer/phase + topk_idx + topk_weights + route semantics
```

`DispatchPlan` is a lowered dispatch layout:

```text
token permutation + expert counts/offsets + capacity/padding/drop + topology
```

The two objects must not be conflated. A route may be valid while a dispatch
layout is invalid for the current topology.

## R3 fields

- token permutation;
- inverse permutation;
- expert counts;
- expert offsets;
- capacity;
- padding/drop mask;
- rank mapping;
- expert placement;
- communication group id;
- topology signature;
- dispatcher backend/version.

## Validation requirements

- R3 replay requires exact token count, expert count, top-k, capacity, and permutation compatibility.
- R3 replay requires exact rank mapping and expert placement compatibility.
- R3 replay fails closed if dispatcher backend/version differs unless an adapter explicitly declares compatibility.
- `DispatchPlan` v0 must not serialize backend handles.

## MVP behavior

Requests for R3/R4/R5 are rejected by the central replay guard. Phase 1-4 code
may define schemas but must not pretend dispatch replay is available.
