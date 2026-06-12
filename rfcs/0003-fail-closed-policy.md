# RFC 0003: Fail-closed replay policy

## Status

Draft, implementation target for MVP validation.

## Rule

Route replay is correctness-sensitive. RouteForge defaults to rejecting replay whenever the runtime cannot prove compatibility.

## Required checks

### ABI/model

- ABI version;
- model architecture/config digest when available;
- layer id/count;
- number of experts;
- top-k;
- expert namespace;
- shared expert semantics;
- weight semantics.

### Token alignment

- phase;
- step id;
- token count;
- request/sequence ids when available;
- token positions;
- block ids when available;
- token order.

### Tensor payload

- shape;
- dtype;
- chunk integrity;
- top-k dimension.

### Dispatch/topology for R3+

- rank count;
- expert placement;
- communication group;
- backend/library version;
- token permutation and capacity semantics.

## Default actions

- Unknown compatibility: fail closed.
- Shape/token mismatch: fail closed.
- Weight semantic mismatch: reject R2, optionally downgrade only if policy permits.
- Backend unsupported level: reject or fallback live router.
- Partial replay: disabled by default.
- Dtype cast: disabled by default.

Every fallback must include a reason code.
