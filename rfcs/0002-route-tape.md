# RFC 0002: RouteTape v0

## Status

Draft, implementation target for the first storage layer.

## Goal

Persist route decisions as runtime-consumable tensor artifacts. `RouteTape` is not a trace log; it is a replay substrate.

## Format v0

Use a directory container for debuggability and safe incremental evolution:

```text
request_001.mrt/
  manifest.json
  index.json
  chunks/
    000000.safetensors
    000001.safetensors
```

Metadata sidecars are JSON. Token-level route payloads are binary tensors.

## Manifest

Manifest fields include:

- tape version;
- ABI version;
- created by/version;
- recorded replay level;
- model architecture/config digest when known;
- backend id;
- route semantics;
- chunk index summary;
- integrity checks.

## Tensor chunks

Each chunk stores tensors for one or more `RouteRecord`s. MVP may store one record per chunk for clarity. Later versions can stream append multiple records per chunk.

Required tensor payload:

- `topk_idx`;
- `topk_weights`;
- token identity tensors when available.

Small scalar metadata for a record is represented in `index.json`, not repeated per token.

## Rules

- No Python pickle.
- No JSON token-level route payloads.
- No backend handle serialization in v0.
- Corruption or missing chunks fail closed.
