# RFC 0007: vLLM/SGLang serving integrations

## Status

Draft boundary. `ServingTokenBatch` can build `RuntimeContext`; no real vLLM or
SGLang engine hook is implemented.

## Integration principle

Serving runtimes must provide `RuntimeContext`. The HF reference adapter cannot
prove request/token identity in continuous batching environments.

## Required runtime context

- request id / sequence id;
- token position;
- scheduler slot or token order;
- phase: prefill/decode;
- decode step;
- layer id;
- block/KV metadata when applicable;
- model/config digest;
- backend capability and topology signature.

## Unsafe by default

The following require explicit integration design and must not silently replay:

- continuous batching reorder;
- prefix/paged KV cache reuse;
- speculative decoding accept/reject;
- mixed prefill/decode batches;
- request cancellation/retry;
- scheduler compaction;
- tensor-parallel/expert-parallel topology drift.

## Suggested hook points

vLLM/SGLang integrations should target TopK/FusedMoE/token-dispatcher boundaries,
not `output_router_logits`. The serving adapter should validate a `RouteRecord`
through the central replay guard before passing external routes to fused kernels.

## API direction

Serving APIs may expose a tape id or route session id, but replay must remain
fail-closed. If runtime cannot prove token alignment for a layer/step, it must
live-route or hard fail according to policy.
