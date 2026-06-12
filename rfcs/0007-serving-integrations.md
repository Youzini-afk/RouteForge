# RFC 0007: vLLM/SGLang serving integrations

## Status

Placeholder for Phase 4.

Serving integrations must be designed around TopK/FusedMoE/token-dispatcher/executor boundaries and strict token alignment. Continuous batching, prefix cache, speculative decoding, and scheduler reorder make silent replay unsafe.
