# RFC 0004: Shared Expert ABI

## Status

Placeholder for Phase 4.

Shared experts must be modeled separately from routed top-k experts. Qwen3.5-style `routed experts + gated shared expert` should not be flattened into the routed `topk_idx` namespace unless a specific backend explicitly encodes it that way.
