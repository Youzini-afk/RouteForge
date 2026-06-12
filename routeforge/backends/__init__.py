"""Model and framework adapters."""

from .base import BackendAdapter
from .hf_qwen3_moe import HFQwen3MoeAdapter, SparseMoeBlockRef
from .negotiation import CapabilityNegotiationResult, negotiate_capabilities
from .registry import (
    DEFAULT_ADAPTER_REGISTRY,
    AdapterRegistration,
    AdapterRegistry,
    create_adapter,
    list_adapters,
    register_adapter,
)

register_adapter(
    HFQwen3MoeAdapter.backend_id,
    HFQwen3MoeAdapter,
    description="Hugging Face Qwen3MoE Python reference adapter",
)

__all__ = [
    "AdapterRegistration",
    "AdapterRegistry",
    "BackendAdapter",
    "CapabilityNegotiationResult",
    "DEFAULT_ADAPTER_REGISTRY",
    "HFQwen3MoeAdapter",
    "SparseMoeBlockRef",
    "create_adapter",
    "list_adapters",
    "negotiate_capabilities",
    "register_adapter",
]
