"""Small tensor-like helpers without hard depending on a tensor framework."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def tensor_shape(value: Any) -> tuple[int, ...]:
    """Return a tensor-like object's shape as a tuple.

    Supports NumPy arrays, PyTorch tensors, and nested Python sequences. The
    Python sequence fallback is intentionally small and only meant for metadata
    tests and adapter glue; production tensors should expose `.shape`.
    """

    if value is None:
        return ()
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Iterable):
        seq = list(value)
        if not seq:
            return (0,)
        first_shape = tensor_shape(seq[0])
        return (len(seq), *first_shape)
    return ()


def tensor_dtype_name(value: Any) -> str | None:
    """Return a normalized dtype name for a tensor-like object."""

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    name = getattr(dtype, "name", None) or str(dtype)
    # torch dtypes stringify as "torch.float32"; normalize to "float32".
    return name.rsplit(".", 1)[-1]


def tensor_min(value: Any) -> Any:
    if hasattr(value, "min"):
        result = value.min()
        return result.item() if hasattr(result, "item") else result
    flat = _flatten(value)
    return min(flat) if flat else None


def tensor_max(value: Any) -> Any:
    if hasattr(value, "max"):
        result = value.max()
        return result.item() if hasattr(result, "item") else result
    flat = _flatten(value)
    return max(flat) if flat else None


def tensor_all_finite(value: Any) -> bool:
    """Return True if every scalar is finite."""

    try:
        import numpy as np  # type: ignore

        return bool(np.isfinite(value).all())
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all().item())
    except Exception:
        pass
    flat = _flatten(value)
    return all(item == item and item not in (float("inf"), float("-inf")) for item in flat)


def tensor_equal(left: Any, right: Any) -> bool:
    """Framework-light equality for arrays/tensors/sequences."""

    if left is right:
        return True
    if left is None or right is None:
        return left is right
    try:
        import numpy as np  # type: ignore

        return bool(np.array_equal(left, right))
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.is_tensor(left) and torch.is_tensor(right):
            return bool(torch.equal(left, right))
    except Exception:
        pass
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    return left == right


def tensor_hash(value: Any) -> int:
    """Best-effort stable hash for immutable dataclasses holding arrays."""

    if value is None:
        return hash(None)
    shape = tensor_shape(value)
    dtype = tensor_dtype_name(value)
    if hasattr(value, "tobytes"):
        payload = value.tobytes()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        payload = value.detach().cpu().numpy().tobytes()
    elif hasattr(value, "tolist"):
        payload = repr(value.tolist()).encode()
    else:
        payload = repr(value).encode()
    return hash((shape, dtype, payload))


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]
