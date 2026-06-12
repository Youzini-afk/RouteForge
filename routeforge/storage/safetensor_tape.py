"""RouteTape reader/writer.

Safetensors is the preferred production direction. This phase keeps the public
API stable and uses binary NumPy `.npz` chunks when the optional safetensors
dependency is not installed. Token-level route payload never goes into JSON.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from routeforge.core.enums import ReplayLevel, RoutePhase
from routeforge.core.route_record import ExpertNamespace, RouteRecord, WeightSemantics
from routeforge.core.tensor import tensor_dtype_name, tensor_shape

from .errors import RouteTapeError
from .manifest import TAPE_VERSION, RouteTapeManifest


class RouteTapeWriter:
    """Write RouteRecords into a RouteTape directory container."""

    def __init__(
        self,
        tape_dir: str | Path,
        tape_id: str,
        model_id: str,
        *,
        backend_id: str | None = None,
        abi_version: str = "routeforge.route_abi.v0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tape_dir = Path(tape_dir)
        self.chunks_dir = self.tape_dir / "chunks"
        self.tape_id = tape_id
        self.model_id = model_id
        self.backend_id = backend_id
        self.abi_version = abi_version
        self.metadata = metadata or {}
        self._closed = False
        self._records: dict[str, dict[str, Any]] = {}
        self.tape_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "RouteTapeWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def write_record(self, record: RouteRecord) -> str:
        if self._closed:
            raise RouteTapeError("cannot write_record after RouteTapeWriter.close()")
        if record.record_id in self._records:
            raise RouteTapeError(f"duplicate RouteRecord id: {record.record_id}")

        chunk_name = f"{len(self._records):06d}.npz"
        chunk_path = self.chunks_dir / chunk_name
        tensors: dict[str, Any] = {"topk_idx": _to_numpy(record.topk_idx)}
        if record.topk_weights is not None:
            tensors["topk_weights"] = _to_numpy(record.topk_weights)
        for key, value in (
            ("request_ids", record.request_ids),
            ("sequence_ids", record.sequence_ids),
            ("token_positions", record.token_positions),
            ("block_ids", record.block_ids),
            ("token_order", record.token_order),
        ):
            if value is not None:
                tensors[key] = _to_numpy(value)

        try:
            np.savez(chunk_path, **tensors)
        except Exception as exc:  # pragma: no cover - defensive IO wrapper
            raise RouteTapeError(f"failed to write chunk {chunk_name}: {exc}") from exc

        self._records[record.record_id] = _record_index_entry(record, chunk_name, tensors)
        self._records[record.record_id]["checksum"] = _sha256(chunk_path)
        return record.record_id

    def close(self) -> None:
        if self._closed:
            return
        manifest = RouteTapeManifest(
            tape_id=self.tape_id,
            model_id=self.model_id,
            num_records=len(self._records),
            version=TAPE_VERSION,
            abi_version=self.abi_version,
            backend_id=self.backend_id,
            backend_version=self.metadata.get("backend_version"),
            model_arch=self.metadata.get("model_arch"),
            model_config_digest=self.metadata.get("model_config_digest"),
            recorded_replay_levels=tuple(
                sorted({entry["replay_level"] for entry in self._records.values()})
            ),
            metadata=self.metadata,
        )
        _write_json(self.tape_dir / "manifest.json", manifest.to_dict())
        _write_json(self.tape_dir / "index.json", {"records": self._records})
        self._closed = True


class RouteTapeReader:
    """Read records from a fail-closed RouteTape directory."""

    def __init__(self, tape_dir: str | Path) -> None:
        self.tape_dir = Path(tape_dir)
        self.chunks_dir = self.tape_dir / "chunks"
        self.manifest = self._load_manifest()
        self._index = self._load_index()
        if self.manifest.num_records != len(self._index["records"]):
            raise RouteTapeError(
                f"manifest num_records={self.manifest.num_records} does not match "
                f"index records={len(self._index['records'])}"
            )

    def __enter__(self) -> "RouteTapeReader":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def record_ids(self) -> list[str]:
        return list(self._index["records"].keys())

    def manifest_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        """Return JSON-safe manifest metadata."""

        data = self.manifest.to_dict()
        if not include_metadata:
            data["metadata"] = {"redacted": True, "key_count": len(self.manifest.metadata)}
        return data

    def list_records(self, *, include_metadata: bool = False) -> list[dict[str, Any]]:
        """Return JSON-safe record summaries without loading tensor payloads."""

        return [self.record_summary(record_id, include_metadata=include_metadata) for record_id in self.record_ids()]

    def record_summary(self, record_id: str, *, include_metadata: bool = False) -> dict[str, Any]:
        """Return one JSON-safe record summary from the tape index."""

        records = self._index.get("records", {})
        if record_id not in records:
            raise RouteTapeError(f"RouteRecord id not found: {record_id}")
        entry = dict(records[record_id])
        metadata = entry.get("metadata", {})
        return {
            "record_id": record_id,
            "layer_id": entry.get("layer_id"),
            "replay_level": entry.get("replay_level"),
            "token_count": entry.get("token_count"),
            "top_k": entry.get("top_k"),
            "num_experts": entry.get("num_experts"),
            "phase": entry.get("phase"),
            "step_id": entry.get("step_id"),
            "chunk": entry.get("chunk"),
            "checksum": entry.get("checksum"),
            "tensors": entry.get("tensors", {}),
            "weight_semantics": entry.get("weight_semantics", {}),
            "expert_namespace": entry.get("expert_namespace", {}),
            "metadata": metadata if include_metadata else {"redacted": True, "key_count": len(metadata)},
        }

    def validate_record_integrity(self, record_id: str) -> dict[str, Any]:
        """Validate checksum and tensor metadata for one record.

        This is an operator-facing structural check. It does not perform replay
        compatibility; callers can pass the loaded record through replay_guard.
        """

        records = self._index.get("records", {})
        if record_id not in records:
            raise RouteTapeError(f"RouteRecord id not found: {record_id}")
        entry = records[record_id]
        chunk_path = self.tape_dir / entry["chunk"]
        result: dict[str, Any] = {
            "record_id": record_id,
            "chunk_exists": chunk_path.is_file(),
            "checksum_ok": None,
            "checksum_expected": entry.get("checksum"),
            "checksum_actual": None,
            "tensor_shapes_ok": None,
            "tensor_dtypes_ok": None,
            "status": "pass",
            "message": None,
        }
        if not chunk_path.is_file():
            result.update(status="fail", message=f"RouteTape chunk missing: {chunk_path}")
            return result

        expected_checksum = entry.get("checksum")
        actual_checksum = _sha256(chunk_path)
        result["checksum_actual"] = actual_checksum
        if expected_checksum:
            result["checksum_ok"] = actual_checksum == expected_checksum
            if not result["checksum_ok"]:
                result.update(status="fail", message="RouteTape chunk checksum mismatch")
                return result

        try:
            with np.load(chunk_path, allow_pickle=False) as payload:
                tensor_names = set(payload.files)
                shape_ok = True
                dtype_ok = True
                for name, meta in entry.get("tensors", {}).items():
                    if name not in tensor_names:
                        shape_ok = False
                        dtype_ok = False
                        continue
                    value = payload[name]
                    if list(tensor_shape(value)) != meta.get("shape"):
                        shape_ok = False
                    if tensor_dtype_name(value) != meta.get("dtype"):
                        dtype_ok = False
                result["tensor_shapes_ok"] = shape_ok
                result["tensor_dtypes_ok"] = dtype_ok
                if not shape_ok or not dtype_ok:
                    result.update(status="fail", message="RouteTape tensor metadata mismatch")
        except Exception as exc:
            result.update(
                status="fail",
                tensor_shapes_ok=False,
                tensor_dtypes_ok=False,
                message=f"failed to read RouteTape chunk: {exc}",
            )
        return result

    def read_record(self, record_id: str) -> RouteRecord:
        records = self._index.get("records", {})
        if record_id not in records:
            raise RouteTapeError(f"RouteRecord id not found: {record_id}")
        entry = records[record_id]
        chunk_path = self.tape_dir / entry["chunk"]
        if not chunk_path.is_file():
            raise RouteTapeError(f"RouteTape chunk missing: {chunk_path}")
        expected_checksum = entry.get("checksum")
        if expected_checksum and _sha256(chunk_path) != expected_checksum:
            raise RouteTapeError(f"RouteTape chunk checksum mismatch: {chunk_path}")
        try:
            with np.load(chunk_path, allow_pickle=False) as payload:
                topk_idx = payload["topk_idx"].copy()
                topk_weights = payload["topk_weights"].copy() if "topk_weights" in payload else None
                optional_tensors = {
                    key: payload[key].copy()
                    for key in (
                        "request_ids",
                        "sequence_ids",
                        "token_positions",
                        "block_ids",
                        "token_order",
                    )
                    if key in payload
                }
        except Exception as exc:
            raise RouteTapeError(f"failed to read RouteTape chunk for {record_id}: {exc}") from exc

        _validate_loaded_tensor(entry, "topk_idx", topk_idx)
        if topk_weights is not None:
            _validate_loaded_tensor(entry, "topk_weights", topk_weights)

        try:
            return RouteRecord(
                record_id=record_id,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                layer_id=int(entry["layer_id"]),
                num_experts=int(entry["num_experts"]),
                top_k=int(entry["top_k"]),
                token_count=int(entry["token_count"]),
                replay_level=ReplayLevel(entry["replay_level"]),
                abi_version=entry.get("abi_version", self.manifest.abi_version),
                phase=RoutePhase(entry.get("phase", RoutePhase.DECODE.value)),
                step_id=entry.get("step_id"),
                request_ids=optional_tensors.get("request_ids"),
                sequence_ids=optional_tensors.get("sequence_ids"),
                token_positions=optional_tensors.get("token_positions"),
                block_ids=optional_tensors.get("block_ids"),
                token_order=optional_tensors.get("token_order"),
                weight_semantics=WeightSemantics(**entry.get("weight_semantics", {})),
                expert_namespace=ExpertNamespace(**entry.get("expert_namespace", {})),
                metadata=entry.get("metadata", {}),
            )
        except Exception as exc:
            raise RouteTapeError(f"failed to reconstruct RouteRecord {record_id}: {exc}") from exc

    def _load_manifest(self) -> RouteTapeManifest:
        path = self.tape_dir / "manifest.json"
        if not path.is_file():
            raise RouteTapeError(f"RouteTape manifest missing: {path}")
        try:
            data = json.loads(path.read_text())
            manifest = RouteTapeManifest.from_dict(data)
        except Exception as exc:
            raise RouteTapeError(f"RouteTape manifest is corrupt: {exc}") from exc
        if manifest.version != TAPE_VERSION:
            raise RouteTapeError(
                f"RouteTape version mismatch: expected {TAPE_VERSION}, got {manifest.version}"
            )
        return manifest

    def _load_index(self) -> dict[str, Any]:
        path = self.tape_dir / "index.json"
        if not path.is_file():
            raise RouteTapeError(f"RouteTape index missing: {path}")
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            raise RouteTapeError(f"RouteTape index is corrupt: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("records"), dict):
            raise RouteTapeError("RouteTape index missing records mapping")
        return data


def _record_index_entry(record: RouteRecord, chunk_name: str, tensors: dict[str, Any]) -> dict[str, Any]:
    tensor_meta = {
        name: {"shape": list(tensor_shape(value)), "dtype": tensor_dtype_name(value)}
        for name, value in tensors.items()
    }
    return {
        "record_id": record.record_id,
        "chunk": f"chunks/{chunk_name}",
        "tensors": tensor_meta,
        "abi_version": record.abi_version,
        "layer_id": record.layer_id,
        "phase": str(record.phase.value) if isinstance(record.phase, RoutePhase) else str(record.phase),
        "step_id": record.step_id,
        "token_count": record.token_count,
        "num_experts": record.num_experts,
        "top_k": record.top_k,
        "replay_level": record.replay_level.value,
        "weight_semantics": {
            "score_function": record.weight_semantics.score_function,
            "normalized": record.weight_semantics.normalized,
            "includes_bias": record.weight_semantics.includes_bias,
        },
        "expert_namespace": {
            "namespace_id": record.expert_namespace.namespace_id,
            "routed_expert_count": record.expert_namespace.routed_expert_count,
            "shared_expert_count": record.expert_namespace.shared_expert_count,
        },
        "metadata": record.metadata,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_loaded_tensor(entry: dict[str, Any], name: str, value: Any) -> None:
    meta = entry.get("tensors", {}).get(name)
    if not meta:
        raise RouteTapeError(f"RouteTape index missing tensor metadata for {name}")
    actual_shape = list(tensor_shape(value))
    actual_dtype = tensor_dtype_name(value)
    if actual_shape != meta.get("shape"):
        raise RouteTapeError(
            f"RouteTape tensor shape mismatch for {name}: expected {meta.get('shape')}, got {actual_shape}"
        )
    if actual_dtype != meta.get("dtype"):
        raise RouteTapeError(
            f"RouteTape tensor dtype mismatch for {name}: expected {meta.get('dtype')}, got {actual_dtype}"
        )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch  # type: ignore

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)
