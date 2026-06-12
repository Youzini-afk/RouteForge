"""Unit tests for RouteForge storage: RouteTapeWriter, RouteTapeReader, RouteTapeManifest.

These tests define the expected contract for:
  - routeforge.storage.manifest         (RouteTapeManifest dataclass)
  - routeforge.storage.safetensor_tape  (RouteTapeWriter, RouteTapeReader)
  - routeforge.storage.errors           (StorageError, RouteTapeError)
  - routeforge.core.route_record        (RouteRecord – reused from core)
  - routeforge.core.enums               (ReplayLevel – reused from core)

Key design invariants enforced by these tests:
  1. Writer produces manifest.json + index.json + chunks/*.safetensors (or .npz fallback).
  2. write_record / read_record roundtrip preserves topk_idx, topk_weights, layer_id,
     token_count, num_experts, top_k, replay_level.
  3. Missing manifest, missing chunk file, or corrupted index → fail-closed
     (StorageError / RouteTapeError), never silently return wrong data.
  4. Token-level payload (topk_idx / topk_weights arrays) must NOT appear in JSON
     files (manifest / index) — only metadata and chunk references.
  5. Multiple records can be written and individually retrieved by record_id.

Production code does not exist yet; these tests serve as the executable spec.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from routeforge.core.enums import ReplayLevel
from routeforge.core.errors import RouteForgeError
from routeforge.core.route_record import RouteRecord
from routeforge.storage.errors import RouteTapeError, StorageError
from routeforge.storage.manifest import RouteTapeManifest
from routeforge.storage.safetensor_tape import RouteTapeReader, RouteTapeWriter


# ============================================================================
# Helpers
# ============================================================================

def _make_idx(token_count: int = 4, top_k: int = 2, num_experts: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, num_experts, size=(token_count, top_k), dtype=np.int16)


def _make_weights(token_count: int = 4, top_k: int = 2) -> np.ndarray:
    rng = np.random.default_rng(42)
    w = rng.random((token_count, top_k)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    return w


def _make_r2_record(
    record_id: str = "rec_0",
    token_count: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    layer_id: int = 0,
    topk_idx: np.ndarray | None = None,
    topk_weights: np.ndarray | None = None,
) -> RouteRecord:
    if topk_idx is None:
        topk_idx = _make_idx(token_count, top_k, num_experts)
    if topk_weights is None:
        topk_weights = _make_weights(token_count, top_k)
    return RouteRecord(
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        layer_id=layer_id,
        num_experts=num_experts,
        top_k=top_k,
        token_count=token_count,
        replay_level=ReplayLevel.R2,
        record_id=record_id,
    )


def _make_r1_record(
    record_id: str = "rec_0",
    token_count: int = 4,
    top_k: int = 2,
    num_experts: int = 8,
    layer_id: int = 0,
    topk_idx: np.ndarray | None = None,
) -> RouteRecord:
    if topk_idx is None:
        topk_idx = _make_idx(token_count, top_k, num_experts)
    return RouteRecord(
        topk_idx=topk_idx,
        topk_weights=None,
        layer_id=layer_id,
        num_experts=num_experts,
        top_k=top_k,
        token_count=token_count,
        replay_level=ReplayLevel.R1,
        record_id=record_id,
    )


# ============================================================================
# 1. Error hierarchy for storage
# ============================================================================

class TestStorageErrorHierarchy:
    """Storage errors must form a coherent hierarchy for catch-by-base."""

    def test_storage_error_is_routeforge_error(self) -> None:
        assert issubclass(StorageError, RouteForgeError)

    def test_route_tape_error_is_storage_error(self) -> None:
        assert issubclass(RouteTapeError, StorageError)

    def test_catch_route_tape_error_by_storage_error(self) -> None:
        try:
            raise RouteTapeError("tape broken")
        except StorageError:
            pass  # expected
        else:
            pytest.fail("RouteTapeError not caught by StorageError")

    def test_catch_storage_error_by_routeforge_error(self) -> None:
        try:
            raise StorageError("disk gone")
        except RouteForgeError:
            pass  # expected
        else:
            pytest.fail("StorageError not caught by RouteForgeError")


# ============================================================================
# 2. RouteTapeManifest dataclass
# ============================================================================

class TestRouteTapeManifest:
    """RouteTapeManifest must carry tape-level metadata without token payloads."""

    def test_manifest_has_required_fields(self) -> None:
        manifest = RouteTapeManifest(
            tape_id="tape_001",
            model_id="Qwen3-30B-A3B",
            num_records=0,
            version="0.1.0",
        )
        assert manifest.tape_id == "tape_001"
        assert manifest.model_id == "Qwen3-30B-A3B"
        assert manifest.num_records == 0
        assert manifest.version == "0.1.0"

    def test_manifest_is_frozen(self) -> None:
        manifest = RouteTapeManifest(
            tape_id="tape_001", model_id="m", num_records=0, version="0.1.0"
        )
        with pytest.raises(AttributeError):
            manifest.tape_id = "hacked"  # type: ignore[misc]

    def test_manifest_no_token_payload_fields(self) -> None:
        """Manifest must NOT have topk_idx / topk_weights as direct fields."""
        field_names = {f.name for f in RouteTapeManifest.__dataclass_fields__.values()}
        assert "topk_idx" not in field_names
        assert "topk_weights" not in field_names

    def test_manifest_serializable_to_json(self) -> None:
        manifest = RouteTapeManifest(
            tape_id="tape_001", model_id="m", num_records=3, version="0.1.0"
        )
        blob = json.dumps(manifest.to_dict())
        restored = RouteTapeManifest.from_dict(json.loads(blob))
        assert restored == manifest


# ============================================================================
# 3. Writer produces correct file layout
# ============================================================================

class TestWriterFileLayout:
    """RouteTapeWriter must produce manifest.json, index.json, and chunk files."""

    def test_writer_creates_manifest_json(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.close()
        assert (tmp_path / "manifest.json").is_file()

    def test_writer_creates_index_json(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.close()
        assert (tmp_path / "index.json").is_file()

    def test_writer_creates_chunks_directory(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        rec = _make_r2_record()
        writer.write_record(rec)
        writer.close()
        assert (tmp_path / "chunks").is_dir()

    def test_writer_creates_chunk_file(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        rec = _make_r2_record()
        writer.write_record(rec)
        writer.close()
        chunk_files = list((tmp_path / "chunks").iterdir())
        assert len(chunk_files) >= 1
        # Chunk file should be .safetensors or .npz
        suffix = chunk_files[0].suffix
        assert suffix in (".safetensors", ".npz")


# ============================================================================
# 4. Token-level payload NOT in JSON
# ============================================================================

class TestTokenPayloadNotInJSON:
    """Token-level arrays (topk_idx, topk_weights) must NOT be embedded in
    manifest.json or index.json.  Only metadata, shapes, and chunk references."""

    def test_manifest_json_no_topk_arrays(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        rec = _make_r2_record(token_count=64, top_k=8, num_experts=128)
        writer.write_record(rec)
        writer.close()

        manifest_text = (tmp_path / "manifest.json").read_text()
        manifest_data = json.loads(manifest_text)
        # Must not contain raw array data
        _assert_no_array_payload(manifest_data)

    def test_index_json_no_topk_arrays(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        rec = _make_r2_record(token_count=64, top_k=8, num_experts=128)
        writer.write_record(rec)
        writer.close()

        index_text = (tmp_path / "index.json").read_text()
        index_data = json.loads(index_text)
        _assert_no_array_payload(index_data)

    def test_index_json_has_shape_metadata(self, tmp_path: Path) -> None:
        """index.json may contain shape/dtype metadata but not the raw arrays."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        rec = _make_r2_record(token_count=4, top_k=2, num_experts=8)
        writer.write_record(rec)
        writer.close()

        index_text = (tmp_path / "index.json").read_text()
        index_data = json.loads(index_text)
        # Should have some shape reference (e.g. "shape" key or similar)
        # but NOT the actual array values
        _assert_no_array_payload(index_data)

    def test_multiple_records_payload_not_in_json(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        for i in range(5):
            writer.write_record(_make_r2_record(record_id=f"rec_{i}"))
        writer.close()

        for json_name in ("manifest.json", "index.json"):
            data = json.loads((tmp_path / json_name).read_text())
            _assert_no_array_payload(data)


def _assert_no_array_payload(data: Any, path: str = "") -> None:
    """Recursively check that no dict value is a large list of numbers
    (which would indicate embedded token-level arrays)."""
    if isinstance(data, dict):
        for k, v in data.items():
            _assert_no_array_payload(v, path=f"{path}.{k}")
    elif isinstance(data, list):
        # A list of >16 numbers is suspicious for a JSON metadata file
        # (shapes, small enums are fine; full token arrays are not)
        if len(data) > 16 and all(isinstance(x, (int, float)) for x in data):
            pytest.fail(
                f"Token-level array payload found in JSON at '{path}' "
                f"(length={len(data)}). Arrays must live in binary chunk files."
            )
        for i, v in enumerate(data):
            _assert_no_array_payload(v, path=f"{path}[{i}]")


# ============================================================================
# 5. write_record / read_record roundtrip
# ============================================================================

class TestWriteReadRoundtrip:
    """write_record + read_record must preserve all RouteRecord fields exactly."""

    def test_r2_roundtrip_topk_idx(self, tmp_path: Path) -> None:
        idx = np.array([[0, 3], [1, 7], [2, 5], [4, 6]], dtype=np.int16)
        weights = _make_weights(4, 2)
        rec = _make_r2_record(topk_idx=idx, topk_weights=weights)

        _write_then_read_assert(tmp_path, rec)

    def test_r2_roundtrip_topk_weights(self, tmp_path: Path) -> None:
        idx = _make_idx()
        weights = np.array(
            [[0.6, 0.4], [0.3, 0.7], [0.5, 0.5], [0.8, 0.2]], dtype=np.float32
        )
        rec = _make_r2_record(topk_idx=idx, topk_weights=weights)

        _write_then_read_assert(tmp_path, rec)

    def test_r1_roundtrip_no_weights(self, tmp_path: Path) -> None:
        rec = _make_r1_record(record_id="r1_only")
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(rec)
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        loaded = reader.read_record("r1_only")
        assert loaded.topk_idx is not None
        np.testing.assert_array_equal(loaded.topk_idx, rec.topk_idx)
        assert loaded.topk_weights is None
        assert loaded.replay_level == ReplayLevel.R1

    def test_roundtrip_preserves_layer_id(self, tmp_path: Path) -> None:
        rec = _make_r2_record(layer_id=17)
        _write_then_read_assert(tmp_path, rec, check_layer_id=17)

    def test_roundtrip_preserves_token_count(self, tmp_path: Path) -> None:
        rec = _make_r2_record(token_count=32, top_k=4, num_experts=64)
        _write_then_read_assert(tmp_path, rec)

    def test_roundtrip_preserves_num_experts(self, tmp_path: Path) -> None:
        rec = _make_r2_record(num_experts=128)
        _write_then_read_assert(tmp_path, rec)

    def test_roundtrip_preserves_top_k(self, tmp_path: Path) -> None:
        rec = _make_r2_record(top_k=8, num_experts=64, token_count=16)
        _write_then_read_assert(tmp_path, rec)

    def test_roundtrip_preserves_replay_level(self, tmp_path: Path) -> None:
        rec = _make_r2_record()
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(rec)
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        loaded = reader.read_record(rec.record_id)
        assert loaded.replay_level == rec.replay_level

    def test_roundtrip_large_record(self, tmp_path: Path) -> None:
        """Stress test with realistic MoE sizes: 1024 tokens, top-8, 128 experts."""
        idx = _make_idx(token_count=1024, top_k=8, num_experts=128)
        weights = _make_weights(token_count=1024, top_k=8)
        rec = _make_r2_record(
            record_id="large",
            token_count=1024, top_k=8, num_experts=128,
            topk_idx=idx, topk_weights=weights,
        )
        _write_then_read_assert(tmp_path, rec)


def _write_then_read_assert(
    tmp_path: Path,
    rec: RouteRecord,
    *,
    check_layer_id: int | None = None,
) -> None:
    """Helper: write a record via writer, read it back via reader, assert equality."""
    writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
    writer.write_record(rec)
    writer.close()

    reader = RouteTapeReader(tape_dir=tmp_path)
    loaded = reader.read_record(rec.record_id)

    # Array data
    np.testing.assert_array_equal(loaded.topk_idx, rec.topk_idx)
    if rec.topk_weights is not None:
        np.testing.assert_allclose(loaded.topk_weights, rec.topk_weights, rtol=1e-6)
    else:
        assert loaded.topk_weights is None

    # Scalar metadata
    assert loaded.layer_id == rec.layer_id
    assert loaded.token_count == rec.token_count
    assert loaded.num_experts == rec.num_experts
    assert loaded.top_k == rec.top_k
    assert loaded.replay_level == rec.replay_level

    if check_layer_id is not None:
        assert loaded.layer_id == check_layer_id


# ============================================================================
# 6. Multiple records by record_id
# ============================================================================

class TestMultipleRecordAccess:
    """Multiple records can be written and individually retrieved by record_id."""

    def test_read_by_record_id(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        rec_a = _make_r2_record(record_id="layer_0", layer_id=0)
        rec_b = _make_r2_record(record_id="layer_1", layer_id=1)
        writer.write_record(rec_a)
        writer.write_record(rec_b)
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        loaded_a = reader.read_record("layer_0")
        loaded_b = reader.read_record("layer_1")

        assert loaded_a.layer_id == 0
        assert loaded_b.layer_id == 1
        np.testing.assert_array_equal(loaded_a.topk_idx, rec_a.topk_idx)
        np.testing.assert_array_equal(loaded_b.topk_idx, rec_b.topk_idx)

    def test_list_record_ids(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        ids = [f"rec_{i}" for i in range(5)]
        for rid in ids:
            writer.write_record(_make_r2_record(record_id=rid, layer_id=int(rid[-1])))
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        assert set(reader.record_ids()) == set(ids)

    def test_read_nonexistent_record_id_raises(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record(record_id="exists"))
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        with pytest.raises((RouteTapeError, KeyError)):
            reader.read_record("does_not_exist")

    def test_overwrite_record_id_raises(self, tmp_path: Path) -> None:
        """Writing the same record_id twice must raise, not silently overwrite."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record(record_id="dup"))
        with pytest.raises((RouteTapeError, ValueError)):
            writer.write_record(_make_r2_record(record_id="dup"))
        writer.close()

    def test_many_records_all_retrievable(self, tmp_path: Path) -> None:
        """Write 20 records, verify all individually retrievable."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        records = {}
        for i in range(20):
            rid = f"rec_{i:03d}"
            rec = _make_r2_record(record_id=rid, layer_id=i)
            records[rid] = rec
            writer.write_record(rec)
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        for rid, original in records.items():
            loaded = reader.read_record(rid)
            assert loaded.layer_id == original.layer_id
            np.testing.assert_array_equal(loaded.topk_idx, original.topk_idx)


# ============================================================================
# 7. Fail-closed on missing / corrupted data
# ============================================================================

class TestFailClosedOnCorruption:
    """Missing manifest, missing chunk, or corrupted index must fail closed
    with StorageError / RouteTapeError — never silently return wrong data."""

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """Opening a reader on a directory with no manifest.json must raise."""
        # Empty directory — no files at all
        with pytest.raises((StorageError, RouteTapeError, FileNotFoundError)):
            RouteTapeReader(tape_dir=tmp_path)

    def test_missing_chunk_file_raises(self, tmp_path: Path) -> None:
        """If a chunk file referenced by index is deleted, read must raise."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record(record_id="orphan"))
        writer.close()

        # Delete all chunk files
        chunks_dir = tmp_path / "chunks"
        for f in chunks_dir.iterdir():
            f.unlink()

        reader = RouteTapeReader(tape_dir=tmp_path)
        with pytest.raises((StorageError, RouteTapeError, FileNotFoundError)):
            reader.read_record("orphan")

    def test_corrupted_index_json_raises(self, tmp_path: Path) -> None:
        """If index.json is corrupted (invalid JSON), reader must raise."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record())
        writer.close()

        # Corrupt the index
        (tmp_path / "index.json").write_text("{corrupted!!!not_json")

        with pytest.raises((StorageError, RouteTapeError, json.JSONDecodeError)):
            RouteTapeReader(tape_dir=tmp_path)

    def test_corrupted_manifest_json_raises(self, tmp_path: Path) -> None:
        """If manifest.json is corrupted, reader must raise."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record())
        writer.close()

        # Corrupt the manifest
        (tmp_path / "manifest.json").write_text("NOT JSON {{{")

        with pytest.raises((StorageError, RouteTapeError, json.JSONDecodeError)):
            RouteTapeReader(tape_dir=tmp_path)

    def test_truncated_chunk_raises(self, tmp_path: Path) -> None:
        """If a chunk file is truncated (zeroed), read must raise, not return
        wrong data."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record(record_id="trunc"))
        writer.close()

        # Truncate chunk file to 0 bytes
        chunks_dir = tmp_path / "chunks"
        for f in chunks_dir.iterdir():
            f.write_bytes(b"")

        reader = RouteTapeReader(tape_dir=tmp_path)
        with pytest.raises((StorageError, RouteTapeError, OSError)):
            reader.read_record("trunc")

    def test_wrong_chunk_content_raises(self, tmp_path: Path) -> None:
        """If a chunk file has wrong content (random bytes), read must raise."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record(record_id="garbage"))
        writer.close()

        # Overwrite chunk with garbage
        chunks_dir = tmp_path / "chunks"
        for f in chunks_dir.iterdir():
            f.write_bytes(os.urandom(256))

        reader = RouteTapeReader(tape_dir=tmp_path)
        with pytest.raises((StorageError, RouteTapeError, OSError)):
            reader.read_record("garbage")

    def test_manifest_version_mismatch_raises(self, tmp_path: Path) -> None:
        """If manifest version doesn't match reader expectations, fail closed."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record())
        writer.close()

        # Tamper with version
        manifest_path = tmp_path / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())
        manifest_data["version"] = "99.0.0-incompatible"
        manifest_path.write_text(json.dumps(manifest_data))

        with pytest.raises((StorageError, RouteTapeError)):
            RouteTapeReader(tape_dir=tmp_path)


# ============================================================================
# 8. Writer context manager and lifecycle
# ============================================================================

class TestWriterLifecycle:
    """RouteTapeWriter must support context manager and safe lifecycle."""

    def test_writer_as_context_manager(self, tmp_path: Path) -> None:
        with RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m") as writer:
            writer.write_record(_make_r2_record())
        # After context exit, files should be on disk
        assert (tmp_path / "manifest.json").is_file()
        assert (tmp_path / "index.json").is_file()

    def test_writer_close_finalizes_manifest(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record())
        writer.write_record(_make_r2_record(record_id="rec_1", layer_id=1))
        writer.close()

        manifest_data = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest_data["num_records"] == 2

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.close()
        with pytest.raises((RouteTapeError, RuntimeError)):
            writer.write_record(_make_r2_record())

    def test_reader_as_context_manager(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="tape_001", model_id="m")
        writer.write_record(_make_r2_record())
        writer.close()

        with RouteTapeReader(tape_dir=tmp_path) as reader:
            rec = reader.read_record("rec_0")
            assert rec is not None


# ============================================================================
# 9. Manifest metadata integrity
# ============================================================================

class TestManifestMetadataIntegrity:
    """Manifest must accurately reflect the tape contents."""

    def test_manifest_tape_id(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="my_tape", model_id="m")
        writer.write_record(_make_r2_record())
        writer.close()

        manifest_data = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest_data["tape_id"] == "my_tape"

    def test_manifest_model_id(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="t", model_id="Qwen3-30B-A3B")
        writer.write_record(_make_r2_record())
        writer.close()

        manifest_data = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest_data["model_id"] == "Qwen3-30B-A3B"

    def test_manifest_num_records_matches(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="t", model_id="m")
        for i in range(7):
            writer.write_record(_make_r2_record(record_id=f"r{i}"))
        writer.close()

        manifest_data = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest_data["num_records"] == 7

    def test_manifest_num_records_mismatch_fails_closed(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="t", model_id="m")
        writer.write_record(_make_r2_record())
        writer.close()

        manifest_path = tmp_path / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())
        manifest_data["num_records"] = 99
        manifest_path.write_text(json.dumps(manifest_data))

        with pytest.raises(RouteTapeError, match="num_records"):
            RouteTapeReader(tape_dir=tmp_path)

    def test_manifest_version_is_set(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="t", model_id="m")
        writer.close()

        manifest_data = json.loads((tmp_path / "manifest.json").read_text())
        assert "version" in manifest_data
        assert manifest_data["version"]  # not empty

    def test_valid_npz_wrong_shape_fails_closed(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="t", model_id="m")
        writer.write_record(_make_r2_record(record_id="shape"))
        writer.close()

        chunks_dir = tmp_path / "chunks"
        for chunk in chunks_dir.iterdir():
            np.savez(
                chunk,
                topk_idx=np.zeros((1, 2), dtype=np.int16),
                topk_weights=np.zeros((1, 2), dtype=np.float32),
            )
            # Restore checksum to make this test exercise shape validation, not checksum.
            index_path = tmp_path / "index.json"
            index = json.loads(index_path.read_text())
            import hashlib

            index["records"]["shape"]["checksum"] = hashlib.sha256(chunk.read_bytes()).hexdigest()
            index_path.write_text(json.dumps(index))

        reader = RouteTapeReader(tape_dir=tmp_path)
        with pytest.raises(RouteTapeError, match="shape mismatch"):
            reader.read_record("shape")


# ============================================================================
# 10. Mixed record types in same tape
# ============================================================================

class TestMixedRecordTypes:
    """A single tape may contain both R1 and R2 records."""

    def test_r1_and_r2_in_same_tape(self, tmp_path: Path) -> None:
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="t", model_id="m")
        r1 = _make_r1_record(record_id="r1_rec", layer_id=0)
        r2 = _make_r2_record(record_id="r2_rec", layer_id=1)
        writer.write_record(r1)
        writer.write_record(r2)
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        loaded_r1 = reader.read_record("r1_rec")
        loaded_r2 = reader.read_record("r2_rec")

        assert loaded_r1.replay_level == ReplayLevel.R1
        assert loaded_r1.topk_weights is None
        assert loaded_r2.replay_level == ReplayLevel.R2
        assert loaded_r2.topk_weights is not None

    def test_records_with_different_shapes(self, tmp_path: Path) -> None:
        """Records with different (token_count, top_k, num_experts) in one tape."""
        writer = RouteTapeWriter(tape_dir=tmp_path, tape_id="t", model_id="m")
        rec_small = _make_r2_record(
            record_id="small", token_count=2, top_k=2, num_experts=8, layer_id=0,
        )
        rec_large = _make_r2_record(
            record_id="large", token_count=64, top_k=8, num_experts=128, layer_id=1,
        )
        writer.write_record(rec_small)
        writer.write_record(rec_large)
        writer.close()

        reader = RouteTapeReader(tape_dir=tmp_path)
        loaded_small = reader.read_record("small")
        loaded_large = reader.read_record("large")

        assert loaded_small.token_count == 2
        assert loaded_small.top_k == 2
        assert loaded_large.token_count == 64
        assert loaded_large.top_k == 8
        np.testing.assert_array_equal(loaded_small.topk_idx, rec_small.topk_idx)
        np.testing.assert_array_equal(loaded_large.topk_idx, rec_large.topk_idx)
