"""Storage roundtrip conformance through replay guard."""

from __future__ import annotations

import numpy as np

from routeforge.core import ReplayDecision, ReplayLevel, TapeCompatibility, decide_replay_request, validate_replay_request
from routeforge.storage import RouteTapeReader, RouteTapeWriter

from .conftest import FakeBackendSpec, fake_context, fake_record


def test_storage_roundtrip_then_validate_replay_request(tmp_path) -> None:
    record = fake_record(record_id="roundtrip")
    with RouteTapeWriter(tmp_path, tape_id="tape", model_id="model") as writer:
        writer.write_record(record)

    with RouteTapeReader(tmp_path) as reader:
        loaded = reader.read_record("roundtrip")

    spec = FakeBackendSpec()
    safe = validate_replay_request(loaded, fake_context(), spec.policy(), spec.capabilities())
    np.testing.assert_array_equal(safe.topk_idx, record.topk_idx)
    assert safe.topk_weights is not None
    assert record.topk_weights is not None
    np.testing.assert_allclose(safe.topk_weights, record.topk_weights)


def test_storage_roundtrip_then_decide_replay_r2(tmp_path) -> None:
    with RouteTapeWriter(tmp_path, tape_id="tape", model_id="model") as writer:
        writer.write_record(fake_record(record_id="rec"))
    loaded = RouteTapeReader(tmp_path).read_record("rec")
    spec = FakeBackendSpec()
    result = decide_replay_request(loaded, fake_context(), spec.policy(), spec.capabilities())
    assert result.decision == ReplayDecision.REPLAY_R2
    assert result.level == ReplayLevel.R2


def test_storage_roundtrip_identity_mismatch_fails_closed(tmp_path) -> None:
    with RouteTapeWriter(tmp_path, tape_id="tape", model_id="model") as writer:
        writer.write_record(fake_record(record_id="rec"))
    loaded = RouteTapeReader(tmp_path).read_record("rec")
    spec = FakeBackendSpec()
    ctx = fake_context(token_positions=np.array([10, 11, 12, 13], dtype=np.int64))
    result = decide_replay_request(loaded, ctx, spec.policy(), spec.capabilities())
    assert result.decision == ReplayDecision.FAIL_CLOSED
    assert result.reason_code == "TOKEN_POSITIONS_MISMATCH"


def test_storage_manifest_participates_in_replay_decision(tmp_path) -> None:
    with RouteTapeWriter(
        tmp_path,
        tape_id="tape",
        model_id="model",
        backend_id="fake",
        metadata={"model_arch": "moe-test", "model_config_digest": "digest-a"},
    ) as writer:
        writer.write_record(fake_record(record_id="rec"))
    reader = RouteTapeReader(tmp_path)
    loaded = reader.read_record("rec")
    spec = FakeBackendSpec()
    ctx = fake_context(model_arch="moe-test", model_config_digest="digest-a")
    result = decide_replay_request(
        loaded,
        ctx,
        spec.policy(),
        spec.capabilities(),
        TapeCompatibility.from_manifest(reader.manifest),
    )
    assert result.decision == ReplayDecision.REPLAY_R2


def test_storage_manifest_backend_mismatch_fails_closed(tmp_path) -> None:
    with RouteTapeWriter(tmp_path, tape_id="tape", model_id="model", backend_id="other") as writer:
        writer.write_record(fake_record(record_id="rec"))
    reader = RouteTapeReader(tmp_path)
    loaded = reader.read_record("rec")
    spec = FakeBackendSpec(backend_id="fake")
    result = decide_replay_request(
        loaded,
        fake_context(),
        spec.policy(),
        spec.capabilities(),
        TapeCompatibility.from_manifest(reader.manifest),
    )
    assert result.decision == ReplayDecision.FAIL_CLOSED
    assert result.reason_code == "TAPE_BACKEND_MISMATCH"
