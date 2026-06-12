"""RouteTape CLI inspect/validate tests."""

from __future__ import annotations

import json

import numpy as np

from routeforge.cli.main import main
from routeforge.core import ReplayLevel, RouteRecord
from routeforge.storage import RouteTapeWriter


def _record(record_id: str = "rec_0", replay_level: ReplayLevel = ReplayLevel.R2) -> RouteRecord:
    idx = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int16)
    weights = None if replay_level == ReplayLevel.R1 else np.full((4, 2), 0.5, dtype=np.float32)
    return RouteRecord(
        record_id=record_id,
        topk_idx=idx,
        topk_weights=weights,
        layer_id=0,
        num_experts=8,
        top_k=2,
        token_count=4,
        replay_level=replay_level,
    )


def _write_tape(path, *, backend_id: str | None = "fake", replay_level: ReplayLevel = ReplayLevel.R2, metadata=None) -> None:
    with RouteTapeWriter(path, tape_id="tape", model_id="model", backend_id=backend_id, metadata=metadata or {}) as writer:
        record = _record("rec_0", replay_level)
        if metadata:
            record = RouteRecord(**{**record.__dict__, "metadata": metadata})
        writer.write_record(record)


def test_tape_inspect_json_valid_tape(tmp_path, capsys) -> None:
    _write_tape(tmp_path)
    code = main(["tape", "inspect", str(tmp_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["manifest"]["tape_id"] == "tape"
    assert payload["manifest"]["model_id"] == "model"
    assert payload["manifest"]["recorded_replay_levels"] == [ReplayLevel.R2.value]
    assert payload["records"][0]["record_id"] == "rec_0"
    assert "topk_idx" in payload["records"][0]["tensors"]
    assert "topk_weights" in payload["records"][0]["tensors"]
    assert "[[" not in json.dumps(payload)
    assert payload["metadata_redacted"] is True


def test_tape_inspect_human_valid_tape(tmp_path, capsys) -> None:
    _write_tape(tmp_path)
    code = main(["tape", "inspect", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "TAPE" in out
    assert "tape_id" in out
    assert "rec_0" in out


def test_tape_validate_json_valid_tape(tmp_path, capsys) -> None:
    _write_tape(tmp_path)
    code = main(["tape", "validate", str(tmp_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["summary"] == {"total": 1, "pass": 1, "fail": 0}
    assert payload["records"][0]["checksum_ok"] is True
    assert payload["records"][0]["tensor_shapes_ok"] is True
    assert payload["records"][0]["tensor_dtypes_ok"] is True
    assert payload["records"][0]["guard_decision"] == "REPLAY_R2"


def test_tape_validate_human_valid_tape(tmp_path, capsys) -> None:
    _write_tape(tmp_path)
    code = main(["tape", "validate", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "[PASS]" in out
    assert "0 failures" in out


def test_tape_inspect_missing_directory_exit_3(tmp_path, capsys) -> None:
    missing = tmp_path / "missing"
    code = main(["tape", "inspect", str(missing), "--json"])
    assert code == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"


def test_tape_inspect_missing_manifest_exit_1(tmp_path, capsys) -> None:
    tmp_path.mkdir(exist_ok=True)
    code = main(["tape", "inspect", str(tmp_path), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"


def test_tape_validate_checksum_mismatch_exit_2(tmp_path, capsys) -> None:
    _write_tape(tmp_path)
    chunk = next((tmp_path / "chunks").iterdir())
    chunk.write_bytes(b"tampered")
    code = main(["tape", "validate", str(tmp_path), "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["records"][0]["checksum_ok"] is False
    assert payload["records"][0]["guard_reason_code"] == "TAPE_INTEGRITY_FAILED"


def test_tape_inspect_redacts_metadata_by_default(tmp_path, capsys) -> None:
    _write_tape(tmp_path, metadata={"prompt": "SECRET", "safe_key": "value"})
    code = main(["tape", "inspect", str(tmp_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    text = json.dumps(payload)
    assert "SECRET" not in text
    assert "prompt" not in text
    assert "safe_key" not in text
    assert payload["manifest"]["metadata"] == {"redacted": True, "key_count": 2}
    assert payload["records"][0]["metadata"] == {"redacted": True, "key_count": 2}


def test_tape_inspect_show_metadata_is_explicit(tmp_path, capsys) -> None:
    _write_tape(tmp_path, metadata={"prompt": "SECRET"})
    code = main(["tape", "inspect", str(tmp_path), "--json", "--show-metadata"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata_redacted"] is False
    assert payload["manifest"]["metadata"] == {"prompt": "SECRET"}
    assert payload["records"][0]["metadata"] == {"prompt": "SECRET"}


def test_tape_validate_marks_token_identity_check_skipped(tmp_path, capsys) -> None:
    _write_tape(tmp_path)
    code = main(["tape", "validate", str(tmp_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["token_identity_check"] == "skipped_cli_integrity_mode"


def test_tape_validate_human_marks_token_identity_skipped(tmp_path, capsys) -> None:
    _write_tape(tmp_path)
    code = main(["tape", "validate", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "token_identity_check: skipped_cli_integrity_mode" in out


def test_tape_inspect_single_record(tmp_path, capsys) -> None:
    with RouteTapeWriter(tmp_path, tape_id="tape", model_id="model") as writer:
        writer.write_record(_record("a"))
        writer.write_record(_record("b", ReplayLevel.R1))
    code = main(["tape", "inspect", str(tmp_path), "--record", "b", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [record["record_id"] for record in payload["records"]] == ["b"]
    assert "topk_weights" not in payload["records"][0]["tensors"]
