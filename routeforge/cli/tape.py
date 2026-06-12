"""Operator-facing RouteTape inspect/validate CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from routeforge.core import (
    BackendCapabilities,
    ReplayDecision,
    ReplayLevel,
    ReplayPolicy,
    RuntimeContext,
    TapeCompatibility,
    decide_replay_request,
)
from routeforge.storage import RouteTapeError, RouteTapeReader

EXIT_OK = 0
EXIT_CORRUPT = 1
EXIT_GUARD_FAIL = 2
EXIT_USAGE = 3


def add_tape_subparser(subparsers: argparse._SubParsersAction) -> None:
    tape = subparsers.add_parser("tape", help="Inspect or validate RouteTape containers")
    tape_sub = tape.add_subparsers(dest="tape_command")

    inspect = tape_sub.add_parser("inspect", help="Summarize a RouteTape")
    inspect.add_argument("tape_dir")
    inspect.add_argument("--json", action="store_true", dest="json_output")
    inspect.add_argument("--record", dest="record_id")
    inspect.add_argument("--show-metadata", action="store_true", dest="show_metadata")
    inspect.set_defaults(func=cmd_inspect)

    validate = tape_sub.add_parser("validate", help="Validate RouteTape integrity")
    validate.add_argument("tape_dir")
    validate.add_argument("--json", action="store_true", dest="json_output")
    validate.add_argument("--record", dest="record_id")
    validate.add_argument("--show-metadata", action="store_true", dest="show_metadata")
    validate.set_defaults(func=cmd_validate)


def cmd_inspect(args: argparse.Namespace) -> int:
    tape_dir = Path(args.tape_dir)
    if not tape_dir.exists() or not tape_dir.is_dir():
        _emit_error(args, f"not a directory: {tape_dir}")
        return EXIT_USAGE
    try:
        reader = RouteTapeReader(tape_dir)
        output = _inspect_output(tape_dir, reader, args.record_id, args.show_metadata)
    except RouteTapeError as exc:
        _emit_error(args, str(exc))
        return EXIT_CORRUPT
    _emit(args, output, human=_human_inspect(output))
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    tape_dir = Path(args.tape_dir)
    if not tape_dir.exists() or not tape_dir.is_dir():
        _emit_error(args, f"not a directory: {tape_dir}")
        return EXIT_USAGE
    try:
        reader = RouteTapeReader(tape_dir)
        output = _validate_output(tape_dir, reader, args.record_id, args.show_metadata)
    except RouteTapeError as exc:
        _emit_error(args, str(exc))
        return EXIT_CORRUPT
    _emit(args, output, human=_human_validate(output))
    return EXIT_OK if output["status"] == "ok" else EXIT_GUARD_FAIL


def _inspect_output(
    tape_dir: Path,
    reader: RouteTapeReader,
    record_id: str | None,
    include_metadata: bool = False,
) -> dict[str, Any]:
    records = [reader.record_summary(record_id, include_metadata=include_metadata)] if record_id else reader.list_records(include_metadata=include_metadata)
    return {
        "status": "ok",
        "tape_dir": str(tape_dir),
        "manifest": reader.manifest_dict(include_metadata=include_metadata),
        "records": records,
        "metadata_redacted": not include_metadata,
    }


def _validate_output(
    tape_dir: Path,
    reader: RouteTapeReader,
    record_id: str | None,
    include_metadata: bool = False,
) -> dict[str, Any]:
    record_ids = [record_id] if record_id else reader.record_ids()
    caps = _default_capabilities(reader)
    policy = ReplayPolicy(require_token_identity=False)
    tape = TapeCompatibility.from_manifest(reader.manifest)
    records: list[dict[str, Any]] = []
    failures = 0

    for rid in record_ids:
        try:
            integrity = reader.validate_record_integrity(rid)
        except RouteTapeError as exc:
            records.append({"record_id": rid, "status": "fail", "message": str(exc)})
            failures += 1
            continue
        item = dict(integrity)
        item.update(reader.record_summary(rid, include_metadata=include_metadata))
        if integrity["status"] == "pass":
            try:
                record = reader.read_record(rid)
                decision = decide_replay_request(
                    record,
                    _context_from_record(record, reader),
                    policy,
                    caps,
                    tape,
                )
                item["guard_decision"] = decision.decision.name
                item["guard_reason_code"] = decision.reason_code
                item["guard_message"] = decision.message
                if decision.decision == ReplayDecision.FAIL_CLOSED:
                    item["status"] = "fail"
            except Exception as exc:  # pragma: no cover - defensive CLI boundary
                item.update(status="fail", guard_decision=None, guard_reason_code="READ_RECORD_FAILED", guard_message=str(exc))
        else:
            item["guard_decision"] = None
            item["guard_reason_code"] = "TAPE_INTEGRITY_FAILED"
            item["guard_message"] = integrity.get("message")
        if item["status"] != "pass":
            failures += 1
        records.append(item)

    return {
        "status": "ok" if failures == 0 else "fail",
        "tape_dir": str(tape_dir),
        "manifest": reader.manifest_dict(include_metadata=include_metadata),
        "metadata_redacted": not include_metadata,
        "token_identity_check": "skipped_cli_integrity_mode",
        "checks": {
            "manifest_loadable": True,
            "index_loadable": True,
            "num_records_consistent": reader.manifest.num_records == len(reader.record_ids()),
            "version_compatible": reader.manifest.version == "0.1.0",
        },
        "records": records,
        "summary": {"total": len(records), "pass": len(records) - failures, "fail": failures},
    }


def _default_capabilities(reader: RouteTapeReader) -> BackendCapabilities:
    return BackendCapabilities(
        backend_id=reader.manifest.backend_id or "routeforge_tape_cli",
        backend_version=reader.manifest.backend_version,
        supported_replay_levels=frozenset({ReplayLevel.R0, ReplayLevel.R1, ReplayLevel.R2}),
        supports_index_replay=True,
        supports_weight_replay=True,
        supported_index_dtypes=frozenset({"int16", "int32", "int64"}),
        supported_weight_dtypes=frozenset({"float16", "bfloat16", "float32", "float64"}),
    )


def _context_from_record(record: Any, reader: RouteTapeReader) -> RuntimeContext:
    return RuntimeContext(
        token_count=record.token_count,
        layer_id=record.layer_id,
        top_k=record.top_k,
        num_experts=record.num_experts,
        replay_level=record.replay_level,
        abi_version=record.abi_version,
        backend_id=reader.manifest.backend_id or "routeforge_tape_cli",
        backend_version=reader.manifest.backend_version,
        model_arch=reader.manifest.model_arch,
        model_config_digest=reader.manifest.model_config_digest,
        phase=record.phase,
        step_id=record.step_id,
        request_ids=record.request_ids,
        sequence_ids=record.sequence_ids,
        token_positions=record.token_positions,
        block_ids=record.block_ids,
        token_order=record.token_order,
        weight_semantics=record.weight_semantics,
        expert_namespace=record.expert_namespace,
    )


def _emit(args: argparse.Namespace, payload: dict[str, Any], *, human: str) -> None:
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(human)


def _emit_error(args: argparse.Namespace, message: str) -> None:
    payload = {"status": "error", "message": message, "manifest": None, "records": []}
    if getattr(args, "json_output", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ERROR  {message}", file=sys.stderr)


def _human_inspect(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        f"TAPE    {payload['tape_dir']}",
        f"  version:       {manifest.get('version')}",
        f"  tape_id:       {manifest.get('tape_id')}",
        f"  model_id:      {manifest.get('model_id')}",
        f"  backend_id:    {manifest.get('backend_id')}",
        f"  abi_version:   {manifest.get('abi_version')}",
        f"  num_records:   {manifest.get('num_records')}",
        "",
        "RECORDS",
    ]
    for rec in payload["records"]:
        lines.append(
            f"  {rec['record_id']}  layer={rec['layer_id']}  {rec['replay_level']}  "
            f"tokens={rec['token_count']}  top_k={rec['top_k']}  experts={rec['num_experts']}  "
            f"phase={rec['phase']}  chunk={rec['chunk']}"
        )
    return "\n".join(lines)


def _human_validate(payload: dict[str, Any]) -> str:
    lines = [
        f"VALIDATING  {payload['tape_dir']}",
        f"token_identity_check: {payload.get('token_identity_check', 'unknown')}",
    ]
    for rec in payload["records"]:
        marker = "PASS" if rec.get("status") == "pass" else "FAIL"
        reason = rec.get("guard_reason_code") or rec.get("message") or "ok"
        lines.append(f"  [{marker}]  {rec.get('record_id')}  {reason}")
    summary = payload["summary"]
    lines.append(f"RESULT  {summary['fail']} failures, {summary['pass']} passes")
    return "\n".join(lines)
