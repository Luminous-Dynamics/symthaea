"""Exercise supersession through the compiled Rust JSONL worker.

This is an explicit write-side lifecycle probe, distinct from heuristic
currentness inference and from question-answer accuracy measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures"
MAX_LINE_BYTES = 32 * 1024 * 1024
PROTOCOL = "broca-explicit-supersession-probe-v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{line_number}: exceeds line size bound")
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def request(request_id: str, op: str, **fields: Any) -> dict[str, Any]:
    return {"request_id": request_id, "op": op, **fields}


def invoke_worker(worker: Path, requests: list[dict[str, Any]], timeout_seconds: float) -> tuple[list[dict[str, Any]], bytes, bytes]:
    request_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in requests
    ).encode("utf-8")
    if any(len(line) > MAX_LINE_BYTES for line in request_bytes.splitlines()):
        raise ValueError("a generated request exceeds the worker line bound")
    completed = subprocess.run(
        [str(worker)],
        input=request_bytes,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker exited {completed.returncode}; stderr: {completed.stderr.decode('utf-8', 'replace')[-8000:]}"
        )
    if completed.stderr:
        raise RuntimeError(f"worker wrote unexpected stderr: {completed.stderr.decode('utf-8', 'replace')[-8000:]}")
    response_lines = completed.stdout.splitlines()
    if len(response_lines) != len(requests):
        raise ValueError(f"expected {len(requests)} responses, received {len(response_lines)}")
    responses: list[dict[str, Any]] = []
    for index, (line, expected) in enumerate(zip(response_lines, requests), 1):
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(f"response {index} exceeds line size bound")
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"response {index} is not an object")
        if parsed.get("request_id") != expected["request_id"]:
            raise ValueError(f"response {index} request_id mismatch")
        if parsed.get("ok") is not True:
            raise RuntimeError(f"worker rejected {expected['request_id']}: {parsed.get('error')}")
        if not isinstance(parsed.get("result"), dict):
            raise ValueError(f"response {index} missing result object")
        responses.append(parsed["result"])
    return responses, request_bytes, completed.stdout


def _validate_context_audit(result: dict[str, Any], label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit = result.get("audit")
    context = result.get("context")
    if not isinstance(audit, list) or not isinstance(context, list):
        raise ValueError(f"{label}: query result must contain audit and context arrays")
    if len(context) != len(audit):
        raise ValueError(f"{label}: reader context and audit cardinalities disagree")
    typed_audit: list[dict[str, Any]] = []
    for index, (raw_item, raw_context) in enumerate(zip(audit, context), 1):
        if not isinstance(raw_item, dict) or not isinstance(raw_context, dict):
            raise ValueError(f"{label}: item {index} must be an audit/context object")
        evidence_id = raw_item.get("evidence_id")
        content_digest = raw_item.get("content_sha256")
        value = raw_context.get("value")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError(f"{label}: item {index} lacks a stable evidence ID")
        if raw_context.get("type") != "text" or not isinstance(value, str):
            raise ValueError(f"{label}: item {index} is not a text context item")
        if not isinstance(content_digest, str) or len(content_digest) != 64:
            raise ValueError(f"{label}: item {index} lacks a content digest")
        if sha256_bytes(value.encode("utf-8")) != content_digest:
            raise ValueError(f"{label}: item {index} context value does not match its audit digest")
        typed_audit.append(raw_item)
    return typed_audit, context


def assert_transition_result(
    old_evidence_id: str,
    new_evidence_id: str,
    supersede_result: dict[str, Any],
    old_query_result: dict[str, Any],
    new_query_result: dict[str, Any],
) -> None:
    if old_evidence_id == new_evidence_id:
        raise ValueError("old and replacement evidence IDs must differ")
    if supersede_result.get("lifecycle") != "superseded":
        raise ValueError("worker did not confirm superseded lifecycle")
    if supersede_result.get("old_evidence_id") != old_evidence_id:
        raise ValueError("worker superseded the wrong old evidence ID")
    if supersede_result.get("new_evidence_id") != new_evidence_id:
        raise ValueError("worker superseded to the wrong replacement ID")
    old_audit, _old_context = _validate_context_audit(old_query_result, "old-value query")
    new_audit, _new_context = _validate_context_audit(new_query_result, "replacement query")
    if any(item["evidence_id"] == old_evidence_id for item in old_audit):
        raise ValueError("retired evidence remained retrievable after supersession")
    if not any(item["evidence_id"] == new_evidence_id for item in new_audit):
        raise ValueError("replacement evidence is not retrievable after supersession")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    worker = args.worker.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not worker.is_file() or not worker.stat().st_mode & 0o111:
        raise SystemExit(f"worker must exist and be executable: {worker}")
    if not 0.1 <= args.timeout_seconds <= 3600:
        raise SystemExit("--timeout-seconds must be in [0.1, 3600]")

    trajectory_path = FIXTURE_DIR / "trajectories.jsonl"
    trajectories = load_jsonl(trajectory_path)
    matches = [row for row in trajectories if row.get("id") == "episode-office-address"]
    if len(matches) != 1:
        raise SystemExit("expected exactly one episode-office-address fixture")
    trajectory = matches[0]

    # Phase 1 retrieves the deterministic IDs for the exact old and current spans.
    discovery_requests = [
        request("discover:init", "init", scope_id="web", bm25={"k1": 1.2, "b": 0.75, "max_items": 20}),
        request("discover:insert", "insert", trajectory=trajectory),
        request("discover:query", "query", query="support office address 10 Old Mill Road"),
    ]
    discovery, discovery_input, discovery_output = invoke_worker(worker, discovery_requests, args.timeout_seconds)
    discovery_insert = discovery[1]
    if discovery_insert.get("trajectory_id") != trajectory["id"]:
        raise ValueError("discovery insert returned the wrong trajectory ID")
    source_sha256 = discovery_insert.get("source_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("worker insert did not return a lowercase source SHA-256")
    discovery_audit = discovery[2].get("audit")
    if not isinstance(discovery_audit, list):
        raise ValueError("discovery query has no audit array")

    def evidence_id_at(pointer: str) -> str:
        matches_at_pointer = [
            item.get("evidence_id") for item in discovery_audit
            if isinstance(item, dict) and item.get("trajectory_id") == trajectory["id"]
            and item.get("source_pointer") == pointer
        ]
        if len(matches_at_pointer) != 1 or not isinstance(matches_at_pointer[0], str):
            raise ValueError(f"expected exactly one retrieved evidence ID at {pointer}")
        return matches_at_pointer[0]

    old_id = evidence_id_at("/states/0/text")
    new_id = evidence_id_at("/states/1/text")
    if old_id == new_id:
        raise ValueError("old and replacement evidence IDs must differ")

    # Phase 2 replays the identical insertion, then exercises the public operation.
    transition_requests = [
        request("transition:init", "init", scope_id="web", bm25={"k1": 1.2, "b": 0.75, "max_items": 20}),
        request("transition:insert", "insert", trajectory=trajectory),
        request("transition:supersede", "supersede", old_evidence_id=old_id, new_evidence_id=new_id),
        request("transition:query-old", "query", query="support office address 10 Old Mill Road"),
        request("transition:query-new", "query", query="current support office address 42 New Harbour Road"),
    ]
    transitioned, transition_input, transition_output = invoke_worker(worker, transition_requests, args.timeout_seconds)
    transition_insert = transitioned[1]
    if transition_insert.get("trajectory_id") != trajectory["id"]:
        raise ValueError("transition insert returned the wrong trajectory ID")
    if transition_insert.get("source_sha256") != source_sha256:
        raise ValueError("replayed trajectory source digest differs across worker runs")
    assert_transition_result(old_id, new_id, transitioned[2], transitioned[3], transitioned[4])

    output_record = {
        "schema": PROTOCOL,
        "trajectory_id": trajectory["id"],
        "source_sha256": source_sha256,
        "old_source_pointer": "/states/0/text",
        "replacement_source_pointer": "/states/1/text",
        "old_evidence_id": old_id,
        "new_evidence_id": new_id,
        "discovery_input_sha256": sha256_bytes(discovery_input),
        "discovery_output_sha256": sha256_bytes(discovery_output),
        "transition_input_sha256": sha256_bytes(transition_input),
        "transition_output_sha256": sha256_bytes(transition_output),
        "worker_sha256": sha256_bytes(worker.read_bytes()),
        "fixture_sha256": sha256_bytes(trajectory_path.read_bytes()),
        "scope_id": "web",
        "bm25": {"k1": 1.2, "b": 0.75, "max_items": 20},
        "assertions": {
            "supersede_acknowledged": True,
            "retired_evidence_not_retrievable": True,
            "replacement_evidence_retrievable": True,
            "context_audit_cardinality_matches": True,
        },
        "claim_ceiling": "explicit worker lifecycle operation probe; no automatic update detection, QA accuracy, or causal truth claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "old_evidence_id": old_id, "new_evidence_id": new_id, "verified": True}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
