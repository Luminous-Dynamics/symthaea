#!/usr/bin/env python3
"""Generate deterministic evaluator execution-custody plan and receipt fixtures.

The receipt and transparency checkpoint are synthetic. A locally valid inclusion
proof is not an authenticated external log checkpoint or runner attestation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "symthaea.science_bench_refresh_execution_custody_plan.v1"
RECEIPT_SCHEMA = "symthaea.science_bench_refresh_execution_custody_receipt.v1"
PLAN_DOMAIN = b"symthaea.execution_custody.plan.v1\0"
EVENT_DOMAIN = b"symthaea.execution_custody.event.v1\0"
BUNDLE_DOMAIN = b"symthaea.execution_custody.bundle.v1\0"
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "research/science-bench-refresh/Cargo.toml"
ZERO_DIGEST = "0" * 64


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def domain_digest(domain: bytes, payload: bytes) -> str:
    return digest(domain + len(payload).to_bytes(8, "big") + payload)


def merkle_leaf(event_or_bundle_digest: str) -> str:
    return digest(b"\x00" + bytes.fromhex(event_or_bundle_digest))


def merkle_root(leaf_digests: list[str]) -> str:
    level = [merkle_leaf(item) for item in leaf_digests]
    if not level:
        raise ValueError("cannot build empty Merkle tree")
    while len(level) > 1:
        next_level = []
        for offset in range(0, len(level), 2):
            left = bytes.fromhex(level[offset])
            right = bytes.fromhex(level[offset + 1] if offset + 1 < len(level) else level[offset])
            next_level.append(digest(b"\x01" + left + right))
        level = next_level
    return level[0]


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer_revision = git_revision()
    run_id = "synthetic-custody-run-001"
    condition = {
        "condition_id": "condition-001",
        "planned_ordinal": 0,
        "input_sha256": digest(b"exact synthetic rendered evaluator input"),
        "expected_input_bytes": len(b"exact synthetic rendered evaluator input"),
    }
    plan_fields = {
        "schema": PLAN_SCHEMA,
        "analyzer_source_revision": analyzer_revision,
        "purpose": "example_only",
        "run_id": run_id,
        "run_nonce_sha256": digest(b"synthetic one-run nonce"),
        "evaluator_source_revision": "b" * 40,
        "evaluator_prompt_sha256": digest(b"synthetic evaluator prompt"),
        "rubric_sha256": digest(b"synthetic fixed rubric"),
        "task_set_sha256": digest(b"synthetic one-task set"),
        "environment_sha256": digest(b"synthetic container environment"),
        "max_parallelism": 1,
        "conditions": [condition],
    }
    plan_fields_path = output_dir / "execution-custody-plan-fields.example.json"
    plan_fields_path.write_bytes(pretty_json(plan_fields))
    command = [
        "cargo", "run", "--quiet", "--locked", "--manifest-path", str(MANIFEST),
        "--", "commit-execution-custody-plan", str(plan_fields_path),
    ]
    committed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    commitment = json.loads(committed.stdout)["plan_commitment_sha256"]
    plan = {**plan_fields, "plan_commitment_sha256": commitment}
    plan_bytes = pretty_json(plan)
    plan_path = output_dir / "execution-custody-plan.example.json"
    plan_path.write_bytes(plan_bytes)
    plan_sha = digest(plan_bytes)

    payloads = [
        {
            "event_type": "run_started",
            "run_id": run_id,
            "plan_sha256": plan_sha,
            "runner_id_sha256": digest(b"synthetic pseudonymous runner"),
            "environment_sha256": plan_fields["environment_sha256"],
        },
        {
            "event_type": "input_rendered",
            "condition_id": condition["condition_id"],
            "planned_ordinal": 0,
            "input_sha256": condition["input_sha256"],
            "input_bytes": condition["expected_input_bytes"],
        },
        {
            "event_type": "evaluation_started",
            "condition_id": condition["condition_id"],
            "planned_ordinal": 0,
            "evaluator_source_revision": plan_fields["evaluator_source_revision"],
            "evaluator_prompt_sha256": plan_fields["evaluator_prompt_sha256"],
        },
        {
            "event_type": "tool_call",
            "condition_id": condition["condition_id"],
            "tool_name_sha256": digest(b"synthetic retrieval tool"),
            "request_sha256": digest(b"synthetic tool request"),
            "request_bytes": len(b"synthetic tool request"),
            "response_sha256": digest(b"synthetic tool response"),
            "response_bytes": len(b"synthetic tool response"),
        },
        {
            "event_type": "evaluation_finished",
            "condition_id": condition["condition_id"],
            "planned_ordinal": 0,
            "status": "completed",
            "output_sha256": digest(b"synthetic evaluator output"),
            "output_bytes": len(b"synthetic evaluator output"),
            "evaluator_call_count": 1,
            "evaluator_token_count": 123,
        },
        {
            "event_type": "run_finished",
            "run_id": run_id,
            "planned_condition_count": 1,
            "completed_condition_count": 1,
            "terminal_condition_count": 1,
            "run_status": "completed",
        },
    ]
    events = []
    previous = ZERO_DIGEST
    base_time = 1_800_000_000_000
    for sequence, payload in enumerate(payloads):
        fields = {
            "sequence": sequence,
            "timestamp_unix_milliseconds": base_time + sequence,
            "previous_event_sha256": previous,
            "payload": payload,
        }
        event_hash = domain_digest(EVENT_DOMAIN, compact_json(fields))
        events.append({**fields, "event_sha256": event_hash})
        previous = event_hash

    event_root = merkle_root([event["event_sha256"] for event in events])
    bundle_identity = [plan_sha, commitment, event_root, len(events), run_id]
    bundle_sha = domain_digest(BUNDLE_DOMAIN, compact_json(bundle_identity))
    external_checkpoint = {
        "provider": "synthetic-local-transparency-log",
        "log_id": "synthetic-log-001",
        "record_id": "synthetic-record-001",
        "bundle_sha256": bundle_sha,
        "checkpoint_root_sha256": merkle_root([bundle_sha]),
        "tree_size": 1,
        "anchored_at_unix_milliseconds": base_time + 100,
        "inclusion_proof": {
            "leaf_index": 0,
            "tree_size": 1,
            "sibling_sha256": [],
        },
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "analyzer_source_revision": analyzer_revision,
        "purpose": "example_only",
        "run_id": run_id,
        "plan_sha256": plan_sha,
        "plan_commitment_sha256": commitment,
        "evaluator_source_revision": plan_fields["evaluator_source_revision"],
        "evaluator_prompt_sha256": plan_fields["evaluator_prompt_sha256"],
        "rubric_sha256": plan_fields["rubric_sha256"],
        "environment_sha256": plan_fields["environment_sha256"],
        "events": events,
        "external_checkpoint": external_checkpoint,
    }
    receipt_path = output_dir / "execution-custody-receipt.example.json"
    receipt_path.write_bytes(pretty_json(receipt))
    return {
        "schema": "symthaea.science_bench_refresh_execution_custody_fixture_summary.v1",
        "synthetic_positive_control": True,
        "planned_conditions": 1,
        "events": len(events),
        "event_merkle_root_sha256": event_root,
        "execution_bundle_sha256": bundle_sha,
        "external_checkpoint_inclusion_proof_locally_valid": True,
        "external_checkpoint_authenticity_proven": False,
        "runner_identity_authenticated": False,
        "raw_prompt_custody_proven": False,
        "evaluator_execution_proven": False,
        "wall_clock_authenticity_proven": False,
        "causal_effect_identified": False,
        "scoring_eligible": False,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: execution_custody_fixture.py OUTPUT_DIR", file=sys.stderr)
        return 2
    print(json.dumps(generate(Path(sys.argv[1])), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
