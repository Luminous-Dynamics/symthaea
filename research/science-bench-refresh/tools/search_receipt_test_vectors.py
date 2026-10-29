#!/usr/bin/env python3
"""Test-only deterministic search-receipt vectors for the Rust CLI.

This intentionally mirrors only the public v1 serialization/hash contract. It is
not production receipt collection and cannot attest that network events were all
recorded by a real runner.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PREFIX = b"symthaea.science_bench_refresh.search_receipt.v1\0"
ZERO = "0" * 64
RECEIPT_SCHEMA = "symthaea.science_bench_refresh_search_exposure_receipt.v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def framed_hash(domain: bytes, payload: bytes) -> str:
    return sha256(
        PREFIX
        + len(domain).to_bytes(8, "big")
        + domain
        + len(payload).to_bytes(8, "big")
        + payload
    )


def reseal(
    policy: str,
    allowed_hosts: list[str],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    previous = ZERO
    for index, action in enumerate(actions):
        event = {
            "sequence": index,
            "event_kind": action["event_kind"],
            "decision": action["decision"],
            "request_sha256": sha256(action["request"].encode()),
            "host": action["host"],
            "response_sha256": (
                sha256(action["response"].encode())
                if action.get("response") is not None
                else None
            ),
            "previous_event_sha256": previous,
            "event_sha256": ZERO,
        }
        payload = {key: value for key, value in event.items() if key != "event_sha256"}
        event["event_sha256"] = framed_hash(b"event.v1", encode(payload))
        previous = event["event_sha256"]
        events.append(event)

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "run_id": "test-run-0001",
        "generation_id": "test-generation-0001",
        "policy_revision_sha256": sha256(b"frozen-policy-revision"),
        "task_ref_sha256": sha256(b"opaque-task-reference"),
        "network_policy": policy,
        "allowed_hosts": allowed_hosts,
        "expected_event_count": len(events),
        "final_event_sha256": previous,
        "events": events,
        "receipt_commitment_sha256": ZERO,
    }
    context = {
        "schema": receipt["schema"],
        "run_id": receipt["run_id"],
        "generation_id": receipt["generation_id"],
        "policy_revision_sha256": receipt["policy_revision_sha256"],
        "task_ref_sha256": receipt["task_ref_sha256"],
        "network_policy": receipt["network_policy"],
        "allowed_hosts": receipt["allowed_hosts"],
        "expected_event_count": receipt["expected_event_count"],
        "final_event_sha256": receipt["final_event_sha256"],
    }
    receipt["receipt_commitment_sha256"] = framed_hash(b"receipt.v1", encode(context))
    return receipt


def write(path: Path, receipt: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: search_receipt_test_vectors.py OUTPUT_DIRECTORY", file=sys.stderr)
        return 2
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)

    blocked = [
        {"event_kind": "network_request", "decision": "block",
         "request": "query: example", "host": "search.example.org"},
        {"event_kind": "network_request", "decision": "block",
         "request": "fetch: example", "host": "docs.example.org"},
    ]
    valid = reseal("isolated", [], blocked)
    write(output / "valid-isolated.json", valid)

    allowed = [{
        "event_kind": "fetch", "decision": "allow",
        "request": "fetch: example", "host": "docs.example.org",
        "response": "observed-response",
    }]
    write(output / "isolated-policy-violation.json", reseal("isolated", [], allowed))

    changed = json.loads(json.dumps(valid))
    changed["events"][0]["request_sha256"] = sha256(b"mutated-request")
    write(output / "mutated-event.json", changed)

    removed = json.loads(json.dumps(valid))
    removed["events"].pop()
    write(output / "missing-event.json", removed)

    reordered = json.loads(json.dumps(valid))
    reordered["events"].reverse()
    write(output / "reordered-events.json", reordered)

    allowed_list = [{
        "event_kind": "fetch", "decision": "allow",
        "request": "fetch: outsider", "host": "outsider.example.net",
        "response": "observed-response",
    }]
    write(
        output / "allowlist-policy-violation.json",
        reseal("controlled_allowlist", ["docs.example.org"], allowed_list),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
