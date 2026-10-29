#!/usr/bin/env python3
"""Generate deterministic exact-digest matcher fixtures for CI.

The protected index is evaluator-only. Fixtures contain only synthetic content,
hashes, and fake references.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PREFIX = b"symthaea.science_bench_refresh.protected_search_index.v1\0"
SCHEMA = "symthaea.science_bench_refresh_protected_search_artifact_index.v1"


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


def make_index(
    generation: str,
    protected: list[tuple[str, str, str]],
) -> dict[str, Any]:
    artifacts = [
        {
            "reference_id_sha256": sha256(reference_id.encode()),
            "classification": classification,
            "content_sha256": sha256(content.encode()),
        }
        for reference_id, classification, content in protected
    ]
    artifacts.sort(
        key=lambda artifact: (
            artifact["content_sha256"],
            artifact["classification"],
            artifact["reference_id_sha256"],
        )
    )
    index: dict[str, Any] = {
        "schema": SCHEMA,
        "generation_id": generation,
        "matcher_revision_sha256": sha256(b"exact-protected-matcher-v1"),
        "decision_rules_sha256": sha256(b"severity-priority-answer-context-metadata-v1"),
        "artifacts": artifacts,
        "index_commitment_sha256": "0" * 64,
    }
    payload = {key: value for key, value in index.items() if key != "index_commitment_sha256"}
    index["index_commitment_sha256"] = framed_hash(b"protected-index.v1", encode(payload))
    return index


def make_receipt(
    generation: str,
    content: str | None,
    policy: str = "audited_open_web",
) -> dict[str, Any]:
    from search_receipt_test_vectors import reseal

    if content is None:
        actions = [{
            "event_kind": "network_request",
            "decision": "block",
            "request": "synthetic blocked request",
            "host": "search.example.org",
        }]
        policy = "isolated"
    else:
        actions = [{
            "event_kind": "fetch",
            "decision": "allow",
            "request": "synthetic fetch request",
            "host": "docs.example.org",
            "response": content,
        }]
    receipt = reseal(policy, [], actions)
    receipt["generation_id"] = generation
    # reseal the receipt commitment after binding the requested generation ID.
    # Use the exact v1 receipt hash profile from the receipt generator.
    receipt_payload = {
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
    receipt_prefix = b"symthaea.science_bench_refresh.search_receipt.v1\0"
    def receipt_hash(domain: bytes, raw: bytes) -> str:
        return sha256(
            receipt_prefix + len(domain).to_bytes(8, "big") + domain
            + len(raw).to_bytes(8, "big") + raw
        )
    receipt["receipt_commitment_sha256"] = receipt_hash(b"receipt.v1", encode(receipt_payload))
    return receipt


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: search_exact_match_test_vectors.py OUTPUT_DIRECTORY", file=sys.stderr)
        return 2
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    generation = "exact-matcher-generation-001"

    cases = [
        ("answer", "exact protected answer bytes", [
            ("meta-ref", "benchmark_metadata", "exact protected answer bytes"),
            ("question-ref", "question_context", "exact protected answer bytes"),
            ("answer-ref", "explicit_answer", "exact protected answer bytes"),
        ]),
        ("context", "exact protected question context", [
            ("context-ref", "question_context", "exact protected question context"),
        ]),
        ("metadata", "exact protected benchmark metadata", [
            ("metadata-ref", "benchmark_metadata", "exact protected benchmark metadata"),
        ]),
        ("no-match", "benign unrelated source content", [
            ("answer-ref", "explicit_answer", "different answer content"),
        ]),
    ]
    for name, response, protected in cases:
        receipt = make_receipt(generation, response)
        index = make_index(generation, protected)
        write_json(output / f"{name}-receipt.json", receipt)
        write_json(output / f"{name}-index.json", index)
        if name == "answer":
            fields = {
                "generation_id": index["generation_id"],
                "matcher_revision_sha256": index["matcher_revision_sha256"],
                "decision_rules_sha256": index["decision_rules_sha256"],
                "artifacts": index["artifacts"],
            }
            write_json(output / "answer-index-fields.json", fields)

    blocked = make_receipt(generation, None)
    blocked_index = make_index(generation, [])
    write_json(output / "blocked-receipt.json", blocked)
    write_json(output / "blocked-index.json", blocked_index)

    bad_index = make_index(
        generation, [("answer-ref", "explicit_answer", "some answer")]
    )
    bad_index["artifacts"][0]["content_sha256"] = sha256(b"mutated answer")
    # Leave the previous commitment in place to test fail-closed index validation.
    write_json(output / "mutated-index.json", bad_index)

    wrong_generation = make_index(
        "other-generation", [("answer-ref", "explicit_answer", "exact protected answer bytes")]
    )
    write_json(output / "wrong-generation-index.json", wrong_generation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
