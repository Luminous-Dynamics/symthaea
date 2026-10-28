#!/usr/bin/env python3
"""Fail-closed tests for source-anchored legacy claim evidence."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools/vleb/legacy-claim-evidence.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def valid_text_claim() -> dict[str, Any]:
    return {
        "schema_version": "vleb-legacy-claim-v1",
        "task_id": "vleb-cobol-index-pilot",
        "claim_id": "COPY.BANK.REC",
        "claim_type": "copybook_reference",
        "claim_text": "The source references copybook BANK-REC.",
        "claim_scope": "textual_anchor",
        "source": {
            "path": "tools/vleb/cobol-index/fixtures/settlement.cbl",
            "sha256": "a" * 64,
            "spans": [{
                "line_start": 12,
                "line_end": 12,
                "column_start": 8,
                "column_end": 24,
                "excerpt_sha256": "b" * 64,
            }],
        },
        "verdict": "supported",
        "witnesses": [{
            "witness_id": "indexer-output-1",
            "kind": "source_index",
            "artifact_path": "evidence/cobol-index.json",
            "artifact_sha256": "c" * 64,
            "execution_receipt_path": "evidence/cobol-index-receipt.json",
            "execution_receipt_sha256": "e" * 64,
            "invocation_sha256": "f" * 64,
            "verifier_id": "vleb-cobol-index",
            "verifier_sha256": "d" * 64,
            "observed_result": "supports",
        }],
        "unresolved_reason": None,
    }


def semantic_errors(claim: dict[str, Any]) -> list[str]:
    """Checks requiring more than JSON shape; verifier hashes need external pins."""
    errors: list[str] = []
    scope = claim.get("claim_scope")
    verdict = claim.get("verdict")
    witnesses = claim.get("witnesses", [])
    witness_results = [witness.get("observed_result") for witness in witnesses]
    execution_results = [
        witness.get("observed_result")
        for witness in witnesses
        if witness.get("kind") == "independent_execution"
    ]
    if verdict == "supported" and "supports" not in witness_results:
        errors.append("supported_verdict_lacks_supporting_witness")
    if verdict == "contradicted" and "contradicts" not in witness_results:
        errors.append("contradicted_verdict_lacks_contradicting_witness")
    if verdict == "supported" and scope == "runtime_behavior" and "supports" not in execution_results:
        errors.append("runtime_claim_requires_independent_execution_support")
    if verdict == "contradicted" and scope == "runtime_behavior" and "contradicts" not in execution_results:
        errors.append("runtime_claim_requires_independent_execution_contradiction")
    if verdict in ("supported", "contradicted") and not witnesses:
        errors.append("verified_claim_requires_witness")
    for witness in witnesses:
        if not witness.get("verifier_id") or not witness.get("verifier_sha256"):
            errors.append("witness_verifier_identity_missing")
            break
    return errors


class LegacyClaimEvidenceTests(unittest.TestCase):
    def errors(self, claim: dict[str, Any]) -> list[str]:
        schema_errors = list(VALIDATOR.iter_errors(claim))
        semantic = semantic_errors(claim)
        return [
            f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
            for error in schema_errors
        ] + semantic

    def assert_rejected(self, claim: dict[str, Any]) -> None:
        errors = self.errors(claim)
        self.assertTrue(errors, "invalid claim unexpectedly accepted")

    def test_supported_textual_anchor_requires_pinned_witness_shape(self) -> None:
        claim = valid_text_claim()
        self.assertEqual(self.errors(claim), [])

    def test_runtime_behavior_cannot_be_supported_by_static_index_alone(self) -> None:
        claim = valid_text_claim()
        claim.update({
            "claim_id": "TX.COMMIT.ORDER",
            "claim_type": "transaction_effect",
            "claim_text": "Database updates commit only after the balance check.",
            "claim_scope": "runtime_behavior",
        })
        self.assertIn("runtime_claim_requires_independent_execution_support", self.errors(claim))

    def test_runtime_behavior_passes_only_with_execution_witness(self) -> None:
        claim = valid_text_claim()
        claim.update({
            "claim_id": "NUM.PACKED.SIGN",
            "claim_type": "numeric_behavior",
            "claim_text": "Negative packed-decimal values retain their sign on round-trip.",
            "claim_scope": "runtime_behavior",
        })
        claim["witnesses"].append({
            "witness_id": "runtime-test-1",
            "kind": "independent_execution",
            "artifact_path": "evidence/roundtrip-test.log",
            "artifact_sha256": "e" * 64,
            "execution_receipt_path": "evidence/roundtrip-receipt.json",
            "execution_receipt_sha256": "a" * 64,
            "invocation_sha256": "b" * 64,
            "verifier_id": "pinned-cobol-test-runner",
            "verifier_sha256": "f" * 64,
            "observed_result": "supports",
        })
        self.assertEqual(self.errors(claim), [])

    def test_contradicted_runtime_claim_needs_failing_execution_witness(self) -> None:
        claim = valid_text_claim()
        claim.update({
            "claim_id": "TX.REPLAY.IDEMPOTENT",
            "claim_type": "transaction_effect",
            "claim_text": "Replaying the same batch twice is idempotent.",
            "claim_scope": "runtime_behavior",
            "verdict": "contradicted",
        })
        claim["witnesses"] = [{
            "witness_id": "replay-witness",
            "kind": "independent_execution",
            "artifact_path": "evidence/replay-test.log",
            "artifact_sha256": "e" * 64,
            "execution_receipt_path": "evidence/replay-receipt.json",
            "execution_receipt_sha256": "a" * 64,
            "invocation_sha256": "b" * 64,
            "verifier_id": "pinned-transaction-model",
            "verifier_sha256": "f" * 64,
            "observed_result": "contradicts",
        }]
        self.assertEqual(self.errors(claim), [])

    def test_supported_claim_without_witness_is_rejected(self) -> None:
        claim = valid_text_claim()
        claim["witnesses"] = []
        self.assert_rejected(claim)

    def test_unresolved_claim_requires_reason(self) -> None:
        claim = valid_text_claim()
        claim.update({"verdict": "unresolved", "witnesses": [], "unresolved_reason": None})
        self.assert_rejected(claim)

    def test_textual_anchor_cannot_claim_numeric_or_transaction_semantics(self) -> None:
        claim = valid_text_claim()
        claim["claim_scope"] = "textual_anchor"
        claim["claim_type"] = "numeric_behavior"
        self.assert_rejected(claim)

    def test_source_traversal_and_column_73_are_rejected(self) -> None:
        traversal = valid_text_claim()
        traversal["source"]["path"] = "../private.cbl"
        self.assert_rejected(traversal)
        column = valid_text_claim()
        column["source"]["spans"][0]["column_end"] = 73
        self.assert_rejected(column)

    def test_mutating_verdict_or_witness_cannot_silently_pass(self) -> None:
        claim = valid_text_claim()
        claim["verdict"] = "contradicted"
        self.assert_rejected(claim)
        claim = valid_text_claim()
        claim["witnesses"][0]["verifier_sha256"] = "NOT-A-DIGEST"
        self.assert_rejected(claim)


if __name__ == "__main__":
    unittest.main(verbosity=2)
