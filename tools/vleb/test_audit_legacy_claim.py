#!/usr/bin/env python3
"""Adversarial tests for the trusted legacy-claim evidence auditor."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import audit_legacy_claim as audit


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class ClaimFixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="vleb-claim-audit-")
        self.root = Path(self.tmp.name)
        self.source_root = self.root / "source"
        self.evidence_root = self.root / "evidence"
        self.source_root.mkdir()
        self.evidence_root.mkdir()
        self.source_raw = b"       COPY BANK-REC.\n"
        (self.source_root / "legacy.cbl").write_bytes(self.source_raw)
        self.source_sha = sha(self.source_raw)
        snippet = self.source_raw.splitlines()[0][7:21]
        self.evidence_raw = b'{"copy_references":["BANK-REC"]}\n'
        (self.evidence_root / "index.json").write_bytes(self.evidence_raw)
        self.execution_receipt_raw = b'{"schema_version":"vleb-witness-execution-receipt-v1"}\n'
        (self.evidence_root / "index-receipt.json").write_bytes(self.execution_receipt_raw)
        self.claim = {
            "schema_version": "vleb-legacy-claim-v1",
            "task_id": "vleb-cobol-index-pilot",
            "claim_id": "COPY.BANK.REC",
            "claim_type": "copybook_reference",
            "claim_text": "The source references copybook BANK-REC.",
            "claim_scope": "textual_anchor",
            "source": {
                "path": "legacy.cbl",
                "sha256": self.source_sha,
                "spans": [{
                    "line_start": 1,
                    "line_end": 1,
                    "column_start": 8,
                    "column_end": 21,
                    "excerpt_sha256": sha(snippet),
                }],
            },
            "verdict": "supported",
            "witnesses": [{
                "witness_id": "index-output-1",
                "kind": "source_index",
                "artifact_path": "index.json",
                "artifact_sha256": sha(self.evidence_raw),
                "execution_receipt_path": "index-receipt.json",
                "execution_receipt_sha256": sha(self.execution_receipt_raw),
                "verifier_id": "test-source-indexer",
                "verifier_sha256": "d" * 64,
                "observed_result": "supports",
            }],
            "unresolved_reason": None,
        }
        self.claim_path = self.root / "claim.json"
        self.write_claim()
        self.schema_path = Path(audit.__file__).with_name("legacy-claim-evidence.schema.json")
        self.trusted_auditor_sha = sha(Path(audit.__file__).read_bytes())
        self.trusted_schema_sha = sha(self.schema_path.read_bytes())

    def write_claim(self) -> None:
        self.claim_path.write_text(json.dumps(self.claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def validate(self, **kwargs: Any) -> dict[str, Any]:
        self.write_claim()
        return audit.validate_claim(
            self.claim_path,
            self.schema_path,
            self.source_root,
            self.evidence_root,
            expected_claim_sha256=kwargs.pop("expected_claim_sha256", sha(self.claim_path.read_bytes())),
            expected_auditor_sha256=kwargs.pop("expected_auditor_sha256", self.trusted_auditor_sha),
            expected_schema_sha256=kwargs.pop("expected_schema_sha256", self.trusted_schema_sha),
            expected_source_sha256=kwargs.pop("expected_source_sha256", self.source_sha),
            trusted_verifier_pins=kwargs.pop("trusted_verifier_pins", {"source_index": "d" * 64}),
            **kwargs,
        )

    def close(self) -> None:
        self.tmp.cleanup()


class LegacyClaimAuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = ClaimFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def assert_issue(self, result: dict[str, Any], code: str) -> None:
        self.assertFalse(result["evidence_chain_valid"], result)
        self.assertIn(code, {row["code"] for row in result["issues"]}, result)

    def test_complete_source_and_witness_chain_passes(self) -> None:
        result = self.fx.validate()
        self.assertTrue(result["evidence_chain_valid"], result["issues"])
        self.assertEqual(result["claim_verdict"], "supported")

    def test_running_auditor_out_of_band_pin_is_required(self) -> None:
        self.assert_issue(self.fx.validate(expected_auditor_sha256="0" * 64), "auditor_pin_mismatch")

    def test_claim_bytes_must_match_external_pin(self) -> None:
        self.assert_issue(self.fx.validate(expected_claim_sha256="0" * 64), "claim_pin_mismatch")

    def test_schema_bytes_must_match_external_pin(self) -> None:
        self.assert_issue(self.fx.validate(expected_schema_sha256="0" * 64), "schema_pin_mismatch")

    def test_source_bytes_must_match_claim_and_external_pin(self) -> None:
        (self.fx.source_root / "legacy.cbl").write_bytes(b"       COPY OTHER-REC.\n")
        self.assert_issue(self.fx.validate(), "source_digest_mismatch")

    def test_source_excerpt_hash_is_recomputed(self) -> None:
        self.fx.claim["source"]["spans"][0]["excerpt_sha256"] = "e" * 64
        result = self.fx.validate()
        self.assert_issue(result, "source_excerpt_digest_mismatch")

    def test_witness_artifact_bytes_are_rehashed(self) -> None:
        (self.fx.evidence_root / "index.json").write_bytes(b'{"copy_references":[]}\n')
        self.assert_issue(self.fx.validate(), "witness_artifact_digest_mismatch")

    def test_witness_verifier_must_match_out_of_band_allowlist(self) -> None:
        self.assert_issue(
            self.fx.validate(trusted_verifier_pins={"source_index": "9" * 64}),
            "witness_verifier_pin_mismatch",
        )

    def test_runtime_behavior_cannot_use_static_index_as_sole_witness(self) -> None:
        self.fx.claim.update({
            "claim_id": "TX.COMMIT.ORDER",
            "claim_type": "transaction_effect",
            "claim_text": "The transaction commits only after the balance check.",
            "claim_scope": "runtime_behavior",
        })
        self.assert_issue(self.fx.validate(), "runtime_claim_requires_execution")

    def test_path_traversal_is_rejected(self) -> None:
        self.fx.claim["source"]["path"] = "../outside.cbl"
        self.assert_issue(self.fx.validate(), "source_path_invalid")

    def test_conflicting_witnesses_do_not_pass(self) -> None:
        other = b"independent execution trace: counterexample\n"
        (self.fx.evidence_root / "run.log").write_bytes(other)
        self.fx.claim["witnesses"].append({
            "witness_id": "execution-counterexample",
            "kind": "independent_execution",
            "artifact_path": "run.log",
            "artifact_sha256": sha(other),
            "verifier_id": "test-execution-verifier",
            "verifier_sha256": "e" * 64,
            "observed_result": "contradicts",
        })
        # Runtime behavior cannot be verified by the textual witness, and the
        # mixed witnesses must not be reduced to a majority vote.
        self.fx.claim.update({
            "claim_id": "NUM.SIGN",
            "claim_type": "numeric_behavior",
            "claim_text": "Negative amounts preserve their sign after conversion.",
            "claim_scope": "runtime_behavior",
        })
        result = self.fx.validate(trusted_verifier_pins={
            "source_index": "d" * 64,
            "independent_execution": "e" * 64,
        })
        self.assert_issue(result, "witnesses_conflict")


if __name__ == "__main__":
    unittest.main(verbosity=2)
