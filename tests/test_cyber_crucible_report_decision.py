"""Tests for exact report-to-decision binding."""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.verify_cyber_crucible_decision_receipt import (
    POLICY_VERSION,
    SCHEMA_VERSION,
    ReplayLedger,
    canonical_json_digest,
    canonical_signed_bytes,
)
from scripts.verify_cyber_crucible_report_decision import (
    ReportDecisionBindingError,
    validate_input_receipt_binding,
    validate_report_binding,
    verify_report_decision,
)


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class ReportBindingTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "schema_version": "cyber-crucible-diagnostic-loop.v1",
            "lane": "public_calibration",
            "non_authorizing": True,
            "run": {
                "run_id": "attempt-0001",
                "subject_id": "symthaea-test-build",
                "scenario": {
                    "scenario_id": "scenario-12",
                    "revision": 3,
                    "scenario_digest": "1" * 64,
                },
            },
        }
        self.input_receipt = {
            "payload": {
                "campaign_id": "campaign-fixture-1",
                "subject_id": "symthaea-test-build",
                "attempt_id": "attempt-0001",
                "scenario_id": "scenario-12",
                "scenario_revision": 3,
                "scenario_digest": "1" * 64,
            },
            "signature": "fixture-signature",
        }
        self.context = {
            "campaign_id": "campaign-fixture-1",
            "subject_id": "symthaea-test-build",
            "attempt_id": "attempt-0001",
            "scenario_id": "scenario-12",
            "scenario_revision": 3,
            "scenario_digest": "1" * 64,
            "input_receipt_digest": canonical_json_digest(self.input_receipt),
            "report_digest": canonical_json_digest(self.report),
            "evaluator_policy_digest": "3" * 64,
            "verifier_build_digest": "4" * 64,
            "decision_nonce": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
        }

    def test_exact_upstream_receipt_is_bound(self):
        digest = validate_input_receipt_binding(self.input_receipt, self.context)
        self.assertEqual(digest, self.context["input_receipt_digest"])

    def test_modified_upstream_receipt_is_rejected(self):
        changed = deepcopy(self.input_receipt)
        changed["payload"]["subject_id"] = "other-subject"
        with self.assertRaises(ReportDecisionBindingError):
            validate_input_receipt_binding(changed, self.context)

    def test_upstream_receipt_scenario_mismatch_is_rejected(self):
        changed = deepcopy(self.input_receipt)
        changed["payload"]["scenario_revision"] = 4
        context = deepcopy(self.context)
        context["input_receipt_digest"] = canonical_json_digest(changed)
        with self.assertRaises(ReportDecisionBindingError):
            validate_input_receipt_binding(changed, context)

    def test_exact_report_digest_and_run_context_bind(self):
        digest = validate_report_binding(self.report, self.context)
        self.assertEqual(digest, self.context["report_digest"])

    def test_report_tampering_is_rejected(self):
        changed = deepcopy(self.report)
        changed["run"]["subject_id"] = "substituted-subject"
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(changed, self.context)

    def test_wrong_attempt_id_is_rejected(self):
        context = deepcopy(self.context)
        context["attempt_id"] = "different-attempt"
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(self.report, context)

    def test_wrong_scenario_revision_is_rejected(self):
        context = deepcopy(self.context)
        context["scenario_revision"] = 4
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(self.report, context)

    def test_wrong_report_digest_is_rejected(self):
        context = deepcopy(self.context)
        context["report_digest"] = "f" * 64
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(self.report, context)

    def test_authorizing_report_is_rejected(self):
        changed = deepcopy(self.report)
        changed["non_authorizing"] = False
        context = deepcopy(self.context)
        context["report_digest"] = canonical_json_digest(changed)
        with self.assertRaises(ReportDecisionBindingError):
            validate_report_binding(changed, context)

    def test_end_to_end_public_report_decision_binding(self):
        report_path = Path(__file__).resolve().parents[1] / "validation" / "cyber_crucible_diagnostic_loop_public_v1.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run = report["run"]
        scenario = run["scenario"]
        campaign_id = "public-calibration-campaign-v1"
        attempt_id = run["run_id"]

        input_receipt = {
            "payload": {
                "schema_version": "cyber-crucible-signed-receipt.v1",
                "receipt_id": "b" * 32,
                "role": "security_verdict",
                "campaign_id": campaign_id,
                "subject_id": run["subject_id"],
                "attempt_id": attempt_id,
                "scenario_id": scenario["scenario_id"],
                "scenario_revision": scenario["revision"],
                "scenario_digest": scenario["scenario_digest"],
            },
            "signature": b64url(bytes([17]) * 64),
        }

        private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        policy = {
            "schema_version": POLICY_VERSION,
            "policy_id": "public-e2e-decision-policy-v1",
            "keys": [{
                "key_id": "e2e-verifier-key",
                "verifier_id": "e2e-independent-verifier",
                "public_key": b64url(public_key),
                "not_before": "2029-01-01T00:00:00Z",
                "not_after": "2031-01-01T00:00:00Z",
                "revoked_at": None,
            }],
        }
        context = {
            "campaign_id": campaign_id,
            "subject_id": run["subject_id"],
            "attempt_id": attempt_id,
            "scenario_id": scenario["scenario_id"],
            "scenario_revision": scenario["revision"],
            "scenario_digest": scenario["scenario_digest"],
            "input_receipt_digest": canonical_json_digest(input_receipt),
            "report_digest": canonical_json_digest(report),
            "evaluator_policy_digest": "3" * 64,
            "verifier_build_digest": "4" * 64,
            "decision_nonce": b64url(bytes(range(32, 64))),
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "Ed25519",
            "decision_id": "c" * 32,
            "decision_nonce": context["decision_nonce"],
            "signer_key_id": "e2e-verifier-key",
            "verifier_id": "e2e-independent-verifier",
            "decision": "accept",
            "reason_code": "authenticated_under_policy",
            **context,
            "context_digest": canonical_json_digest(context),
            "decision_trust_policy_digest": canonical_json_digest(policy),
            "issued_at": "2030-01-01T00:00:00Z",
            "expires_at": "2030-01-01T00:05:00Z",
            "non_authorizing": True,
        }
        decision_receipt = {
            "payload": payload,
            "signature": b64url(private_key.sign(canonical_signed_bytes(payload))),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = verify_report_decision(
                report,
                input_receipt,
                decision_receipt,
                policy,
                context,
                replay_ledger=ReplayLedger(Path(temp_dir) / "e2e-replay.sqlite3"),
                now=datetime(2030, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
            )
        self.assertTrue(result["report_contract_valid"])
        self.assertTrue(result["decision_receipt_verified"])
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["report_digest"], context["report_digest"])
        self.assertTrue(result["non_authorizing"])


if __name__ == "__main__":
    unittest.main()
