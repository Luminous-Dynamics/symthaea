"""Tests for the Cyber Crucible diagnostic-loop report contract."""
from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_cyber_crucible_diagnostic_loop.py"
REPORT_PATH = ROOT / "validation" / "cyber_crucible_diagnostic_loop_public_v1.json"

SPEC = importlib.util.spec_from_file_location("cyber_crucible_diagnostic_loop_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class DiagnosticLoopContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def changed(self):
        return copy.deepcopy(self.report)

    def redigest_receipt(self, receipt):
        receipt["receipt_digest"] = VALIDATOR.content_digest(receipt, "receipt_digest")

    def redigest_evidence(self, evidence):
        evidence["evidence_digest"] = VALIDATOR.content_digest(evidence, "evidence_digest")

    def test_public_fixture_is_internally_consistent(self):
        self.assertEqual(VALIDATOR.validate_report(self.report), [])

    def test_evidence_digest_binds_payload_content(self):
        report = self.changed()
        report["evidence"][0]["payload"]["tls_result"] = "success"
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("evidence_digest does not bind" in error for error in errors), errors)

    def test_receipt_digest_binds_actual_receipt_content(self):
        report = self.changed()
        report["evaluator_receipts"][1]["payload"]["finding"] = "unconditionally safe"
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("receipt_digest does not bind" in error for error in errors), errors)

    def test_receipt_must_bind_exact_scenario_identity(self):
        report = self.changed()
        report["evaluator_receipts"][1]["scenario_digest"] = "0" * 64
        self.redigest_receipt(report["evaluator_receipts"][1])
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("scenario identity does not match" in error for error in errors), errors)

    def test_diagnostic_loop_requires_every_phase(self):
        report = self.changed()
        report["steps"] = [step for step in report["steps"] if step["phase"] != "recovery"]
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("missing phase recovery" in error for error in errors), errors)

    def test_step_sequence_must_be_contiguous(self):
        report = self.changed()
        report["steps"][4]["sequence"] = 9
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("sequence must be unique, ordered, and contiguous" in error for error in errors), errors)

    def test_next_observation_requires_explicit_bounded_scope_and_budget(self):
        report = self.changed()
        del report["steps"][4]["observation_budget"]
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("observation_budget" in error for error in errors), errors)

    def test_competing_hypotheses_cannot_collapse_to_one(self):
        report = self.changed()
        report["hypotheses"] = report["hypotheses"][:1]
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("at least two competing hypotheses" in error for error in errors), errors)

    def test_live_containment_cannot_be_marked_executed_without_authority_receipt(self):
        report = self.changed()
        report["proposals"][0]["status"] = "executed"
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("requires a matching authorization_decision receipt" in error for error in errors), errors)

    def test_authorized_status_requires_receipt(self):
        report = self.changed()
        authority_step = next(step for step in report["steps"] if step["phase"] == "authority_check")
        authority_step["authority_status"] = "authorized"
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("authority_status=authorized requires" in error for error in errors), errors)

    def test_functional_and_security_verdicts_must_be_independent(self):
        report = self.changed()
        report["evaluator_receipts"][1]["evaluator_id"] = report["evaluator_receipts"][0]["evaluator_id"]
        self.redigest_receipt(report["evaluator_receipts"][1])
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("must come from distinct evaluators" in error for error in errors), errors)

    def test_public_simulated_fixture_cannot_claim_correct_and_secure(self):
        report = self.changed()
        report["assessment"]["correct_and_secure"] = True
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("recomputed conjunction" in error for error in errors), errors)

    def test_every_supported_claim_requires_evidence(self):
        report = self.changed()
        report["final_claims"][0]["evidence_refs"] = []
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("supported claim must cite evidence" in error for error in errors), errors)

    def test_malformed_untrusted_values_fail_closed_without_traceback(self):
        report = self.changed()
        report["run"]["execution_mode"] = []
        report["evaluator_receipts"][0]["status"] = []
        report["proposals"][0]["status"] = {}
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("execution_mode is unsupported" in error for error in errors), errors)
        self.assertTrue(any("status is unsupported" in error for error in errors), errors)

    def test_unknown_evidence_reference_is_rejected(self):
        report = self.changed()
        report["steps"][0]["evidence_refs"] = ["E-NOT-PRESENT"]
        errors = VALIDATOR.validate_report(report)
        self.assertTrue(any("unknown reference 'E-NOT-PRESENT'" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
