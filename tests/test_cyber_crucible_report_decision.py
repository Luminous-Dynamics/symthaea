"""Tests for exact report-to-decision binding."""
from __future__ import annotations

from copy import deepcopy
import unittest

from scripts.verify_cyber_crucible_decision_receipt import canonical_json_digest
from scripts.verify_cyber_crucible_report_decision import (
    ReportDecisionBindingError,
    validate_report_binding,
)


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
        self.context = {
            "campaign_id": "campaign-fixture-1",
            "subject_id": "symthaea-test-build",
            "attempt_id": "attempt-0001",
            "scenario_id": "scenario-12",
            "scenario_revision": 3,
            "scenario_digest": "1" * 64,
            "input_receipt_digest": "2" * 64,
            "report_digest": canonical_json_digest(self.report),
            "evaluator_policy_digest": "3" * 64,
            "verifier_build_digest": "4" * 64,
            "decision_nonce": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
        }

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


if __name__ == "__main__":
    unittest.main()
