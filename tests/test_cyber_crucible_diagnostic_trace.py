"""Fail-closed semantic regressions for the public diagnostic trace contract."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "validation" / "cyber_crucible_diagnostic_trace_v1.json"
MANIFEST_PATH = ROOT / "validation" / "cyber_crucible_public_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "cyber_crucible_diagnostic_trace_validator",
    ROOT / "scripts" / "validate_cyber_crucible_diagnostic_trace.py",
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class DiagnosticTraceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def errors(self, mutate):
        trace = copy.deepcopy(self.trace)
        mutate(trace)
        return VALIDATOR.validate_trace(trace, self.manifest)

    def test_public_trace_is_consistent_but_non_qualifying(self):
        self.assertEqual(VALIDATOR.validate_trace(self.trace, self.manifest), [])
        self.assertEqual(self.trace["classification"], "public_calibration_non_qualifying")
        self.assertFalse(self.trace["report"]["production_authority_granted"])

    def test_stale_scenario_digest_is_rejected(self):
        errors = self.errors(lambda t: t["scenario_binding"].update(scenario_digest="0" * 64))
        self.assertTrue(any("scenario_binding.scenario_digest" in e for e in errors), errors)

    def test_unknown_evidence_reference_is_rejected(self):
        errors = self.errors(lambda t: t["hypotheses"][0]["supporting_evidence_refs"].append("E999"))
        self.assertTrue(any("unknown ID E999" in e for e in errors), errors)

    def test_phase_reordering_is_rejected(self):
        def mutate(t):
            t["phases"][0], t["phases"][1] = t["phases"][1], t["phases"][0]
        errors = self.errors(mutate)
        self.assertTrue(any("canonical order" in e for e in errors), errors)

    def test_synthetic_trace_cannot_claim_completed_operational_observation(self):
        errors = self.errors(lambda t: t["observations"][0].update(outcome="completed"))
        self.assertTrue(any("synthetic traces cannot claim" in e for e in errors), errors)

    def test_unsafe_observation_cannot_complete(self):
        errors = self.errors(lambda t: t["observations"][0].update(impact="unsafe", outcome="completed"))
        self.assertTrue(any("unsafe observation cannot be recorded as completed" in e for e in errors), errors)

    def test_executed_containment_requires_real_authorized_evidenced_action(self):
        def mutate(t):
            t["containment_proposals"][0].update(disposition="executed", execution_evidence_refs=["E1"])
        errors = self.errors(mutate)
        self.assertTrue(any("requires real execution" in e for e in errors), errors)
        self.assertTrue(any("requires approved authority" in e for e in errors), errors)
        self.assertTrue(any("requires authorization_ref" in e for e in errors), errors)
        self.assertTrue(any("execution evidence must have verified integrity" in e for e in errors), errors)

    def test_verified_recovery_requires_every_obligation_to_pass(self):
        errors = self.errors(lambda t: t["recovery"].update(status="verified"))
        self.assertTrue(any("every obligation to pass" in e for e in errors), errors)

    def test_passing_recovery_requires_verified_evidence(self):
        def mutate(t):
            t["recovery"]["obligations"][0].update(status="pass", evidence_refs=["E1"])
            t["recovery"].update(status="in_progress")
        errors = self.errors(mutate)
        self.assertTrue(any("passing recovery evidence must have verified integrity" in e for e in errors), errors)

    def test_attribution_cannot_be_promoted_without_direct_actor_evidence(self):
        def mutate(t):
            t["report"].update(attribution_status="supported")
            t["residual_uncertainty"].update(attribution_status="supported")
        errors = self.errors(mutate)
        self.assertTrue(any("two verified direct_actor_evidence" in e for e in errors), errors)

    def test_root_cause_support_requires_verified_independent_evidence(self):
        def mutate(t):
            t["report"].update(root_cause_status="supported")
            t["hypotheses"][0].update(status="supported", confidence_bps=9500, confidence_calibration="calibrated")
        errors = self.errors(mutate)
        self.assertTrue(any("independently verified supporting evidence" in e for e in errors), errors)

    def test_malformed_top_level_input_fails_closed_without_exception(self):
        self.assertTrue(VALIDATOR.validate_trace([], self.manifest))
        self.assertTrue(VALIDATOR.validate_trace(self.trace, []))

    def test_malformed_nested_fields_fail_closed_without_exception(self):
        def mutate_impact(t):
            t["observations"][0]["impact"] = []
        def mutate_disposition(t):
            t["containment_proposals"][0]["disposition"] = []
        def mutate_confidence(t):
            t["report"]["root_cause_status"] = "supported"
            t["hypotheses"][0].update(status="supported", confidence_bps=[], confidence_calibration="calibrated")
        for mutate in (mutate_impact, mutate_disposition, mutate_confidence):
            with self.subTest(mutate=mutate.__name__):
                self.assertTrue(VALIDATOR.validate_trace(self._mutated(mutate), self.manifest))

    def _mutated(self, mutate):
        trace = copy.deepcopy(self.trace)
        mutate(trace)
        return trace


if __name__ == "__main__":
    unittest.main()
