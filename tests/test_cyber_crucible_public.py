"""Regression tests for the public Cyber Crucible manifest contract."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_cyber_crucible_public.py"
MANIFEST_PATH = ROOT / "validation" / "cyber_crucible_public_v1.json"

SPEC = importlib.util.spec_from_file_location("cyber_crucible_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class PublicCyberCrucibleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_current_manifest_is_consistent(self) -> None:
        self.assertEqual(VALIDATOR.validate_manifest(self.manifest), [])

    def test_changed_scenario_requires_new_digest(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["scenarios"][0]["title"] = "Altered scenario without digest update"
        errors = VALIDATOR.validate_manifest(mutated)
        self.assertTrue(any("digest mismatch" in error for error in errors), errors)

    def test_required_evidence_refs_must_match_oracle(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["scenarios"][0]["oracle"]["required_evidence_refs"].remove("E1")
        errors = VALIDATOR.validate_manifest(mutated)
        self.assertTrue(any("required_evidence_refs must exactly match" in error for error in errors), errors)

    def test_simulated_execution_cannot_authorize_a_pass(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["scoring_contract"]["simulated_execution_authorizes_pass"] = True
        errors = VALIDATOR.validate_manifest(mutated)
        self.assertTrue(any("simulated execution must never authorize a pass" in error for error in errors), errors)

    def test_diagnosis_does_not_grant_live_execution_authority(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["scoring_contract"]["diagnosis_grants_execution_authority"] = True
        errors = VALIDATOR.validate_manifest(mutated)
        self.assertTrue(any("diagnosis must not grant live-system execution authority" in error for error in errors), errors)

    def test_duplicate_scenario_ids_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["scenarios"][1]["scenario_id"] = mutated["scenarios"][0]["scenario_id"]
        errors = VALIDATOR.validate_manifest(mutated)
        self.assertTrue(any("duplicate scenario_id" in error for error in errors), errors)

    def test_two_competing_hypotheses_are_required(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["scenarios"][0]["oracle"]["plausible_alternatives"] = []
        errors = VALIDATOR.validate_manifest(mutated)
        self.assertTrue(any("at least two competing hypotheses" in error for error in errors), errors)

    def test_public_oracle_cannot_be_misrepresented_as_hidden(self) -> None:
        mutated = copy.deepcopy(self.manifest)
        mutated["scenarios"][0]["oracle"]["visibility"] = "hidden"
        errors = VALIDATOR.validate_manifest(mutated)
        self.assertTrue(any("visibility must say public_training explicitly" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
