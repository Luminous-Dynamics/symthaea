"""Adversarial tests for the static rubric specification auditor."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
FIXTURE = TOOLS.parent / "fixtures" / "rubric-specification.example.json"
SPEC = importlib.util.spec_from_file_location(
    "rubric_specification_audit", TOOLS / "rubric_specification_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RubricSpecificationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_example_is_structurally_complete_but_non_scoring(self) -> None:
        report = MODULE.audit_rubric_specification(self.fixture)
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["criterion_count"], 4)
        self.assertEqual(report["declared_dimension_count"], 4)
        self.assertEqual(report["hard_constraint_count"], 3)
        self.assertEqual(report["test_count"], 16)
        self.assertEqual(report["uncovered_criterion_count"], 0)
        self.assertTrue(report["hard_constraints_veto_not_average"])
        self.assertFalse(report["empirical_dimensionality_estimated"])
        self.assertFalse(report["construct_validity_proven"])
        self.assertFalse(report["preference_fit_estimated"])
        self.assertFalse(report["adversarial_robustness_proven"])
        self.assertFalse(report["scoring_eligible_declared"])

    def test_each_criterion_has_positive_negative_edge_and_attack_coverage(self) -> None:
        report = MODULE.audit_rubric_specification(self.fixture)
        for row in report["coverage_by_criterion"]:
            self.assertGreaterEqual(row["positive"], 1)
            self.assertGreaterEqual(row["negative"], 1)
            self.assertGreaterEqual(row["edge_case"], 1)
            self.assertGreaterEqual(row["adversarial"], 1)

    def test_missing_adversarial_test_is_rejected(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        criterion = value["criteria"][0]
        removed = criterion["planned_test_ids"].pop()
        value["tests"] = [row for row in value["tests"] if row["test_id"] != removed]
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("lacks adversarial test coverage" in item for item in report["issues"]))

    def test_hard_constraint_requires_a_veto(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["criteria"][0]["veto_condition"] = None
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("requires an explicit veto_condition" in item for item in report["issues"]))

    def test_soft_criterion_cannot_have_veto(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["criteria"][3]["veto_condition"] = "This should not be allowed."
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("must have null veto_condition" in item for item in report["issues"]))

    def test_unknown_dependency_target_is_rejected(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["dependency_rules"][0]["to_criterion_id"] = "missing-criterion"
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("references an unknown criterion" in item for item in report["issues"]))

    def test_duplicate_criterion_id_is_rejected(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["criteria"][1]["criterion_id"] = value["criteria"][0]["criterion_id"]
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("duplicate criterion_id" in item for item in report["issues"]))

    def test_unhashable_hostile_criterion_id_does_not_crash(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["tests"][0]["criterion_ids"] = [["not", "a", "string"]]
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("criterion_ids must be strings" in item for item in report["issues"]))

    def test_unhashable_hostile_test_type_does_not_crash(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["tests"][0]["test_type"] = ["positive"]
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("unsupported test_type" in item for item in report["issues"]))

    def test_non_string_planned_test_id_does_not_crash(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["criteria"][0]["planned_test_ids"][0] = {"unexpected": "object"}
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("planned_test_ids" in item for item in report["issues"]))
    def test_unknown_fields_are_rejected(self) -> None:
        value = json.loads(json.dumps(self.fixture))
        value["claims_to_trust_me"] = True
        report = MODULE.audit_rubric_specification(value)
        self.assertFalse(report["valid"])
        self.assertTrue(any("unknown fields" in item for item in report["issues"]))


if __name__ == "__main__":
    unittest.main()
